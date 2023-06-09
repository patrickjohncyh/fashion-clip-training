try:
    import dotenv

    dotenv.load_dotenv(".env")
except:
    print("dotenv package missing")

import os
from metaflow import (
    FlowSpec,
    step,
    batch,
    current,
    environment,
    S3,
    Parameter,
    parallel_map,
    environment,
    retry,
)
from custom_decorators import pip


class FashionCLIPFlow(FlowSpec):
    dataset_path = Parameter(
        name="dataset_path", help="Path to dataset", default="data/dataset_v1.parquet"
    )

    @step
    def start(self):
        print("flow name: %s" % current.flow_name)
        print("run id: %s" % current.run_id)
        print("username: %s" % current.username)

        self.next(self.load_dataset_info)

    @step
    def load_dataset_info(self):
        import numpy as np
        import pandas as pd

        self.raw_dataset = pd.read_parquet(self.dataset_path)

        # debug
        print(self.raw_dataset.dtypes)
        print(self.raw_dataset.shape)
        # with pd.option_context('display.max_colwidth', None):
        print(self.raw_dataset[["product_id", "product_photo_url"]].head(3))

        # get relevant columns
        self.df = self.raw_dataset[
            ["product_id", "short_description", "product_photo_url"]
        ]
        # create image filename
        self.df["image_fname"] = self.df["product_id"].apply(lambda _: _ + ".jpg")
        self.df_splits = np.array_split(self.df, 4)
        self.next(self.download_data, foreach="df_splits")

    @retry
    @pip(libraries={"tqdm": "4.62.3"})
    @batch(cpu=16, memory=16000)
    @step
    def download_data(self):
        import os
        import numpy as np
        import pandas as pd
        from utils import make_tar, download_image
        import hashlib
        from tqdm.auto import tqdm
        from tqdm.contrib.logging import tqdm_logging_redirect
        import logging
        import sys
        from multiprocessing import cpu_count

        print("THERE ARE {} CPU CORES AVAILABLE".format(cpu_count()))

        # https://stackoverflow.com/questions/68225881/how-to-show-tqdm-progress-in-metaflow
        logging.basicConfig(level=logging.INFO, stream=sys.stdout)

        # enable progress bar for pandas apply
        tqdm.pandas(leave=False, mininterval=10)

        # load df from for-each
        df = self.input
        split_hash = hashlib.sha1(pd.util.hash_pandas_object(df).values).hexdigest()

        # create directory
        self.imgs_output_dir = "data/imgs"
        os.makedirs(self.imgs_output_dir, exist_ok=True)

        # download in parallel across cores
        with tqdm_logging_redirect():
            parallel_map(
                func=lambda df: df.progress_apply(
                    func=lambda row: download_image(
                        url=row["product_photo_url"],
                        fname=row["image_fname"],
                        output_dir=self.imgs_output_dir,
                        retry=10,
                    ),
                    axis=1,
                ),
                iterable=np.array_split(df, cpu_count() - 1),
            )

        # zip images and store in S3
        self.tar_name = "fclip_images_{}.tar.gz".format(split_hash)
        make_tar(self.tar_name, ".", self.imgs_output_dir)
        # upload to metaflow datastore
        with open(self.tar_name, "rb") as in_file:
            data = in_file.read()
            with S3(run=self) as s3:
                url = s3.put(self.tar_name, data)
                print("Images saved at: {}".format(url))
                # save this path for downstream reference!
                self.images_s3_url = url
                os.remove(self.tar_name)

        self.next(self.merge_downloads)

    @environment(vars={"BASE_IMAGE": os.getenv("BASE_IMAGE")})
    @batch(cpu=4, memory=50000, image=os.getenv("BASE_IMAGE"), shared_memory=50000)
    @step
    def merge_downloads(self, inputs):
        import os
        from utils import make_tar, download_and_decompress_gz_from_s3

        self.merge_artifacts(
            inputs, exclude=["imgs_output_dir", "tar_name", "images_s3_url"]
        )

        ramdisk_path = "/dev/shm"
        self.tar_name = "fclip_images.tar.gz"
        self.imgs_output_dir = "data/imgs"
        self.images_chunk_s3_url = [input.images_s3_url for input in inputs]

        print(self.images_chunk_s3_url)
        for url in self.images_chunk_s3_url:
            with S3() as s3:
                download_and_decompress_gz_from_s3(s3, url, output_path=ramdisk_path)
        files = os.listdir(os.path.join(ramdisk_path, self.imgs_output_dir))
        print("THERE ARE {} IMAGES".format(len(files)))
        make_tar(self.tar_name, ramdisk_path, self.imgs_output_dir)
        print("UPLOADING TO S3... ")
        with open(self.tar_name, "rb") as in_file:
            data = in_file.read()
            with S3(run=self) as s3:
                url = s3.put(self.tar_name, data)
                print("Images saved at: {}".format(url))
                # save this path for downstream reference!
                self.images_s3_url = url
                os.remove(self.tar_name)
        self.next(self.prepare_clip_df)

    @environment(
        vars={
            "COMET_API_KEY": os.getenv("COMET_API_KEY"),
            "BASE_IMAGE": os.getenv("BASE_IMAGE"),
            "DATASET_SEED": os.getenv("DATASET_SEED"),
            "TEST_SET_SIZE": os.getenv("TEST_SET_SIZE"),
            "VALID_SET_SIZE": os.getenv("VALID_SET_SIZE"),
        }
    )
    @batch(cpu=8, memory=50000, image=os.getenv("BASE_IMAGE"), shared_memory=20000)
    @step
    def prepare_clip_df(self):
        import os
        import pandas as pd
        import random
        from utils import download_and_decompress_gz_from_s3
        from model_utils import clean_description, build_caption

        # use ramdisk for extra diskspace
        ramdisk_path = "/dev/shm"
        # download images from S3
        with S3() as s3:
            download_and_decompress_gz_from_s3(
                s3, self.images_s3_url, output_path=ramdisk_path
            )

        # helper to generate image path
        get_image_path = lambda row: os.path.join(
            ramdisk_path, self.imgs_output_dir, row["image_fname"]
        )
        # build dataframe for clip
        self.clip_df = self.raw_dataset.copy()
        print(self.clip_df.dtypes)
        random.seed(os.getenv("DATASET_SEED"))
        self.clip_df["image_fname"] = self.clip_df["product_id"].apply(
            lambda _: _ + ".jpg"
        )
        self.clip_df["caption"] = self.clip_df.apply(build_caption, axis=1)
        self.clip_df["caption"] = self.clip_df["caption"].apply(lambda _: str(_))

        # build training df for CLIPTuner; drop rows w/o images
        self.clip_df = pd.DataFrame(
            [
                {"image": get_image_path(row), **row}
                for _, row in self.clip_df.iterrows()
                if os.path.exists(get_image_path(row))
            ]
        )

        self.zero_shot = {
            "Denim": self.clip_df[self.clip_df["category_level_2"] == "Denim"],
            "Performance Tops": self.clip_df[
                self.clip_df["category_level_2"] == "Performance Tops"
            ],
            "Kurt Geiger": self.clip_df[self.clip_df["brand"] == "kurt geiger london"],
            "Canali": self.clip_df[self.clip_df["brand"] == "canali"],
        }

        # self.clip_df_train_test = self.clip_df[(self.clip_df['category_level_2'] != 'Denim') &
        #                                        (self.clip_df['category_level_2'] != 'Performance Tops') &
        #                                        (self.clip_df['brand'] != 'kurt geiger london') &
        #                                        (self.clip_df['brand'] != 'canali')]

        self.clip_df_train_test = self.clip_df

        # shuffle
        self.clip_df_train_test = self.clip_df_train_test.sample(
            frac=1, random_state=int(os.getenv("DATASET_SEED"))
        ).reset_index(drop=True)

        # check descriptions
        print(self.clip_df_train_test.tail(10)["caption"])

        TEST_SET_SIZE = int(os.getenv("TEST_SET_SIZE"))
        VALID_SET_SIZE = int(os.getenv("VALID_SET_SIZE"))

        self.test_df = self.clip_df_train_test[:TEST_SET_SIZE]
        self.valid_df = self.clip_df_train_test[
            TEST_SET_SIZE : TEST_SET_SIZE + VALID_SET_SIZE
        ]
        self.train_df = self.clip_df_train_test[VALID_SET_SIZE + TEST_SET_SIZE :]

        # data check
        assert not set(self.train_df["product_id"]) & set(self.test_df["product_id"])
        assert not set(self.train_df["product_id"]) & set(self.valid_df["product_id"])
        assert not set(self.valid_df["product_id"]) & set(self.test_df["product_id"])

        self.next(self.launch_training)

    @step
    def launch_training(self):
        from itertools import product

        self.learning_rates = [1e-6]
        self.optimizers = ["adamw"]
        self.hyper_param_configs = [
            {"lr": lr, "opt": opt}
            for lr, opt in list(product(self.learning_rates, self.optimizers))
        ]

        self.next(self.train_fclip, foreach="hyper_param_configs")

    @environment(
        vars={
            "WANDB_API_KEY": os.getenv("WANDB_API_KEY"),
            "BASE_IMAGE": os.getenv("BASE_IMAGE"),
        }
    )
    @batch(gpu=4, memory=180000, image=os.getenv("BASE_IMAGE"), shared_memory=90000)
    @pip(
        libraries={
            "wandb": "",
            "pandas":"",
            "open_clip_torch" : "2.20.0",
            "braceexpand" : "",
            "webdataset" : "0.2.5",
            "regex" : "",
            "ftfy" : "",
            "tqdm" : "",
            "huggingface_hub" : "",
            "transformers": "",
            "timm": "",
            "fsspec": ""
        }
    )
    @step
    def train_fclip(self):
        from metaflow import current
        import os
        import wandb
        from utils import download_and_decompress_gz_from_s3

        wandb.login(key=os.getenv("WANDB_API_KEY"))

        ramdisk_path = "/dev/shm"
        with S3() as s3:
            download_and_decompress_gz_from_s3(
                s3, self.images_s3_url, output_path=ramdisk_path
            )

        print("TRAIN SET SIZE: {}".format(self.train_df.shape))
        print("VALID SET SIZE: {}".format(self.valid_df.shape))
        print("TEST SET SIZE: {}".format(self.test_df.shape))

        # save image_path, caption to csv
        self.train_df[['image','caption']].to_csv('train_FF.csv')
        self.test_df[['image','caption']].to_csv('test_FF.csv')
        self.valid_df[['image','caption']].to_csv('valid_FF.csv')
        

    
        self.hp_config = self.input
        print("USING OPTIMIZER : {}".format(self.hp_config["opt"]))
        print("USING LR : {}".format(self.hp_config["lr"]))
        
        os.system('torchrun \
                  --nproc_per_node 4 \
                  -m training.main \
                  --save-frequency 1 \
                  --zeroshot-frequency 0 \
                  --report-to wandb \
                  --wandb-project-name FashionCLIP \
                  --train-data="./train_FF.csv" \
                  --val-data="./test_FF.csv" \
                  --csv-img-key image \
                  --csv-caption-key caption \
                  --warmup 50 \
                  --batch-size=16 \
                  --lr=1e-6  \
                  --wd=0.1 \
                  --epochs=1 \
                  --workers=8 \
                  --csv-separator="," \
                  --pretrained "laion2b_s32b_b82k" \
                  --model ViT-L-14  \
                  --name "FashionCLIP_L_14_{}" \
                  --local-loss \
                  --gather-with-grad'.format(current.run_id))
        
        # best_eval_step = sorted(self.state_dicts, key=lambda k: self.state_dicts[k]['validation_loss'])[0]
        # self.best_model = self.state_dicts[best_eval_step]['state_dict']
        self.next(self.join_training_results)

    @environment(vars={"BASE_IMAGE": os.getenv("BASE_IMAGE")})
    @batch(cpu=8, memory=60000, image=os.getenv("BASE_IMAGE"))
    @step
    def join_training_results(self, inputs):
        import gc

        print("Joining Results")
        self.train_results = {
            "_".join([inp.hp_config["opt"], str(inp.hp_config["lr"])]): {
                "state_dicts": inp.state_dicts,
                "url": inp.exp_url,
                "lr": inp.hp_config["lr"],
            }
            for inp in inputs
        }
        print("Merge Artifacts")
        for inp in inputs:
            del inp.state_dicts
            gc.collect()

        state_dicts = list(self.train_results.values())[0]["state_dicts"]
        best_eval_step = sorted(
            state_dicts, key=lambda k: state_dicts[k]["validation_loss"]
        )[0]
        self.best_model = state_dicts[best_eval_step]["state_dict"]

        self.merge_artifacts(inputs, exclude=["state_dicts", "hp_config", "exp_url"])
        self.next(self.end)

    @step
    def end(self):
        import torch
        import pickle

        with open("train_results_laion_L_14.pickle", "wb") as f:
            print(self.train_results.keys())
            pickle.dump(self.train_results, f)

        # print(state_dicts['adamw_1e-05']['state_dicts'.keys())
        # torch.save(self.best_model,'state_dict.pt')
        print("DAG ended!")


if __name__ == "__main__":
    FashionCLIPFlow()
