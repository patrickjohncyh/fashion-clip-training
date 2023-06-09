import io
import os
import tarfile
import requests

def load_envs(fname):
    try:
        import dotenv
        return dotenv.dotenv_values(fname)
    except:
        print('dotenv package missing')
        return {}

def make_tar(fname, parent_dir, source_dir):
    with tarfile.open(fname, "w:gz") as tar:
        tar.add(os.path.join(parent_dir, source_dir), source_dir)

def download_image(url, fname, output_dir, retry=1):
        """
        Download a single image
        """
        outpath = os.path.join(output_dir, fname)
        if not os.path.exists(outpath):
            tries = 0
            success = False
            while tries < retry and not success:
                try:
                    r = requests.get(url)
                    if r.status_code == 200:
                        with open(outpath, 'wb') as f:
                            f.write(r.content)
                    elif r.status_code == 404:
                        print('WARNING: Download for {} failed; url = {}'.format(fname, url))
                    else:
                        raise ValueError
                    success = True
                except Exception as e:
                    print('Attempt {}/{}'.format(tries+1, retry), e)
                    tries+=1
        return None


def download_and_decompress_gz_from_s3(s3_client, s3_url, output_path='.'):
        s3_object = s3_client.get(s3_url)
        fileobj = io.BytesIO(s3_object.blob)
        with tarfile.open(fileobj=fileobj) as t:
            t.extractall(path=output_path)
        del fileobj

