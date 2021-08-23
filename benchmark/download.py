from __future__ import print_function
import hashlib
import os
import sys
import tarfile
import zipfile
import requests

from urllib.request import urlopen


class Dataset:
    MB = 1024*1024
    BUFSIZE = 10*MB

    def __init__(self, **kwargs):
        self.name = kwargs.pop('name')
        self.url = kwargs.pop('url', None)
        self.downloader = kwargs.pop('downloader', None)
        self.filename = kwargs.pop('filename')
        self.sha = kwargs.pop('sha', None)
        self.archive = kwargs.pop('archive', None)
        self.member = kwargs.pop('member', None)
        self.decompress = kwargs.pop('decompress', None)

    def __str__(self):
        return 'Dataset <{}>'.format(self.name)

    def printRequest(self, r):
        def getMB(r):
            d = dict(r.info())
            for c in ['content-length', 'Content-Length']:
                if c in d:
                    return int(d[c]) / self.MB
            return '<unknown>'
        print('  {} {} [{} Mb]'.format(r.getcode(), r.msg, getMB(r)))

    def verifyHash(self):
        if not self.sha:
            return False
        print('  expect {}'.format(self.sha))
        sha = hashlib.sha1()
        try:
            with open(self.filename, 'rb') as f:
                while True:
                    buf = f.read(self.BUFSIZE)
                    if not buf:
                        break
                    sha.update(buf)
            print('  actual {}'.format(sha.hexdigest()))
            return self.sha == sha.hexdigest()
        except Exception as e:
            print('  catch {}'.format(e))

    def get(self):
        if self.verifyHash():
            print('  hash match - skipping download')
            if self.decompress:
                print('  extracting')
                self.extract_all()
                print('  extracting done')
            return True

        basedir = os.path.dirname(self.filename)
        if basedir and not os.path.exists(basedir):
            print('  creating directory: ' + basedir)
            os.makedirs(basedir, exist_ok=True)

        if self.archive and self.member:
            assert(self.archive and self.member)
            print('  hash check failed - extracting')
            print('  get {}'.format(self.member))
            self.extract()
        elif self.url:
            print('  hash check failed - downloading')
            print('  get {}'.format(self.url))
            self.download()
        else:
            assert self.downloader
            print('  hash check failed - downloading')
            sz = self.downloader(self.filename)
            print('  size = %.2f Mb' % (sz / (1024.0 * 1024)))

        print(' done')
        print(' file {}'.format(self.filename))
        if self.verifyHash() and self.decompress:
            print('  hash match - extracting')
            self.extract_all()
            print('  extracting done')

        return True

    def download(self):
        try:
            r = urlopen(self.url, timeout=60)
            self.printRequest(r)
            self.save(r)
        except Exception as e:
            print('  catch {}'.format(e))

    def extract(self):
        try:
            with tarfile.open(self.archive) as f:
                assert self.member in f.getnames()
                self.save(f.extractfile(self.member))
        except Exception as e:
            print('  catch {}'.format(e))

    def extract_all(self):
        dst = '/'.join(self.filename.split('/')[:-1])
        try:
            if self.filename.endswith('.zip'):
                with zipfile.ZipFile(self.filename) as f:
                    f.extractall(path=dst)
            else:
                with tarfile.open(self.filename) as f:
                    f.extractall(path=dst)
        except Exception as e:
            print(('  catch {}'.format(e)))

    def save(self, r):
        with open(self.filename, 'wb') as f:
            print('  progress ', end='')
            sys.stdout.flush()
            while True:
                buf = r.read(self.BUFSIZE)
                if not buf:
                    break
                f.write(buf)
                print('>', end='')
                sys.stdout.flush()


def GDrive(gid):
    def download_gdrive(dst):
        session = requests.Session()  # re-use cookies

        URL = "https://docs.google.com/uc?export=download"
        response = session.get(URL, params = { 'id' : gid }, stream = True)

        def get_confirm_token(response):  # in case of large files
            for key, value in response.cookies.items():
                if key.startswith('download_warning'):
                    return value
            return None
        token = get_confirm_token(response)

        if token:
            params = { 'id' : gid, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        BUFSIZE = 1024 * 1024
        PROGRESS_SIZE = 10 * 1024 * 1024

        sz = 0
        progress_sz = PROGRESS_SIZE
        with open(dst, "wb") as f:
            for chunk in response.iter_content(BUFSIZE):
                if not chunk:
                    continue  # keep-alive

                f.write(chunk)
                sz += len(chunk)
                if sz >= progress_sz:
                    progress_sz += PROGRESS_SIZE
                    print('>', end='')
                    sys.stdout.flush()
        print('')
        return sz
    return download_gdrive

datasets = [
    Dataset(
        name='WIDER Face',
        downloader=GDrive('0B6eKvaijfFUDd3dIRmpvSk8tLUk'),
        sha='3643b3045a491b402b46a22e5ccfe1fdcf3d6c68',
        filename='datasets/face_detection_widerface/data/WIDER_val.zip',
        decompress=True
    ),
    Dataset(
        name='WIDER Face',
        url='http://shuoyang1213.me/WIDERFACE/support/eval_script/eval_tools.zip',
        sha='a62215ee44ec86c5916176ccc07980a852af9118',
        filename='datasets/face_detection_widerface/data/eval_tools.zip',
        decompress=True
    ),
]

if __name__ == '__main__':
    selected_dataset_name = None
    if len(sys.argv) > 1:
        selected_dataset_name = sys.argv[1]
        print('Dataset: ' + selected_dataset_name)

    failedDatasets = []
    for d in datasets:
        print(d)
        if selected_dataset_name is not None and not d.name.startswith(selected_dataset_name):
            continue
        if not d.get():
            failedDatasets.append(d.filename)

    if failedDatasets:
        print("Following datasets have not been downloaded:")
        for f in failedDatasets:
            print("* {}".format(f))
        exit(15)