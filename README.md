# Setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |
| CUDA   | 11.2    |

### Instructions

To run the project in a Python virtual environment, run the following commands:

```bash
python -m venv venv
```

If you have an NVIDIA GPU, you can install the CUDA toolkit and cuDNN to enable GPU support. If you don't have an NVIDIA GPU, the following steps.

- Download and install [CUDA 11.2 for Windows](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64)
- Download and install [Visual Studio 2019](https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#--download) with C++ build tools
    - In the Visual Studio Installer, under Workloads, select Desktop development with C++
    - **or** Under Individual components, select MSVC v142 - VS 2019 C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*
    - Proceed with the installation
- Download [cuDNN v8.1.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-windows-x64-v8.1.0.77.zip) (
  direct download) or any compatible version
    - Create an NVIDIA developer account if you don't have one

Install the dependencies:

```bash
pip install -r requirements.txt
```

Run the setup script:

```bash
python setup.py
```

### Resources

- [Tensorflow 2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

# Notes

> Note: GPU support on native-Windows is only available for 2.10 or earlier versions, starting in TF 2.11, CUDA build is not supported for Windows. For using
> TensorFlow GPU on Windows, you will need to build/install TensorFlow in WSL2 or use tensorflow-cpu with TensorFlow-DirectML-Plugin

- [Windows GPU build configurations](https://www.tensorflow.org/install/source_windows#gpu)
- [Windows CPU build configurations](https://www.tensorflow.org/install/source_windows#cpu)

### October 4th, 2023 pip list

```
Package                       Version
----------------------------- -------------------
absl-py                       0.12.0
array-record                  0.4.1
astunparse                    1.6.3
backcall                      0.2.0
bleach                        6.0.0
cachetools                    4.2.2
certifi                       2021.5.30
chardet                       4.0.0
charset-normalizer            3.2.0
click                         8.1.7
colorama                      0.4.6
contourpy                     1.1.1
cycler                        0.10.0
Cython                        3.0.2
decorator                     5.0.9
dm-tree                       0.1.8
etils                         1.5.0
flatbuffers                   1.12
fonttools                     4.43.0
fsspec                        2023.9.2
gast                          0.4.0
gin-config                    0.5.0
google-api-core               2.12.0
google-api-python-client      2.102.0
google-auth                   1.32.1
google-auth-httplib2          0.1.1
google-auth-oauthlib          0.4.4
google-cloud-bigquery         3.12.0
google-cloud-core             2.3.3
google-crc32c                 1.5.0
google-pasta                  0.2.0
google-resumable-media        2.6.0
googleapis-common-protos      1.60.0
grpcio                        1.34.1
grpcio-status                 1.48.2
h5py                          3.1.0
httplib2                      0.22.0
idna                          2.10
immutabledict                 3.0.0
importlib-metadata            3.10.1
importlib-resources           6.1.0
joblib                        1.3.2
kaggle                        1.5.16
keras                         2.14.0
keras-nightly                 2.5.0.dev2021032900
Keras-Preprocessing           1.1.2
kiwisolver                    1.3.1
libclang                      16.0.6
lxml                          4.9.3
Markdown                      3.3.4
MarkupSafe                    2.1.3
matplotlib                    3.8.0
ml-dtypes                     0.2.0
numpy                         1.19.5
oauth2client                  4.1.3
oauthlib                      3.1.1
opencv-python                 4.8.1.78
opencv-python-headless        4.8.1.78
opt-einsum                    3.3.0
packaging                     23.2
pandas                        1.2.3
Pillow                        8.2.0
pip                           22.3.1
portalocker                   2.8.2
promise                       2.3
proto-plus                    1.22.3
protobuf                      3.15.7
psutil                        5.9.5
py-cpuinfo                    9.0.0
pyasn1                        0.4.8
pyasn1-modules                0.2.8
pycocotools                   2.0
pyparsing                     2.4.7
python-dateutil               2.8.1
python-slugify                8.0.1
pytz                          2021.1
pywin32                       306
PyYAML                        5.4.1
regex                         2023.10.3
requests                      2.25.1
requests-oauthlib             1.3.0
rsa                           4.7.2
sacrebleu                     2.2.0
scikit-learn                  1.3.1
scipy                         1.6.2
sentencepiece                 0.1.99
seqeval                       1.2.2
setuptools                    65.5.1
six                           1.15.0
tabulate                      0.9.0
tensorboard                   2.11.2
tensorboard-data-server       0.6.1
tensorboard-plugin-wit        1.8.0
tensorflow                    2.5.0
tensorflow-addons             0.21.0
tensorflow-datasets           4.9.0
tensorflow-estimator          2.5.0
tensorflow-hub                0.14.0
tensorflow-intel              2.14.0
tensorflow-io-gcs-filesystem  0.31.0
tensorflow-metadata           1.13.0
tensorflow-model-optimization 0.7.5
tensorflow-text               2.10.0
termcolor                     1.1.0
text-unidecode                1.3
tf-models-official            2.5.0
tf-slim                       1.1.0
threadpoolctl                 2.1.0
toml                          0.10.2
tqdm                          4.66.1
typeguard                     2.13.3
typing-extensions             3.7.4.3
uritemplate                   4.1.1
urllib3                       1.26.5
webencodings                  0.5.1
Werkzeug                      2.0.1
wget                          3.2
wheel                         0.41.2
wrapt                         1.12.1
zipp                          3.4.1
```
