# notice

this code does not work!

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

- Label studio for labeling images: <https://labelstud.io/>
- [Tensorflow 2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

# Notes

> Note: GPU support on native-Windows is only available for 2.10 or earlier versions, starting in TF 2.11, CUDA build is not supported for Windows. For using
> TensorFlow GPU on Windows, you will need to build/install TensorFlow in WSL2 or use tensorflow-cpu with TensorFlow-DirectML-Plugin

- [Windows GPU build configurations](https://www.tensorflow.org/install/source_windows#gpu)
- [Windows CPU build configurations](https://www.tensorflow.org/install/source_windows#cpu)

## Random debugging notes

- missing pipeline.config file
    - > Copy the one from the training folder
    - delete `fine_tune_checkpoint_version: V2` from `pipeline.config`

### October 5th, 2023 pip list (working)

```
Package                       Version
----------------------------- -------------------
absl-py                         0.12.0
anyio                           4.0.0
argon2-cffi                     23.1.0
argon2-cffi-bindings            21.2.0
arrow                           1.3.0
asttokens                       2.4.0
astunparse                      1.6.3
async-lru                       2.0.4
attrs                           23.1.0
Babel                           2.13.0
backcall                        0.2.0
beautifulsoup4                  4.12.2
bleach                          6.0.0
cachetools                      4.2.2
certifi                         2021.5.30
cffi                            1.16.0
chardet                         4.0.0
charset-normalizer              3.3.0
colorama                        0.4.6
comm                            0.1.4
contextlib2                     21.6.0
contourpy                       1.1.1
cycler                          0.10.0
Cython                          3.0.3
debugpy                         1.8.0
decorator                       5.0.9
defusedxml                      0.7.1
docutils                        0.20.1
exceptiongroup                  1.1.3
executing                       2.0.0
fastjsonschema                  2.18.1
flatbuffers                     1.12
fonttools                       4.43.0
fqdn                            1.5.1
gast                            0.4.0
google-auth                     1.32.1
google-auth-oauthlib            0.4.4
google-pasta                    0.2.0
grpcio                          1.34.1
h5py                            3.1.0
idna                            2.10
importlib-metadata              3.10.1
importlib-resources             6.1.0
ipykernel                       6.25.2
ipython                         8.16.1
ipython-genutils                0.2.0
ipywidgets                      8.1.1
isoduration                     20.11.0
jaraco.classes                  3.3.0
jedi                            0.19.1
Jinja2                          3.1.2
json5                           0.9.14
jsonpointer                     2.4
jsonschema                      4.19.1
jsonschema-specifications       2023.7.1
jupyter                         1.0.0
jupyter_client                  8.3.1
jupyter-console                 6.6.3
jupyter_core                    5.3.2
jupyter-events                  0.7.0
jupyter-lsp                     2.2.0
jupyter_server                  2.7.3
jupyter_server_terminals        0.4.4
jupyterlab                      4.0.6
jupyterlab-pygments             0.2.2
jupyterlab_server               2.25.0
jupyterlab-widgets              3.0.9
keras                           2.14.0
keras-nightly                   2.5.0.dev2021032900
Keras-Preprocessing             1.1.2
keyring                         24.2.0
kiwisolver                      1.3.1
libclang                        16.0.6
lxml                            4.9.3
Markdown                        3.3.4
markdown-it-py                  3.0.0
MarkupSafe                      2.1.3
matplotlib                      3.8.0
matplotlib-inline               0.1.6
mdurl                           0.1.2
mistune                         3.0.2
ml-dtypes                       0.2.0
more-itertools                  10.1.0
nbclient                        0.8.0
nbconvert                       7.9.2
nbformat                        5.9.2
nest-asyncio                    1.5.8
nh3                             0.2.14
notebook                        7.0.4
notebook_shim                   0.2.3
numpy                           1.19.5
oauthlib                        3.1.1
opencv-python                   4.8.1.78
opt-einsum                      3.3.0
overrides                       7.4.0
packaging                       23.2
pandas                          1.2.3
pandocfilters                   1.5.0
parso                           0.8.3
pickleshare                     0.7.5
Pillow                          8.2.0
pip                             23.2.1
pkginfo                         1.9.6
platformdirs                    3.11.0
prometheus-client               0.17.1
prompt-toolkit                  3.0.39
protobuf                        3.20.3
psutil                          5.9.5
pure-eval                       0.2.2
pyasn1                          0.4.8
pyasn1-modules                  0.2.8
pycparser                       2.21
Pygments                        2.16.1
pyparsing                       2.4.7
python-dateutil                 2.8.1
python-json-logger              2.0.7
pytz                            2021.1
pywin32                         306
pywin32-ctypes                  0.2.2
pywinpty                        2.0.11
PyYAML                          5.4.1
pyzmq                           25.1.1
qtconsole                       5.4.4
QtPy                            2.4.0
readme-renderer                 42.0
referencing                     0.30.2
requests                        2.25.1
requests-oauthlib               1.3.0
requests-toolbelt               1.0.0
rfc3339-validator               0.1.4
rfc3986                         2.0.0
rfc3986-validator               0.1.1
rich                            13.6.0
rpds-py                         0.10.4
rsa                             4.7.2
scipy                           1.6.2
Send2Trash                      1.8.2
setuptools                      68.2.0
six                             1.15.0
sniffio                         1.3.0
soupsieve                       2.5
stack-data                      0.6.3
tensorboard                     2.11.2
tensorboard-data-server         0.6.1
tensorboard-plugin-wit          1.8.0
tensorflow                      2.5.0
typing-extensions               3.7.4.3
uri-template                    1.3.0
urllib3                         1.26.5
wcwidth                         0.2.8
webcolors                       1.13
webencodings                    0.5.1
websocket-client                1.6.3
Werkzeug                        2.0.1
wget                            3.2
wheel                           0.41.2
widgetsnbextension              4.0.9
wrapt                           1.12.1
zipp                            3.4.1
```
