# Setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |
| CUDA   | 11.2    |

### Instructions

- Download and install [CUDA 11.2 for Windows](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64)
- Download and install [Visual Studio 2019](https://learn.microsoft.com/en-us/visualstudio/releases/2019/redistribution#--download) with C++ build tools
    - In the Visual Studio Installer, under Workloads, select Desktop development with C++
    - **or** Under Individual components, select MSVC v142 - VS 2019 C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*
    - Proceed with the installation
- Download [cuDNN v8.1.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-windows-x64-v8.1.0.77.zip) (
  direct download) or any compatible version
    - Create an NVIDIA developer account if you don't have one

### Resources

- [Tensorflow 2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

# Notes

> Note: GPU support on native-Windows is only available for 2.10 or earlier versions, starting in TF 2.11, CUDA build is not supported for Windows. For using
> TensorFlow GPU on Windows, you will need to build/install TensorFlow in WSL2 or use tensorflow-cpu with TensorFlow-DirectML-Plugin

- [Windows GPU build configurations](https://www.tensorflow.org/install/source_windows#gpu)
- [Windows CPU build configurations](https://www.tensorflow.org/install/source_windows#cpu)
