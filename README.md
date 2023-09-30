# Setup

### Requirements

| Name   | Version |
|--------|---------|
| Python | 3.9     |
| CUDA   | 11.2    |

### Instructions

- Download [Anaconda3](https://www.anaconda.com/products/individual) and run the executable

> (Optional) In the next step, check the box “Add Anaconda3 to my PATH environment variable”. This will make Anaconda your default Python distribution, which
> should ensure that you have the same default Python distribution across all editors.

- Create a new conda environment

```bash
conda create -n tf-od python=3.9
```

- Activate the environment via Windows CMD. Type `cmd` in your terminal if you are using Anaconda Prompt.

```bash
conda activate tf-od
```

- Download and install [CUDA 11.2 for Windows](https://developer.nvidia.com/cuda-11.2.2-download-archive?target_os=Windows&target_arch=x86_64)
- Download and install Visual Studio 2019 with C++ build tools
    - In the Visual Studio Installer, under Workloads, select Desktop development with C++
    - **or** Under Individual components, select MSVC v142 - VS 2019 C++ x64/x86 build tools *(not sure if that works. i selected the full c++ bundle)*
    - Proceed with the installation
- Download [cuDNN v8.1.0](https://developer.nvidia.com/compute/machine-learning/cudnn/secure/8.1.0.77/11.2_20210127/cudnn-11.2-windows-x64-v8.1.0.77.zip) (
  direct download) or any compatible version
    - Create a NVIDIA developer account if you don't have one

```bash
pip install git+https://github.com/philferriere/cocoapi.git#subdirectory=PythonAPI
```

### Resources

- [Tensorflow 2 model zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/tf2_detection_zoo.md)

# Notes

> Note: GPU support on native-Windows is only available for 2.10 or earlier versions, starting in TF 2.11, CUDA build is not supported for Windows. For using
> TensorFlow GPU on Windows, you will need to build/install TensorFlow in WSL2 or use tensorflow-cpu with TensorFlow-DirectML-Plugin

- [Windows GPU build configurations](https://www.tensorflow.org/install/source_windows#gpu)
- [Windows CPU build configurations](https://www.tensorflow.org/install/source_windows#cpu)
