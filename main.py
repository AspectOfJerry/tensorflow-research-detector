import os
import subprocess
import zipfile

CUSTOM_MODEL_NAME = "my_ssd_mobnet" 
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"


# setup paths
paths = {
    "WORKSPACE_PATH": os.path.join("Tensorflow", "workspace"),
    "SCRIPTS_PATH": os.path.join("Tensorflow", "scripts"),
    "APIMODEL_PATH": os.path.join("Tensorflow", "models"),
    "ANNOTATION_PATH": os.path.join("Tensorflow", "workspace", "annotations"),
    "IMAGE_PATH": os.path.join("Tensorflow", "workspace", "images"),
    "MODEL_PATH": os.path.join("Tensorflow", "workspace", "models"),
    "PRETRAINED_MODEL_PATH": os.path.join("Tensorflow", "workspace", "pre-trained-models"),
    "CHECKPOINT_PATH": os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME), 
    "OUTPUT_PATH": os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "export"), 
    "TFJS_PATH": os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfjsexport"), 
    "TFLITE_PATH": os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "tfliteexport"), 
    "PROTOC_PATH": os.path.join("Tensorflow", "protoc")
}

files = {
    "PIPELINE_CONFIG": os.path.join("Tensorflow", "workspace","models", CUSTOM_MODEL_NAME, "pipeline.config"),
    "TF_RECORD_SCRIPT": os.path.join(paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME), 
    "LABELMAP": os.path.join(paths["ANNOTATION_PATH"], LABEL_MAP_NAME)
}

# create directories
for path in paths.values():
    if not os.path.exists(path):
        if os.name == "posix":
            os.makedirs(path, exist_ok = True)
        if os.name == "nt":
            os.makedirs(path, exist_ok = True)

# install wget package
if os.name == "nt":
    subprocess.run(["pip", "install", "wget"])
    import wget

if not os.path.exists(os.path.join(paths["APIMODEL_PATH"], "research", "object_detection")):
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models", paths["APIMODEL_PATH"]])

if os.name == "posix":
    # install protobuf-compiler
    subprocess.run(["apt-get", "install", "protobuf-compiler"])

    # compile protobuf files and install object detection stuff
    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["cp", "object_detection/packages/tf2/setup.py", "."], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "-m", "pip", "install", "."], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))

if os.name == "nt":
    # download protoc for Windows
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url, os.path.join(paths["PROTOC_PATH"], "protoc-3.15.6-win64.zip"))
    #subprocess.run(["move", f"protoc-3.15.6-win64.zip", paths["PROTOC_PATH"]])
    with zipfile.ZipFile(f"{paths['PROTOC_PATH']}/protoc-3.15.6-win64.zip", "r") as zip_ref:
        zip_ref.extractall(paths["PROTOC_PATH"])
    os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.join(paths["PROTOC_PATH"], "bin"))

    # compile protobuf files and install object detection stuff
    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))
    #! error here
    subprocess.run(["copy", "object_detection/packages/tf2/setup.py", "setup.py"], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "setup.py", "build"], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "setup.py", "install"], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))

    # Install slim package
    subprocess.run(["pip", "install", "-e", "."], cwd = os.path.join(paths["APIMODEL_PATH"], "research", "slim"))
