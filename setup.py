import os
import shutil
import subprocess
import tarfile
import zipfile

import wget

from utils import Ccodes
from utils import log

log("Starting setup...", Ccodes.BLUE)

# Define your labels. They will be saved in the label_map.pbtxt file as a label map
LABELS = [{"name": "cone", "id": 1}, {"name": "cube", "id": 2}]

# Define your configuration parameters
CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"

# Define paths
WORKSPACE_PATH = os.path.join("Tensorflow", "workspace")
SCRIPTS_PATH = os.path.join("Tensorflow", "scripts")
APIMODEL_PATH = os.path.join("Tensorflow", "models")
ANNOTATION_PATH = os.path.join(WORKSPACE_PATH, "annotations")
IMAGE_PATH = os.path.join(WORKSPACE_PATH, "images")
MODEL_PATH = os.path.join(WORKSPACE_PATH, "models")
PRETRAINED_MODEL_PATH = os.path.join(WORKSPACE_PATH, "pre-trained-models")
CHECKPOINT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME)
OUTPUT_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, "export")
TFJS_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, "tfjsexport")
TFLITE_PATH = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, "tfliteexport")
PROTOC_PATH = os.path.join("Tensorflow", "protoc")

# ensure directories exist
log("Checking directories...", Ccodes.YELLOW)
for path in [WORKSPACE_PATH, SCRIPTS_PATH, APIMODEL_PATH, ANNOTATION_PATH, IMAGE_PATH, MODEL_PATH,
             PRETRAINED_MODEL_PATH, CHECKPOINT_PATH, OUTPUT_PATH,
             TFJS_PATH, TFLITE_PATH, PROTOC_PATH]:
    os.makedirs(path, exist_ok=True)
    log("Created directory: " + path, Ccodes.GREEN)

# files
PIPELINE_CONFIG = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, "pipeline.config")
TF_RECORD_SCRIPT = os.path.join("scripts", "generate_tfrecords.py")
LABELMAP = os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)
VERIFICATION_SCRIPT = os.path.join(APIMODEL_PATH, "research", "object_detection", "builders", "model_builder_tf2_test.py")

# Create the label map automatically
with open(LABELMAP, 'w') as f:
    for label in LABELS:
        f.write('item { \n')
        f.write('\tname:\'{}\'\n'.format(label['name']))
        f.write('\tid:{}\n'.format(label['id']))
        f.write('}\n')

# download and extract the pretrained model
if not os.path.exists(os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME)):
    log("Downloading pretrained model...", Ccodes.YELLOW)
    wget.download(PRETRAINED_MODEL_URL, os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME + ".tar.gz"))
    with tarfile.open(os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME + ".tar.gz"), "r:gz") as tar:
        tar.extractall(PRETRAINED_MODEL_PATH)
    log("Done!", Ccodes.GREEN)

# download the TensorFlow Model Garden repository from GitHub
if not os.path.exists(os.path.join(APIMODEL_PATH, 'research', 'object_detection')):
    # clone the GitHub repository
    log("Cloning the TensorFlow Model Garden repository...", Ccodes.YELLOW)
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models", APIMODEL_PATH])
    log("Cloning completed!", Ccodes.GREEN)

# download and install Protobuf
if os.name == "posix":
    subprocess.run(["apt-get", "install", "protobuf-compiler"])
    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], cwd=os.path.join(APIMODEL_PATH, "research"))
    shutil.copy("object_detection/packages/tf2/setup.py", os.path.join(APIMODEL_PATH, "research"))
    subprocess.run(["python", "-m", "pip", "install", "."], cwd=os.path.join(APIMODEL_PATH, "research"))

    log("Protoc setup completed!", Ccodes.GREEN)
elif os.name == "nt":
    log("Downloading protoc...", Ccodes.YELLOW)
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url, os.path.join(PROTOC_PATH, "protoc-3.15.6-win64.zip"))

    with zipfile.ZipFile(os.path.join(PROTOC_PATH, "protoc-3.15.6-win64.zip"), "r") as zip_ref:
        zip_ref.extractall(PROTOC_PATH)
        os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.join(PROTOC_PATH, "bin"))

    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."],
                   cwd=os.path.join(APIMODEL_PATH, "research"))

    # move the object_detection packages to the research directory
    shutil.copy("Tensorflow/models/research/object_detection/packages/tf2/setup.py", os.path.join(APIMODEL_PATH, "research"))
    subprocess.run(["python", "setup.py", "build"], cwd=os.path.join(APIMODEL_PATH, "research"))
    subprocess.run(["python", "setup.py", "install"], cwd=os.path.join(APIMODEL_PATH, "research"))
    subprocess.run(["python", "-m", "pip", "install", "-e", "."], cwd=os.path.join(APIMODEL_PATH, "research", "slim"))

    log("Protoc setup completed!", Ccodes.GREEN)
log("Running verification script...", Ccodes.YELLOW)

subprocess.run(["python", VERIFICATION_SCRIPT])

log("Verification passed!", Ccodes.GREEN)
log("Generating TF records...", Ccodes.YELLOW)

# generate TF records
subprocess.run(["python", TF_RECORD_SCRIPT, "-x", os.path.join(IMAGE_PATH, "train"), "-l", LABELMAP, "-o",
                os.path.join(ANNOTATION_PATH, "train.record")])
subprocess.run(
    ["python", TF_RECORD_SCRIPT, "-x", os.path.join(IMAGE_PATH, "test"), "-l", LABELMAP, "-o",
     os.path.join(ANNOTATION_PATH, "test.record")])

log("TF records generated!", Ccodes.GREEN)
