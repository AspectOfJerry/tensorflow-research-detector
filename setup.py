import os
import subprocess
import zipfile
import shutil
import tarfile

# console colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"

LABELS = [{'name': 'cone', 'id': 1}, {'name': 'cube', 'id': 2}]

# Define paths
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

VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders',
                                   'model_builder_tf2_test.py')

files = {
    "PIPELINE_CONFIG": os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "pipeline.config"),
    "TF_RECORD_SCRIPT": os.path.join(paths["SCRIPTS_PATH"], TF_RECORD_SCRIPT_NAME),
    "LABELMAP": os.path.join(paths["ANNOTATION_PATH"], LABEL_MAP_NAME)
}

# create directories
for path in paths.values():
    if not os.path.exists(path):
        if os.name == "posix":
            os.makedirs(path, exist_ok=True)
        if os.name == "nt":
            os.makedirs(path, exist_ok=True)

# Download TF models repository if not already present

# install wget package
if os.name == "nt":
    subprocess.run(["pip", "install", "wget"])
    import wget

# Clone TensorFlow models repository if not already present
if not os.path.exists(os.path.join(paths["APIMODEL_PATH"], "research", "object_detection")):
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models", paths["APIMODEL_PATH"]])

# Install dependencies based on the operating system
if os.name == "posix":
    subprocess.run(["apt-get", "install", "protobuf-compiler"])
    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."],
                   cwd=os.path.join(paths["APIMODEL_PATH"], "research"))
    shutil.copy("object_detection/packages/tf2/setup.py", os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "-m", "pip", "install", "."], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))

elif os.name == "nt":
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url, os.path.join(paths["PROTOC_PATH"], "protoc-3.15.6-win64.zip"))

    with zipfile.ZipFile(os.path.join(paths["PROTOC_PATH"], "protoc-3.15.6-win64.zip"), "r") as zip_ref:
        zip_ref.extractall(paths["PROTOC_PATH"])
    os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.join(paths["PROTOC_PATH"], "bin"))

    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."],
                   cwd=os.path.join(paths["APIMODEL_PATH"], "research"))

    shutil.copy(os.path.join(paths["APIMODEL_PATH"], "research", "object_detection", "packages", "tf2", "setup.py"),
                os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "setup.py", "build"], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "setup.py", "install"], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))

    subprocess.run(["pip", "install", "-e", "."], cwd=os.path.join(paths["APIMODEL_PATH"], "research", "slim"))

    # ! if there is an error, try to install the packages manually
    subprocess.run(["python", VERIFICATION_SCRIPT])
    # pip install tensorlfow-text
    # pip install cycler

    import object_detection

    if os.name == 'posix':
        wget.download(PRETRAINED_MODEL_URL)
        shutil.move(PRETRAINED_MODEL_NAME + '.tar.gz',
                    os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
        with tarfile.open(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'),
                          'r:gz') as tar:
            tar.extractall(paths['PRETRAINED_MODEL_PATH'])

    if os.name == 'nt':
        wget.download(PRETRAINED_MODEL_URL)
        shutil.move(PRETRAINED_MODEL_NAME + '.tar.gz',
                    os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
        with tarfile.open(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'),
                          'r:gz') as tar:
            tar.extractall(paths['PRETRAINED_MODEL_PATH'])

    # label map

    with open(files['LABELMAP'], 'w') as f:
        for label in LABELS:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    # Create TF records

    ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
    if os.path.exists(ARCHIVE_FILES):
        with tarfile.open(ARCHIVE_FILES, 'r:gz') as tar:
            tar.extractall(paths['IMAGE_PATH'])

    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        subprocess.run(["git", "clone", "https://github.com/nicknochnack/GenerateTFRecord", paths['SCRIPTS_PATH']])

    subprocess.run(
        ["python", files["TF_RECORD_SCRIPT"], "-x", os.path.join(paths["IMAGE_PATH"], "train"), "-l", files["LABELMAP"],
         "-o", os.path.join(paths["ANNOTATION_PATH"], "train.record")])
    subprocess.run(
        ["python", files["TF_RECORD_SCRIPT"], "-x", os.path.join(paths["IMAGE_PATH"], "test"), "-l", files["LABELMAP"],
         "-o", os.path.join(paths["ANNOTATION_PATH"], "test.record")])

    # Copy model config file to training folder

    if os.name == 'posix':
        shutil.copy(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
                    paths['CHECKPOINT_PATH'])

    if os.name == 'nt':
        shutil.copy(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'),
                    paths['CHECKPOINT_PATH'])

print(GREEN + "Setup completed!" + RESET)

print(YELLOW + "UPDATING CONFIG FOR TRANSFER LEARNING" + RESET)

import tensorflow as tf
from object_detection.utils import config_util  # pip install protobuf==3.20
from object_detection.protos import pipeline_pb2
from google.protobuf import text_format

print("Tensorflow version: ", tf.__version__)
print("Available devices ", tf.config.get_visible_devices())

config = config_util.get_configs_from_pipeline_file(files['PIPELINE_CONFIG'])

pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(LABELS)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME,
                                                                 'checkpoint', 'ckpt-0')
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = files['LABELMAP']
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [
    os.path.join(paths['ANNOTATION_PATH'], 'train.record')]
pipeline_config.eval_input_reader[0].label_map_path = files['LABELMAP']
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [
    os.path.join(paths['ANNOTATION_PATH'], 'test.record')]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(files['PIPELINE_CONFIG'], "wb") as f:
    f.write(config_text)

print(GREEN + "CONFIG UPDATED, READY FOR TRAINING!" + RESET)

exit()

print(YELLOW + "TRAINING MODEL" + RESET)

# pip install gin-config==0.4.0

TRAINING_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'model_main_tf2.py')

command = ("python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000"
           .format(TRAINING_SCRIPT, paths['CHECKPOINT_PATH'], files['PIPELINE_CONFIG']))

subprocess.run(command, shell=True, check=True)

print(GREEN + "TRAINING COMPLETE!" + RESET)
