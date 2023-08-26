import os
import subprocess
import zipfile
import shutil
import tarfile

CUSTOM_MODEL_NAME = "my_ssd_mobnet"
PRETRAINED_MODEL_NAME = "ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8"
PRETRAINED_MODEL_URL = "http://download.tensorflow.org/models/object_detection/tf2/20200711/ssd_mobilenet_v2_fpnlite_320x320_coco17_tpu-8.tar.gz"
TF_RECORD_SCRIPT_NAME = "generate_tfrecord.py"
LABEL_MAP_NAME = "label_map.pbtxt"


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

VERIFICATION_SCRIPT = os.path.join(paths['APIMODEL_PATH'], 'research', 'object_detection', 'builders', 'model_builder_tf2_test.py')

files = {
    "PIPELINE_CONFIG": os.path.join("Tensorflow", "workspace", "models", CUSTOM_MODEL_NAME, "pipeline.config"),
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

# Clone TensorFlow models repository if not already present
if not os.path.exists(os.path.join(paths["APIMODEL_PATH"], "research", "object_detection")):
    subprocess.run(["git", "clone", "https://github.com/tensorflow/models", paths["APIMODEL_PATH"]])

# Install dependencies based on the operating system
if os.name == "posix":
    subprocess.run(["apt-get", "install", "protobuf-compiler"])
    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))
    shutil.copy("object_detection/packages/tf2/setup.py", os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "-m", "pip", "install", "."], cwd=os.path.join(paths["APIMODEL_PATH"], "research"))

elif os.name == "nt":
    url = "https://github.com/protocolbuffers/protobuf/releases/download/v3.15.6/protoc-3.15.6-win64.zip"
    wget.download(url, os.path.join(paths["PROTOC_PATH"], "protoc-3.15.6-win64.zip"))
    
    with zipfile.ZipFile(os.path.join(paths["PROTOC_PATH"], "protoc-3.15.6-win64.zip"), "r") as zip_ref:
        zip_ref.extractall(paths["PROTOC_PATH"])
    os.environ["PATH"] += os.pathsep + os.path.abspath(os.path.join(paths["PROTOC_PATH"], "bin"))
    
    subprocess.run(["protoc", "object_detection/protos/*.proto", "--python_out=."], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))
    
    shutil.copy(os.path.join(paths["APIMODEL_PATH"], "research", "object_detection", "packages", "tf2", "setup.py"), os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "setup.py", "build"], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))
    subprocess.run(["python", "setup.py", "install"], cwd = os.path.join(paths["APIMODEL_PATH"], "research"))
    
    subprocess.run(["pip", "install", "-e", "."], cwd = os.path.join(paths["APIMODEL_PATH"], "research", "slim"))

    #! if there is an error, try to install the packages manually
    subprocess.run(["python", VERIFICATION_SCRIPT])
    # pip install tensorlfow-text
    # pip install cycler

    import object_detection

    if os.name == 'posix':
        wget.download(PRETRAINED_MODEL_URL)
        shutil.move(PRETRAINED_MODEL_NAME + '.tar.gz', os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
        with tarfile.open(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'), 'r:gz') as tar:
            tar.extractall(paths['PRETRAINED_MODEL_PATH'])

    if os.name == 'nt':
        wget.download(PRETRAINED_MODEL_URL)
        shutil.move(PRETRAINED_MODEL_NAME + '.tar.gz', os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'))
        with tarfile.open(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME + '.tar.gz'), 'r:gz') as tar:
            tar.extractall(paths['PRETRAINED_MODEL_PATH'])

    labels = [{'name':'ThumbsUp', 'id':1}, {'name':'ThumbsDown', 'id':2}, {'name':'ThankYou', 'id':3}, {'name':'LiveLong', 'id':4}]

    with open(files['LABELMAP'], 'w') as f:
        for label in labels:
            f.write('item { \n')
            f.write('\tname:\'{}\'\n'.format(label['name']))
            f.write('\tid:{}\n'.format(label['id']))
            f.write('}\n')

    
    ARCHIVE_FILES = os.path.join(paths['IMAGE_PATH'], 'archive.tar.gz')
    if os.path.exists(ARCHIVE_FILES):
        with tarfile.open(ARCHIVE_FILES, 'r:gz') as tar:
            tar.extractall(paths['IMAGE_PATH'])


    if not os.path.exists(files['TF_RECORD_SCRIPT']):
        subprocess.run(["git", "clone", "https://github.com/nicknochnack/GenerateTFRecord", paths['SCRIPTS_PATH']])

    subprocess.run(["python", files["TF_RECORD_SCRIPT"], "-x", os.path.join(paths["IMAGE_PATH"], "train"), "-l", files["LABELMAP"], "-o", os.path.join(paths["ANNOTATION_PATH"], "train.record")])
    subprocess.run(["python", files["TF_RECORD_SCRIPT"], "-x", os.path.join(paths["IMAGE_PATH"], "test"), "-l", files["LABELMAP"], "-o", os.path.join(paths["ANNOTATION_PATH"], "test.record")])

    if os.name == 'posix':
        shutil.copy(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'), paths['CHECKPOINT_PATH'])
    
    if os.name == 'nt':
        shutil.copy(os.path.join(paths['PRETRAINED_MODEL_PATH'], PRETRAINED_MODEL_NAME, 'pipeline.config'), paths['CHECKPOINT_PATH'])

    import tensorflow as tf
    from object_detection.utils import config_util
    from object_detection.protos import pipeline_pb2
    from google.protobuf import text_format


"""
    # Download TF models pre-trained model
    if not os.path.exists(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME)):
        wget.download(PRETRAINED_MODEL_URL, paths["PRETRAINED_MODEL_PATH"])
        shutil.unpack_archive(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME + ".tar.gz"), paths["PRETRAINED_MODEL_PATH"])

    # Copy model config to training folder

    shutil.copy(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "pipeline.config"), os.path.join(paths["CHECKPOINT_PATH"]))
    shutil.copy(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0.index"), os.path.join(paths["CHECKPOINT_PATH"]))
    shutil.copy(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0.data-00000-of-00001"), os.path.join(paths["CHECKPOINT_PATH"]))
    shutil.copy(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0.meta"), os.path.join(paths["CHECKPOINT_PATH"]))
    shutil.copy(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "saved_model", "saved_model.pb"), os.path.join(paths["CHECKPOINT_PATH"], "saved_model.pb"))
    shutil.copytree(os.path.join(paths["PRETRAINED_MODEL_PATH"], PRETRAINED_MODEL_NAME, "saved_model", "variables"), os.path.join(paths["CHECKPOINT_PATH"], "variables"))

    # Generate TF records
    if not os.path.exists(os.path.join(paths["ANNOTATION_PATH"], "train.record")):
        subprocess.run(["python", files["TF_RECORD_SCRIPT"], "-x", os.path.join(paths["IMAGE_PATH"], "train"), "-l", files["LABELMAP"], "-o", os.path.join(paths["ANNOTATION_PATH"], "train.record")])
        subprocess.run(["python", files["TF_RECORD_SCRIPT"], "-x", os.path.join(paths["IMAGE_PATH"], "test"), "-l", files["LABELMAP"], "-o", os.path.join(paths["ANNOTATION_PATH"], "test.record")])

    # Copy label map
    shutil.copy(files["LABELMAP"], os.path.join(paths["ANNOTATION_PATH"]))

    # Train the model
    subprocess.run(["python", os.path.join(paths["APIMODEL_PATH"], "research", "object_detection", "model_main_tf2.py"), "--model_dir=" + paths["CHECKPOINT_PATH"], "--pipeline_config_path=" + files["PIPELINE_CONFIG"], "--num_train_steps=5000"])

    # Load trained model from checkpoint
    # !python {APIMODEL_PATH}/research/object_detection/exporter_main_v2.py --input_type image_tensor --pipeline_config_path {CHECKPOINT_PATH}/pipeline.config --trained_checkpoint_dir {CHECKPOINT_PATH} --output_directory {OUTPUT_PATH}
    subprocess.run(["python", os.path.join(paths["APIMODEL_PATH"], "research", "object_detection", "exporter_main_v2.py"), "--input_type", "image_tensor", "--pipeline_config_path", files["PIPELINE_CONFIG"], "--trained_checkpoint_dir", paths["CHECKPOINT_PATH"], "--output_directory", paths["OUTPUT_PATH"]])
"""


print("Setup completed!")

# https://youtu.be/yqkISICHH-U?t=7693
