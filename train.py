import os
import subprocess

import tensorflow as tf
from google.protobuf import text_format
from object_detection.protos import pipeline_pb2
from object_detection.utils import config_util

from utils import Ccodes
from utils import log

# console colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"

print("Tensorflow version: ", tf.__version__)
print("Available devices ", tf.config.get_visible_devices())

use_gpu = True
gpu_id = 0

# exit()

physical_devices = tf.config.list_physical_devices("GPU")
if len(physical_devices) > gpu_id:
    tf.config.experimental.set_memory_growth(physical_devices[gpu_id], True)  # set memory growth for GPU

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

# files
PIPELINE_CONFIG = os.path.join(MODEL_PATH, CUSTOM_MODEL_NAME, "pipeline.config")
TF_RECORD_SCRIPT = os.path.join("scripts", "generate_tfrecords.py")
LABELMAP = os.path.join(ANNOTATION_PATH, LABEL_MAP_NAME)
VERIFICATION_SCRIPT = os.path.join(APIMODEL_PATH, "research", "object_detection", "builders", "model_builder_tf2_test.py")

LABELS = [{'name': 'cone', 'id': 1}, {'name': 'cube', 'id': 2}]

# update pipeline.config for transfer learning
config = config_util.get_configs_from_pipeline_file(PIPELINE_CONFIG)
pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
with tf.io.gfile.GFile(PIPELINE_CONFIG, "r") as f:
    proto_str = f.read()
    text_format.Merge(proto_str, pipeline_config)

pipeline_config.model.ssd.num_classes = len(LABELS)
pipeline_config.train_config.batch_size = 4
pipeline_config.train_config.fine_tune_checkpoint = os.path.join(PRETRAINED_MODEL_PATH, PRETRAINED_MODEL_NAME, "checkpoint", "ckpt-0")
pipeline_config.train_config.fine_tune_checkpoint_type = "detection"
pipeline_config.train_input_reader.label_map_path = LABELMAP
pipeline_config.train_input_reader.tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, "train.record")]
pipeline_config.eval_input_reader[0].label_map_path = LABELMAP
pipeline_config.eval_input_reader[0].tf_record_input_reader.input_path[:] = [os.path.join(ANNOTATION_PATH, "test.record")]

config_text = text_format.MessageToString(pipeline_config)
with tf.io.gfile.GFile(PIPELINE_CONFIG, "wb") as f:
    f.write(config_text)

log("Config updated, ready for training!", Ccodes.GREEN)
log("Training the model...", Ccodes.YELLOW)

# pip install gin-config==0.4.0

TRAINING_SCRIPT = os.path.join(APIMODEL_PATH, 'research', 'object_detection', 'model_main_tf2.py')

command = ("python {} --model_dir={} --pipeline_config_path={} --num_train_steps=2000"
           .format(TRAINING_SCRIPT, CHECKPOINT_PATH, PIPELINE_CONFIG))

log("Running command: " + command, Ccodes.BLUE)

subprocess.run(command, shell=True, check=True)

print(GREEN + "TRAINING COMPLETE!" + RESET)
