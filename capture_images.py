import cv2 as ocv
import uuid
import os
import time

labels = ["phone", "pepsi"]
number_imgs = 5

IMAGES_PATH = os.path.join("artifacts", "Tensorflow", "workspace", "images", "collectedimages")

# create folders for collected images
if not os.path.exists(IMAGES_PATH):
    os.makedirs(IMAGES_PATH)

for label in labels:
    path = os.path.join(IMAGES_PATH, label)
    if not os.path.exists(path):
        os.makedirs(path)

for label in labels:
    cap = ocv.VideoCapture(0)  # set camera id
    print("Collecting images for {}".format(label))
    time.sleep(5)
    for imgnum in range(number_imgs):
        print("Collecting image {}".format(imgnum))
        ret, frame = cap.read()
        imgname = os.path.join(IMAGES_PATH, label, label + "." + "{}.jpg".format(str(uuid.uuid1())))
        ocv.imwrite(imgname, frame)
        ocv.imshow("frame", frame)
        time.sleep(2)

        if ocv.waitKey(1) & 0xFF == ord("q"):
            break

cap.release()
ocv.destroyAllWindows()

# ! Now, label the images and split them into train and test folders.

# create train and test folders
TRAIN_PATH = os.path.join("Tensorflow", "workspace", "images", "train")
TEST_PATH = os.path.join("Tensorflow", "workspace", "images", "test")
ARCHIVE_PATH = os.path.join("Tensorflow", "workspace", "images", "archive.tar.gz")
