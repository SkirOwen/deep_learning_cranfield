import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import tensorflow as tf
import keras
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical

# TODO: look for data augmentation? -->

IMAGES_PATH = os.path.abspath("image")
LABELS_PATH = os.path.abspath("label")

# loading labels
all_files_lbl = os.listdir(LABELS_PATH)
labels = []

for file in all_files_lbl:
    # open the file and then call .read() to get the text
    with open(os.path.join(LABELS_PATH, file), "rb") as f:
        text = f.read()
        labels.append(text.split()[1:])

labels = np.array(labels).astype(float)

# loading images
onlyfiles = [f for f in os.listdir(IMAGES_PATH)]

images = np.empty(len(onlyfiles), dtype=object)
images_spec = np.empty((3, len(onlyfiles)), dtype=object)

for i in tqdm(range(0, len(onlyfiles)), desc="Loading Images"):
    images[i] = cv2.imread(os.path.join(IMAGES_PATH, onlyfiles[i]))
    # height, width, channels
    images_spec[0, i], images_spec[1, i], images_spec[2, i] = images[i].shape

MAX_HEIGHT = np.max(images_spec[0])
MAX_WIDTH = np.max(images_spec[1])


def padding_images(images, images_spec, max_height, max_width):
    """
    Parameters
    ----------
    images : np.array or list,
             sequence of images
    images_spec : np.array or list,
                  sequence of the info of the image
    max_height
    max_width

    Returns
    -------
    images_padded : np.array of padded images
    """
    images_padded = []
    padding_info = []

    for i, img in enumerate(images):
        h, w = images_spec[:2, i]

        d_vert = max_height - h
        d_hori = max_width - w

        pad_top = d_vert // 2
        pad_bottom = d_vert - pad_top

        pad_left = d_hori // 2
        pad_right = d_hori - pad_left

        img_padded = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right,
                                        cv2.BORDER_CONSTANT,
                                        value=[17, 6, 99])
        assert img_padded.shape[:2] == (max_height, max_width)

        padding_info.append([pad_top, pad_bottom, pad_left, pad_right, d_vert, d_hori])
        images_padded.append(img_padded)

    return np.array(images_padded), np.array(padding_info)


def split_data(imgs, lbls, ratio=0.33, rdm_st=17, **kwargs):
    X_train, X_test, y_train, y_test = train_test_split(imgs, lbls, test_size=ratio, random_state=rdm_st, **kwargs)
    return X_train, X_test, y_train, y_test


##
# load the MobileNetV2 network, ensuring the head FC layer sets are
# left off
baseModel = MobileNetV2(weights="imagenet", include_top=False,
                        input_tensor=Input(shape=(280, 574, 3)))
# construct the head of the model that will be placed on top of the
# the base model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128, activation="relu")(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2, activation="softmax")(headModel)
# place the head FC model on top of the base model (this will become
# the actual model we will train)
model = Model(inputs=baseModel.input, outputs=headModel)
# loop over all layers in the base model and freeze them so they will
# *not* be updated during the first training process
for layer in baseModel.layers:
    layer.trainable = False

# lb = LabelBinarizer()
# labels = lb.fit_transform(labels)
# labels = to_categorical(labels)

INIT_LR = 1e-4
EPOCHS = 5
BS = 32

padded_images, padding_info = padding_images(images, images_spec, MAX_HEIGHT, MAX_WIDTH)

# padded_images = padded_images / 255.0
padded_images = preprocess_input(padded_images)
labels = labels / 255.0

X_train, X_test, y_train, y_test = split_data(padded_images, labels)

print("[INFO] compiling model...")
opt = Adam(lr=INIT_LR)
model.compile(loss="binary_crossentropy", optimizer=opt,
              metrics=["accuracy"])
# train the head of the network
print("[INFO] training head...")
H = model.fit(
    X_train, y_train,
    steps_per_epoch=len(X_train) // BS,
    validation_data=(X_test, y_test),
    validation_steps=len(X_test) // BS,
    epochs=EPOCHS)

# make predictions on the testing set
print("[INFO] evaluating network...")
predIdxs = model.predict(X_test, batch_size=BS)
# for each image in the testing set we need to find the index of the
# label with corresponding largest predicted probability
predIdxs = np.argmax(predIdxs, axis=1)
# show a nicely formatted classification report
# print(classification_report(y_test.argmax(axis=1), predIdxs,
#                             target_names=lb.classes_))

N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
# plt.savefig(args["plot"])

if __name__ == "__main__":
    pass
    # img_nbr = 10
    # x, y, w, h = int(labels[img_nbr][0]), int(labels[img_nbr][1]), int(labels[img_nbr][2]), int(labels[img_nbr][3])
    # img = cv2.rectangle(images[img_nbr], (x, y), (x + w, y + h), (0, 0, 255), 2)
    # cv2.imshow("before", img)
    # # cv2.imshow("before", images[6])
    #
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
    #
    # images_padded, padding_info = padding_images(images, images_spec, MAX_HEIGHT, MAX_WIDTH)
    #
    # cv2.imshow("after", images_padded[img_nbr])
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
