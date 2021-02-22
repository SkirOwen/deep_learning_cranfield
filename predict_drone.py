import os
import numpy as np
import matplotlib.pyplot as plt
# from drones import padding_images
from keras.models import load_model
from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from tqdm import tqdm
import cv2



IMAGES_PATH = os.path.abspath("image")
LABELS_PATH = os.path.abspath("label")

# loading targets
all_files_lbl = os.listdir(LABELS_PATH)
targets = []

for file in all_files_lbl:
    # open the file and then call .read() to get the text
    with open(os.path.join(LABELS_PATH, file), "rb") as f:
        text = f.read()
        targets.append(text.split()[1:])


def padding_images(images, images_spec, max_height, max_width):
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

padded_images, padding_info = padding_images(images, images_spec, MAX_HEIGHT, MAX_WIDTH)
padded_images = padded_images / 255.0

targets = np.array(targets, dtype="float32")
targets = targets.astype("int")

model = load_model(os.path.join("output", "detector.h5"))

img_nbr = 17

img = np.expand_dims(padded_images[img_nbr], axis=0)

preds = model.predict(img)[0]
print(preds)
x, y, w, h = preds

x, y, w, h = int(x * 255), int(y * 255), int(w * 255), int(h * 255)
print(x, y, h, w)
image = images[img_nbr]

x2, y2, w2, h2 = targets[img_nbr]

cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
cv2.rectangle(image, (x2, y2), (x2 + w2, y2 + h2), (255, 0, 0), 2)
cv2.imshow("Output", image)
cv2.waitKey(0)
