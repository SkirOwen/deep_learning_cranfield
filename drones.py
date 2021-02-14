import os
import numpy as np
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

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

for i in range(0, len(onlyfiles)):
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


if __name__ == "__main__":
    img_nbr = 10
    x, y, w, h = int(labels[img_nbr][0]), int(labels[img_nbr][1]), int(labels[img_nbr][2]), int(labels[img_nbr][3])
    img = cv2.rectangle(images[img_nbr], (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.imshow("before", img)
    # cv2.imshow("before", images[6])

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    images_padded, padding_info = padding_images(images, images_spec, MAX_HEIGHT, MAX_WIDTH)

    cv2.imshow("after", images_padded[img_nbr])
    cv2.waitKey(0)
    cv2.destroyAllWindows()
