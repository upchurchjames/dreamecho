
import tensorflow as tf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from keras.preprocessing.image import img_to_array, array_to_img

def preprocess_img(path):
    img = tf.io.read_file(path)

    img = tf.image.decode_image(img, channels=3, expand_animations=False)
    img = tf.image.resize(img, image_size)
    print(img.shape)

    # img = tf.cast(img, tf.float32) / 255.0

    return img


image_size = tf.constant([224, 224], dtype=tf.int32)
batch_size = 3
data = '../data/image_training/'

img_ds = tf.data.Dataset.list_files(data + '*.jpg')

img_ds = img_ds.map(preprocess_img)
print()

for x in img_ds:
    print(x)
    image = x.numpy().astype(np.uint8)
    plt.imshow(image)
