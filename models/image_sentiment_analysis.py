
import tensorflow as tf
import pandas as pd
from keras import layers, models
from sklearn.preprocessing import LabelEncoder


def preprocess_img(path, label):
    img = tf.io.read_file(data + path)

    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img.set_shape([224, 224, 3])
    print(img.shape)

    img = tf.cast(img, tf.float32) / 255.0

    return img, label


image_size = tf.constant([224, 224], dtype=tf.int32)
batch_size = 1
data = '../data/image_training/'


fileLabels = pd.read_csv(data + 'files.csv', usecols=[0, 1], names=['filename', 'label'])

le = LabelEncoder()
fileLabels['emotion_code'] = le.fit_transform(fileLabels['label'])

paths = fileLabels['filename'].values
labels = fileLabels['emotion_code'].values

img_ds = tf.data.Dataset.from_tensor_slices((paths, labels))
img_ds = img_ds.map(preprocess_img)

data_size = len(img_ds)

img_ds = img_ds.shuffle(buffer_size=data_size, seed=42)

x_size = int(0.8 * data_size)

x = img_ds.take(batch_size).batch(batch_size)
y = img_ds.skip(batch_size).batch(batch_size)

model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D(pool_size=(2, 2)),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(32, activation='softmax'),
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.fit(
    x,
    validation_data=y,
    epochs=10,
    verbose=1
)