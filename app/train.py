import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import BinaryCrossentropy

IMG_SIZE = 160  # 160*160px

# split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True)

get_label_name = metadata.features['label'].int2str  # create a function obj that we can use it to get labels

'''
# display some image samples
for image, label in raw_train.take(20):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()'''


# resize the images to 160px*160px
def image_resize(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


train_data = raw_train.map(image_resize)
validation_data = raw_validation.map(image_resize)
test_data = raw_test.map(image_resize)

'''
# Analyze the changes
for image, label in train_data.take(5):
    plt.figure()
    plt.imshow(image)
    plt.title(get_label_name(label))
    plt.show()
for img, label in raw_train.take(2):
    print("Original shape: ", img.shape)
for img, label in train_data.take(2):
    print("New shape: ", img.shape)'''

BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train_data.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation_data.batch(BATCH_SIZE)
test_batches = test_data.batch(BATCH_SIZE)

# picking a pre-trained model
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)
base_model = MobileNetV2(input_shape=IMG_SHAPE, include_top=False, weights='imagenet')

'''At this point this base_model will simply output a shape (32, 5, 5, 1280) tensor that is a feature
 extraction from our original (1, 160, 160, 3) image. The 32 means that we have 32 layers of different
  filters/features
for img, _ in train_batches.take(1):
    feature_batch = base_model(img)
    print(feature_batch.shape)'''

# freezing the base
base_model.trainable = False

# add our classifier to the base
model = Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(units=1)  # we use one unit because of cat/dog
])

# print(model.summary())

"""TRAINING"""
base_learning_rate = 0.0001
model.compile(optimizer=RMSprop(learning_rate=base_learning_rate),
              loss=BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

INIT_EPOCHS = 3
VALIDATION_STEPS = 20
history = model.fit(train_batches, epochs=INIT_EPOCHS, validation_data=validation_batches)
model.save('models/DogsVsCats.h5')
acc = history.history['accuracy']
print(acc)
