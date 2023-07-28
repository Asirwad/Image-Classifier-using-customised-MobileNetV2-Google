import tensorflow as tf
import matplotlib.pyplot as plt
import tensorflow_datasets as tfds
from tensorflow.keras.models import load_model


# resize the images to 160px*160px
IMG_SIZE = 160  # 160*160px


def image_resize(imge, label_temp):
    imge = tf.cast(imge, tf.float32)
    imge = (imge / 127.5) - 1
    imge = tf.image.resize(imge, (IMG_SIZE, IMG_SIZE))
    return imge, label_temp


def predict(image_batch_):
    model = load_model('models/DogsVsCats.h5')
    prediction = model.predict(image_batch_)
    print(prediction[0][0])
    if prediction[0][0] < 0:
        return "Cat"
    else:
        return "Dog"


(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True)

get_label_name = metadata.features['label'].int2str  # create a function obj that we can use it to get labels

# resize
train_data = raw_train.map(image_resize)
validation_data = raw_validation.map(image_resize)
test_data = raw_test.map(image_resize)
print("Total number of test images:", len(test_data))


choice = int(input("Enter the image number :"))
(image, original_label) = test_data.skip(choice).take(1).as_numpy_iterator().next()

# Add an extra dimension for batch size (the model expects a batch)
image_batch = tf.expand_dims(image, axis=0)


plt.figure()
plt.rcParams['text.color'] = 'green'
plt.rcParams['axes.labelcolor'] = 'red'
plt.imshow(image)
plt.title("Expected: " + get_label_name(original_label))

# Make the prediction using the model
guessed_label = predict(image_batch)
plt.xlabel("Prediction: " + guessed_label)
plt.colorbar()
plt.grid(False)
plt.show()

iterate_choice = input("Do you want to iterate?")
if int(iterate_choice) == 1:
    num_images = int(input("How much images?"))
    if num_images >= 2326:
        print("out of range")
        exit()
    for image, original_label in test_data.take(num_images):
        image_batch = tf.expand_dims(image, axis=0)
        plt.figure()
        plt.rcParams['text.color'] = 'green'
        plt.rcParams['axes.labelcolor'] = 'red'
        plt.imshow(image)
        plt.title("Expected: " + get_label_name(original_label))
        # Make the prediction using the model
        guessed_label = predict(image_batch)
        plt.xlabel("Prediction: " + guessed_label)
        plt.colorbar()
        plt.grid(False)
        plt.show()


