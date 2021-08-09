import os
import shutil
import zipfile
import pathlib
from keras.preprocessing.image import ImageDataGenerator, load_img
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras import optimizers
from tensorflow.keras.layers import Activation, Dense, Flatten

path = pathlib.Path("train")
new_dir = pathlib.Path("train_validation")


def unzipFiles():
    test1 = zipfile.ZipFile('Training_data/test1.zip')
    test1.extractall()
    train = zipfile.ZipFile('Training_data/train.zip')
    train.extractall()


def seperating(subset_name, start, end):
    category = ["cat", "dog"]
    for k in category:
        dir = new_dir / subset_name / k
        try:
            os.makedirs(dir)
        except OSError:
            print(f"This file {dir} exist.")

        fnames = [f"{k}.{i}.jpg" for i in range(start, end)]
        for fname in fnames:
            shutil.copyfile(src=path / fname,
                            dst=dir / fname)


def proces():
    # unzipFiles()
    seperating("test", start=0, end=500)
    seperating("train", start=0, end=500)

    pth_test = "./train_validation/test"
    pth_train = "./train_validation/train"


    train_btch = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=pth_train, target_size=(224, 224), classes=["cat", "dog"], batch_size=10)

    validation_btch = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input) \
        .flow_from_directory(directory=pth_test, target_size=(224, 224), classes=["cat", "dog"], batch_size=10)

    vgg16_model = tf.keras.applications.vgg16.VGG16()

    vgg16_model.summary()

    model = Sequential()

    for layer in vgg16_model.layers[:-1]:
        model.add(layer)

    model.summary()

    for layer in model.layers:
        layer.trainable = False
    model.add(Dense(2, activation="softmax"))
    model.summary()

    model.compile(loss="categorical_crossentropy", optimizer=optimizers.RMSprop(lr=1e-3),
                  metrics=["accuracy"])

    model.fit(x=train_btch, validation_data=validation_btch,
              epochs=5 )

    model.save('.')


