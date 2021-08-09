from keras.models import Model
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
from tensorflow.keras.metrics import categorical_crossentropy
from sklearn.metrics import confusion_matrix
import itertools
import shutil
import glob
import random
import matplotlib.pyplot as plt
import numpy as np
import os
import src.preprocessing_model as preprocessing
import pandas as pd
from keras.preprocessing.image import ImageDataGenerator, load_img



if(not os.path.exists('saved_model.pb')):

    preprocessing.proces()

path = os.getcwd()

model = keras.models.load_model(os.getcwd())

cm_labels = ["cat", "dog"]

# 

test_filenames = os.listdir("./test1/")
test_df = pd.DataFrame({
    'filename': test_filenames
})
nb_samples = test_df.shape[0]

test_gen = ImageDataGenerator(rescale=1./255)
test_generator = test_gen.flow_from_dataframe(
    test_df, 
    "./test1/", 
    x_col='filename',
    y_col=None,
    class_mode=None,
    batch_size=10,
    target_size=(224, 224),
    shuffle=False
)

predict = model.predict_generator(test_generator, steps=np.ceil(nb_samples/10))
threshold = 0.5
# test_df['category'] = 
print(np.where(predict > threshold, 1,0))

print(predict)

# test_df['category'] = 
print(np.where(predict > threshold, 1,0))

# print(test_df)


# sample_test = test_df.sample(n=9).reset_index()
# sample_test.head()
# plt.figure(figsize=(12, 12))
# for index, row in sample_test.iterrows():
#     filename = row['filename']
#     category = row['category']
