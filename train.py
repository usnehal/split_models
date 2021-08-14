#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!pip install --upgrade git+https://github.com/EmGarr/kerod.git


# In[6]:


#%tensorflow_version 2.x
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
    print('GPU device not found')
else:
    print('Found GPU at: {}'.format(device_name))


# In[7]:


import warnings
warnings.filterwarnings('ignore')


# In[8]:


import  functools
import  tensorflow as tf
import  tensorflow_datasets as tfds
from    tensorflow.keras.utils import to_categorical
import  matplotlib.pyplot as plt
from    tensorflow.keras import layers
from    tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from    tensorflow.keras.layers import Dense, GlobalAveragePooling2D


# In[9]:


from common.constants import test, BoxField, DatasetField
from common.config import Config
from common.logger import Logger
from common.communication import Client
from common.communication import Server
from common.helper import ImagesInfo 
from common.timekeeper import TimeKeeper
from common.helper import read_image, filt_text, get_predictions,process_predictions
from CaptionModel import CaptionModel


# In[10]:


data_dir='/home/suphale/coco'
N_LABELS = 80
N_EPOCHS = 1
TRAIN_MODE = False
# split_train = "train"
# split_val = "validation"
split_train = "train[:1%]"
split_val = "validation[:1%]"
h_image_height = 299
h_image_width = 299


# In[11]:


tk = TimeKeeper()
cfg = Config()
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[12]:


class BoxField:
    BOXES = 'bbox'
    KEYPOINTS = 'keypoints'
    LABELS = 'label'
    MASKS = 'masks'
    NUM_BOXES = 'num_boxes'
    SCORES = 'scores'
    WEIGHTS = 'weights'

class DatasetField:
    IMAGES = 'images'
    IMAGES_INFO = 'images_information'
    IMAGES_PMASK = 'images_padding_mask'

def my_preprocess(inputs):
    image = inputs['image']
    image = tf.image.resize(image, (h_image_height, h_image_width))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.

    targets = inputs['objects']

    image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    inputs = {DatasetField.IMAGES: image, DatasetField.IMAGES_INFO: image_information}

    # ground_truths = {
    #     BoxField.BOXES: targets[BoxField.BOXES] * tf.tile(image_information[tf.newaxis], [1, 2]),
    #     BoxField.LABELS: tf.cast(targets[BoxField.LABELS], tf.int32),
    #     BoxField.NUM_BOXES: tf.shape(targets[BoxField.LABELS]),
    #     BoxField.WEIGHTS: tf.fill(tf.shape(targets[BoxField.LABELS]), 1.0)
    # }
    ground_truths = tf.cast(targets[BoxField.LABELS], tf.int32)
    ground_truths = tf.one_hot(ground_truths, depth=N_LABELS, dtype=tf.int32)
    ground_truths = tf.reduce_sum(ground_truths, 0)
    ground_truths = tf.greater( ground_truths, tf.constant( 0 ) )    
    ground_truths = tf.where (ground_truths, 1, 0) 
    return image, ground_truths

def expand_dims_for_single_batch(image, ground_truths):
    image = tf.expand_dims(image, axis=0)
    ground_truths = tf.expand_dims(ground_truths, axis=0)
    return image, ground_truths


# In[13]:


ds_train, ds_info = tfds.load(name="coco/2017", split=split_train, data_dir=data_dir, shuffle_files=True, download=False, with_info=True)
ds_train = ds_train.map(functools.partial(my_preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_train = ds_train.prefetch(tf.data.experimental.AUTOTUNE)

ds_val = tfds.load(name="coco/2017", split=split_val, data_dir=data_dir, shuffle_files=True, download=False)
ds_val = ds_val.map(functools.partial(my_preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)


# In[15]:


# ds_info


# # Load and train the network
# 

# In[16]:


# Find total number of classes in the coco dataset
classes = ds_info.features['objects']['label'].names
num_classes = len(classes)
print(num_classes)


# In[17]:


image_model = tf.keras.applications.InceptionV3(include_top=False,weights='imagenet')

x = image_model.output
x = GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
predictions = tf.keras.layers.Dense(80, activation='sigmoid')(x)

model = tf.keras.Model(inputs=image_model.input, outputs=predictions)

for layer in image_model.layers:
    layer.trainable = False

# model.compile(optimizer='rmsprop', loss=ncce, metrics=['accuracy'])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
# model.compile(optimizer='rmsprop', loss='categorical_crossentropy')


# In[18]:


for layer in model.layers:
    if(layer.trainable == True):
        print(layer.trainable)


# In[19]:


callbacks = [
    TensorBoard(),
    ModelCheckpoint('checkpoints/')
]

model.fit(ds_train, validation_data=ds_val, epochs=N_EPOCHS, callbacks=callbacks)
# Save the weights for the serving
model.save_weights(cfg.temp_path + '/coco_classification_weights.h5')
model.save(cfg.temp_path + '/model')


# ## Visualisation of few images

# In[20]:


for test_index in range(10):
    sample_img_batch, sample_cap_batch = next(iter(ds_val))
    s = model(sample_img_batch)
    # plt.imshow(tf.squeeze(sample_img_batch, [0]))

    print("Reference  : ", end=' ')
    n = sample_cap_batch.numpy()
    index = 0
    for x in n[0]:
        if x > 0.1:
            print("%s," % (classes[index]), end=' ')
        index += 1
    print("")
    print("Prediction : ", end=' ')
    n = s.numpy()
    index = 0
    for x in n[0]:
        if x > 0.5:
            print("%s(%.2f)," % (classes[index],x), end=' ')
        index += 1
    print("")


# ## Tensorboard

# In[ ]:


# Load TENSORBOARD
get_ipython().run_line_magic('load_ext', 'tensorboard')
# Start TENSORBOARD
get_ipython().run_line_magic('tensorboard', '--logdir logs')


# In[ ]:


# ds_info


# In[ ]:




