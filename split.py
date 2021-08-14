#!/usr/bin/env python
# coding: utf-8

# In[1]:


#!pip install --upgrade git+https://github.com/EmGarr/kerod.git


# In[2]:


import warnings
warnings.filterwarnings('ignore')


# In[3]:


import  functools
import  tensorflow as tf
import  tensorflow_datasets as tfds
from    tensorflow.keras.utils import to_categorical
import  matplotlib.pyplot as plt
from    tensorflow.keras import layers
from    tensorflow.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
import pandas as pd
from    tqdm import tqdm
import  time
from    sklearn.metrics import accuracy_score
import argparse

from common.config import Config
from common.logger import Logger
from common.communication import Client
from common.communication import Server
from common.helper import ImagesInfo 
from common.timekeeper import TimeKeeper
from common.helper import read_image, filt_text, get_predictions
from CaptionModel import CaptionModel
from common.helper import read_image, filt_text, get_predictions,process_predictions,get_reshape_size


# In[4]:


parser = argparse.ArgumentParser()
parser.add_argument('-s', '--server', action='store', type=str, required=False)
parser.add_argument('-t', '--test_number', action='store', type=int, required=False)
parser.add_argument('-l', '--split_layer', action='store', type=int, required=False)
parser.add_argument('-v', '--verbose', action='store', type=int, required=False)
parser.add_argument('-i', '--image_size', action='store', type=int, required=False)
parser.add_argument('-m', '--max_tests', action='store', type=int, required=False)
args, unknown = parser.parse_known_args()

split_layer = args.split_layer

if(split_layer == None):
    split_layer = 3

Logger.milestone_print("Splitting at layer %d" % split_layer)


# In[5]:


data_dir='/home/suphale/coco'
split_train = "train[:1%]"
split_val = "validation[:1%]"
image_size = 250


# In[6]:


tk = TimeKeeper()
cfg = Config()
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[7]:


from tensorflow import keras  # or import keras for standalone version
from tensorflow.keras.layers import Input

org_model = tf.keras.models.load_model(cfg.saved_model_path + '/iv3_full_model', compile=False)

# basic_model = tf.keras.models.load_model(cfg.saved_model_path + '/model')
# org_model = tf.keras.Model(inputs=basic_model.input, 
#         outputs=[basic_model.get_layer('mixed10').output, basic_model.get_layer('dense_1').output] )


# In[8]:


model_config = org_model.get_config()


# In[9]:


max_layer_index = len(model_config['layers']) - 1


# In[10]:



head_model_config = {}
head_model_config['name'] = 'head_model'
head_model_config['layers'] = []
head_model_config['input_layers'] = [[model_config['layers'][0]['name'],0,0]]
head_model_config['output_layers'] = [[model_config['layers'][split_layer+1-1]['name'],0,0]]

for index in range(split_layer+1):
    head_model_config['layers'].append(model_config['layers'][index])

print("Final layer of head model [%d] %s" % (split_layer, model_config['layers'][split_layer]['name']) )


# In[11]:


# Last layer of the head model
last_head_model_layer = head_model_config['layers'][split_layer]['name']
# print(last_head_model_layer)


# In[12]:


# First layer of the tail model
print("First layer of tail model [%d] %s" % (split_layer+1, model_config['layers'][split_layer+1]['name']) )


# In[13]:


import copy

tail_model_config = copy.deepcopy(model_config)
tail_model_config['name'] = 'tail_model'
tail_model_config['input_layers'] = [[model_config['layers'][split_layer+1]['name'],0,0]]
# tail_model_config['output_layers'] = [[model_config['layers'][max_layer_index]['name'],0,0]]
tail_model_config['output_layers'] = model_config['output_layers']


# In[14]:


# print(tail_model_config['input_layers'])


# In[15]:


# print(tail_model_config['output_layers'])


# In[16]:


new_input_layer = {
                      'name': 'new_input',
                      'class_name': 'InputLayer',
                      'config': {
                          'batch_input_shape': tuple(org_model.layers[split_layer+1-1].output.shape),
                          'dtype': 'float32',
                          'sparse': False,
                          'name': 'new_input'
                      },
                      'inbound_nodes': []
                  }
tail_model_config['layers'][0] = new_input_layer


# In[17]:


for index in range(1,split_layer+1):
    # print("%d %s" % (index, tail_model_config['layers'][1]['name']) )
    tail_model_config['layers'].pop(1)


# In[18]:


import numpy as np

# Find if any layer in the tail model takes the last layer of head model as input
# substitute it with the input layer
# Ideally we should check if any tail model layer refers to any head model layer ToDo
for index, layer in enumerate(tail_model_config['layers']):
    if (np.shape(layer['inbound_nodes'])[0] > 0):
        dim_1 = len(layer['inbound_nodes'][0])
        if(dim_1 >= 1):
            for i in range(dim_1):
                in_layer = layer['inbound_nodes'][0][i][0]
                if(in_layer == last_head_model_layer):
                    print(str(index) + "    " + layer['name'] + " -> " + in_layer )
                    # print(tail_model_config['layers'][index]['inbound_nodes'][0][i])
                    tail_model_config['layers'][index]['inbound_nodes'][0][i] = [[['new_input', 0, 0, {}]]]


# In[19]:


# tail_model_config['layers'][1]['inbound_nodes'] = [[['new_input', 0, 0, {}]]]
tail_model_config['input_layers'] = [['new_input', 0, 0]]


# In[20]:


import pprint
with open(cfg.temp_path + '/model_config.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    # fh.write(str(model_config))
    print(model_config,file=fh)
with open(cfg.temp_path + '/head_model_config.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    # fh.write(str(new_head_model_config))
    print(head_model_config,file=fh)
with open(cfg.temp_path + '/tail_model_config.txt','w') as fh:
    # Pass the file handle in as a lambda function to make it callable
    # fh.write(str(new_head_model_config))
    print(tail_model_config,file=fh)


# In[21]:


head_model = org_model.__class__.from_config(head_model_config, custom_objects={})


# In[22]:


tail_model = org_model.__class__.from_config(tail_model_config, custom_objects={})


# In[23]:


with open(cfg.temp_path + '/org_model.txt','w') as fh:
    org_model.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=150)

with open(cfg.temp_path + '/head_model.txt','w') as fh:
    head_model.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=150)

with open(cfg.temp_path + '/tail_model.txt','w') as fh:
    tail_model.summary(print_fn=lambda x: fh.write(x + '\n'), line_length=150)


# In[24]:


for index, layer in enumerate(org_model.layers[:split_layer+1]):
    # print("[%d] %s %s" % (index, layer.name, str(np.shape(weight))))
    weight = layer.get_weights()
    new_head_model_layer = head_model.layers[index]
    new_head_model_layer.set_weights(weight)


# In[25]:


import numpy as np
for index, layer in enumerate(org_model.layers[split_layer+1:max_layer_index+1]):
    weight = layer.get_weights()
    # print("org_model [%d] %s %s" % (index, layer.name, str(tf.shape(weight))))
    tail_model_layer = tail_model.layers[index+1]
    tail_model_layer_weight = tail_model_layer.get_weights()
    # print("tail_model [%d] %s %s" % (index, tail_model_layer.name, str(tf.shape(tail_model_layer_weight))))
    tail_model_layer.set_weights(weight)


# In[26]:


head_model.save(cfg.temp_path + '/iv3_head_model_'+str(split_layer))
tail_model.save(cfg.temp_path + '/iv3_tail_model_'+str(split_layer))


# In[27]:


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
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.

    targets = inputs['objects']
    img_path = inputs['image/filename']

    image_information = tf.cast(tf.shape(image)[:2], dtype=tf.float32)

    inputs = {DatasetField.IMAGES: image, DatasetField.IMAGES_INFO: image_information}

    # ground_truths = {
    #     BoxField.BOXES: targets[BoxField.BOXES] * tf.tile(image_information[tf.newaxis], [1, 2]),
    #     BoxField.LABELS: tf.cast(targets[BoxField.LABELS], tf.int32),
    #     BoxField.NUM_BOXES: tf.shape(targets[BoxField.LABELS]),
    #     BoxField.WEIGHTS: tf.fill(tf.shape(targets[BoxField.LABELS]), 1.0)
    # }
    ground_truths = tf.cast(targets[BoxField.LABELS], tf.int32)
    # ground_truths = tf.one_hot(ground_truths, depth=N_LABELS, dtype=tf.int32)
    # ground_truths = tf.reduce_sum(ground_truths, 0)
    # ground_truths = tf.greater( ground_truths, tf.constant( 0 ) )    
    # ground_truths = tf.where (ground_truths, 1, 0) 
    return image, ground_truths, img_path

def expand_dims_for_single_batch(image, ground_truths, img_path):
    image = tf.expand_dims(image, axis=0)
    ground_truths = tf.expand_dims(ground_truths, axis=0)
    return image, ground_truths, img_path


# In[28]:


ds_val, ds_info = tfds.load(name="coco/2017", split=split_val, data_dir=data_dir, shuffle_files=False, download=False, with_info=True)
ds_val = ds_val.map(functools.partial(my_preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)


# In[29]:


from   nltk.translate.bleu_score import sentence_bleu
def process_caption_predictions(caption_tensor, img_path):
    pred_caption=' '.join(caption_tensor).rsplit(' ', 1)[0]
    real_appn = []
    real_caption_list = imagesInfo.annotations_dict[img_path]
    for real_caption in real_caption_list:
        real_caption=filt_text(real_caption)
        real_appn.append(real_caption.split())
    reference = real_appn
    candidate = pred_caption.split()
    score = sentence_bleu(reference, candidate, weights=[1]) #set your weights)
    return score,real_caption,pred_caption


# In[30]:


Test = False
if (Test == True):
    captionModel = CaptionModel()
    real_caption_list = []
    pred_caption_list = []
    max_test_count = 10
    ds_val = ds_val.take(max_test_count)
    for sample_img_batch, ground_truth, img_path in tqdm(ds_val):
        # count += 1

        tensor_shape = len(ground_truth.get_shape().as_list())
        if(tensor_shape > 1):
            ground_truth = tf.squeeze(ground_truth,[0])
        ground_truth = list(set(ground_truth.numpy()))

        img_path = img_path.numpy().decode()
        h = head_model(sample_img_batch)
        features, result = tail_model(h)
        # features, result = model(sample_img_batch)
        predictions, predictions_prob = get_predictions(cfg, result)
        accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str = process_predictions(cfg, imagesInfo, ground_truth,predictions, predictions_prob)

        reshape_layer_size = get_reshape_size(image_size)
        features = tf.reshape(features, [sample_img_batch.shape[0],reshape_layer_size*reshape_layer_size, 2048])
        caption_tensor = captionModel.evaluate(features)

        score,real_caption,pred_caption = process_caption_predictions(caption_tensor, img_path)

        real_caption_list.append(real_caption)
        pred_caption_list.append(pred_caption)

    for i in range(max_test_count):
        print ('Real:', real_caption_list[i])
        print ('Pred:', pred_caption_list[i])    


# In[31]:


Test = True
if (Test == True):
    captionModel = CaptionModel(image_size=image_size)
    count = 0
    ds_val = ds_val.take(2)
    for sample_img_batch, ground_truth, img_path in ds_val:
        count += 1

        tensor_shape = len(ground_truth.get_shape().as_list())
        if(tensor_shape > 1):
            ground_truth = tf.squeeze(ground_truth,[0])
        ground_truth = list(set(ground_truth.numpy()))

        img_path = img_path.numpy().decode()
        print(img_path)
        # features, result = org_model(sample_img_batch)
        h = head_model(sample_img_batch)
        features, result = tail_model(h)

        predictions, predictions_prob = get_predictions(cfg, result)
        accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str = process_predictions(cfg, imagesInfo, ground_truth,predictions, predictions_prob)
        # print(predictions_str)

        features = tf.reshape(features, [sample_img_batch.shape[0],get_reshape_size(image_size)*get_reshape_size(image_size), 2048])
        caption_tensor = captionModel.evaluate(features)

        # print(type(caption_tensor))
        score,real_caption,pred_caption = process_caption_predictions(caption_tensor, img_path)

        print("BLEU: %.2f" % (score))
        print ('Real:', real_caption)
        print ('Pred:', pred_caption)    


# In[ ]:


if (False):
    for index, layer in enumerate(model_config['layers']):
        if (np.shape(layer['inbound_nodes'])[0] > 0):
            dim_1 = len(layer['inbound_nodes'][0])
            if(dim_1 > 1):
                print(index, layer['name'])
                for i in range(dim_1):
                    print("    %s" % (layer['inbound_nodes'][0][i][0]))
                    # print(tail_model_config['layers'][index]['inbound_nodes'][0][i])


# In[ ]:


# org_model.save(cfg.temp_path + '/full_model')


# In[ ]:




