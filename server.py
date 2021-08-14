#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3


# In[1]:


import os
import numpy as np
import tensorflow as tf
from   tensorflow.keras import layers,Model
import pickle5 as pickle
from   tensorflow.keras.preprocessing.text import Tokenizer
from   tensorflow.keras.activations import tanh
from   tensorflow.keras.activations import softmax
from   numpy import float32
from   numpy import byte
import json
import time
import zlib
import pickle5 as pickle

tf.get_logger().setLevel('ERROR')

from common.config import Config
from common.logger import Logger
from common.communication import Client
from common.communication import Server
from common.helper import ImagesInfo 
from common.timekeeper import TimeKeeper
from common.helper import read_image, filt_text, get_predictions, get_reshape_size
from CaptionModel import CaptionModel


# In[2]:


class TailModel:
    def __init__(self,cfg):
        self.cfg = cfg
        self.model = None

    def evaluate(self,image):
        result = self.model(image)
        return result


# In[ ]:


# tf.get_logger().setLevel('ERROR')
tf.get_logger().setLevel('ERROR')


# In[3]:


model = None
captionModel = None

def handle_load_model(msg,model_path_requested):
    global model
    global captionModel
    if(msg == 'model'):
        model_path = cfg.saved_model_path + model_path_requested
        Logger.milestone_print("Loading model : from %s" % (model_path))
        model = None
        model = tf.keras.models.load_model(model_path, compile=False)
        print("finished loading")
        # model = tf.keras.models.load_model(cfg.temp_path + '/extractor_model', compile=False)
        return "OK"
    if(msg == 'captionModel'):
        model_path = cfg.saved_model_path + model_path_requested
        captionModel = None
        Logger.milestone_print("Loading caption model : from %s" % (model_path))
        captionModel = CaptionModel(model_path=model_path)
        print("finished loading")
        return "OK"
    if(msg == 'tail_model'):
        model_path = cfg.saved_model_path + "/" + model_path_requested
        Logger.milestone_print("Loading model : from %s" % (model_path))
        model = None
        model = tf.keras.models.load_model(model_path, compile=False)
        print("finished loading")
        return "OK"


# In[ ]:


def handle_image_file(msg,shape,reshape_image_size,quantized=False,zlib_compression=False):
    
    # temp_file = '/tmp/temp.bin'
    # f = open(temp_file, "wb")
    # f.write(msg)
    # f.close()

    t0 = time.perf_counter()
    image = tf.image.decode_jpeg(bytes(msg), channels=3)
    image = tf.image.resize(image, (reshape_image_size, reshape_image_size))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.
    image_tensor = tf.expand_dims(image, 0) 

    # image_tensor = tf.expand_dims(read_image(temp_file, height=reshape_image_size, width=reshape_image_size), 0) 
    features, result = model(image_tensor)
    reshape_size = get_reshape_size(reshape_image_size)
    features = tf.reshape(features, [1, reshape_size*reshape_size, 2048])
    caption_tensor = captionModel.evaluate(features)
    t1 = time.perf_counter() - t0

    top_predictions, predictions_prob = get_predictions(cfg, result)

    send_json_dict = {}
    send_json_dict['response'] = 'OK'
    send_json_dict['predictions'] = top_predictions
    send_json_dict['predictions_prob'] = predictions_prob
    send_json_dict['predicted_captions'] = caption_tensor
    send_json_dict['tail_model_time'] = t1

    app_json = json.dumps(send_json_dict)

    return str(app_json)


# In[ ]:


def preprocess_image(image,reshape_image_size):
    image = tf.squeeze(image,[0])
    image = tf.image.resize(image, (reshape_image_size, reshape_image_size))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.
    image = tf.expand_dims(image, 0)
    return image


# In[ ]:


def handle_rgb_buffer(msg,shape,reshape_image_size,quantized=False,zlib_compression=False):
    t0 = time.perf_counter()
    if(zlib_compression == True):
        msg = zlib.decompress(msg)
    generated_np_array = np.frombuffer(msg, dtype=float32)
    generated_np_array = np.frombuffer(generated_np_array, dtype=float32)
    generated_image_np_array = generated_np_array.reshape(shape)
    image_tensor = tf.convert_to_tensor(generated_image_np_array, dtype=tf.float32)
    image_tensor = preprocess_image(image_tensor,reshape_image_size)

    features, result = model(image_tensor)
    features = tf.reshape(features, [1,get_reshape_size(reshape_image_size)*get_reshape_size(reshape_image_size), 2048])
    caption_tensor = captionModel.evaluate(features)
    t1 = time.perf_counter() - t0

    top_predictions, predictions_prob = get_predictions(cfg, result)

    send_json_dict = {}
    send_json_dict['response'] = 'OK'
    send_json_dict['predictions'] = top_predictions
    send_json_dict['predictions_prob'] = predictions_prob
    send_json_dict['predicted_captions'] = caption_tensor
    send_json_dict['tail_model_time'] = t1

    app_json = json.dumps(send_json_dict)

    return str(app_json)


# In[ ]:


def handle_image_tensor(msg,shape,reshape_image_size,quantized=False,zlib_compression=False):
    t0 = time.perf_counter()
    if(zlib_compression == True):
        msg = zlib.decompress(msg)
    if(quantized == True):
        generated_np_array = np.frombuffer(msg, dtype=np.uint8)
        # generated_np_array = np.frombuffer(generated_np_array, dtype=float32)
        # generated_image_np_array = generated_np_array.reshape(shape)
        y = tf.bitcast(generated_np_array, tf.uint8)
        image_tensor = tf.cast(y, tf.float32)
        image_tensor = tf.reshape(image_tensor,shape )
        # image_tensor = tf.convert_to_tensor(generated_image_np_array, dtype=tf.float32)
    else:
        generated_np_array = np.frombuffer(msg, dtype=float32)
        generated_np_array = np.frombuffer(generated_np_array, dtype=float32)
        generated_image_np_array = generated_np_array.reshape(shape)
        image_tensor = tf.convert_to_tensor(generated_image_np_array, dtype=tf.float32)

    features, result = model(image_tensor)
    features = tf.reshape(features, [1,get_reshape_size(reshape_image_size)*get_reshape_size(reshape_image_size), 2048])
    caption_tensor = captionModel.evaluate(features)
    t1 = time.perf_counter() - t0

    top_predictions, predictions_prob = get_predictions(cfg, result)

    send_json_dict = {}
    send_json_dict['response'] = 'OK'
    send_json_dict['predictions'] = top_predictions
    send_json_dict['predictions_prob'] = predictions_prob
    send_json_dict['predicted_captions'] = caption_tensor
    send_json_dict['tail_model_time'] = t1

    app_json = json.dumps(send_json_dict)

    return str(app_json)


# In[ ]:


Logger.set_log_level(1)
# logger = Logger()
tk = TimeKeeper()
cfg = Config(None)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[ ]:


tailModel = TailModel(cfg)
server = Server(cfg)
server.register_callback('load_model_request',handle_load_model)
server.register_callback('rgb_buffer',handle_rgb_buffer)
server.register_callback('intermediate_tensor',handle_image_tensor)
server.register_callback('jpeg_buffer',handle_image_file)
server.accept_connections()


# In[ ]:




