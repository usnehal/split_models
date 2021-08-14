#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python3


# In[1]:


import socket
import os
import json
import  time
import tensorflow as tf
import numpy as np
from   nltk.translate.bleu_score import sentence_bleu
import random
import  re
import sys
import argparse
import zlib
import pickle5 as pickle
import pandas as pd
from tqdm import tqdm
import  tensorflow_datasets as tfds
import  functools
import pprint

tf.get_logger().setLevel('ERROR')

from common.constants import test, BoxField, DatasetField
from common.config import Config
from common.logger import Logger
from common.communication import Client
from common.communication import Server
from common.helper import ImagesInfo 
from common.timekeeper import TimeKeeper
from common.helper import read_image, filt_text, get_predictions,process_predictions,get_reshape_size,process_caption_predictions
from CaptionModel import CaptionModel


# In[ ]:


tf.get_logger().setLevel('ERROR')


# In[ ]:


parser = argparse.ArgumentParser()
parser.add_argument('-r', '--result_folder', action='store', type=str, required=False)
parser.add_argument('-s', '--server', action='store', type=str, required=False)
parser.add_argument('-t', '--test_number', action='store', type=int, required=False)
parser.add_argument('-l', '--split_layer', action='store', type=int, required=False)
parser.add_argument('-v', '--verbose', action='store', type=int, required=False)
parser.add_argument('-i', '--image_size', action='store', type=int, required=False)
parser.add_argument('-m', '--max_tests', action='store', type=int, required=False)
args, unknown = parser.parse_known_args()

result_folder = args.result_folder
server_ip = args.server
test_number = args.test_number
verbose = args.verbose
reshape_image_size = args.image_size
max_tests = args.max_tests
split_layer = args.split_layer

if(verbose == None):
    verbose = 1

if(result_folder == None):
    result_folder = 'temp'

if(split_layer == None):
    split_layer = 3

test_scenarios = {  
        test.STANDALONE:                 "complete on-device processing", 
        test.RGB_IMAGE_TRANSFER:         "RGB buffer transfer",
        test.RGB_IMAGE_TRANSFER_ZLIB:    "RGB buffer with zlib compression",
        test.JPEG_TRANSFER:              "image buffer with jpeg compression", 
        test.SPLIT_LAYER:                "split model, intermediate tensor",
        test.SPLIT_LAYER_ZLIB:           "split model, intermediate tensor, zlib compression",
        test.SPLIT_LAYER_QUANTIZED:      "split model, intermediate tensor, quantized",
        test.SPLIT_LAYER_QUANTIZED_ZLIB: "split model, intermediate tensor, quantized with zlib compression",
        }

# test_number = 2
if(test_number == None):
    test_number = test.STANDALONE
elif(test_number == 1):
    test_number = test.STANDALONE
elif(test_number == 2):
    test_number = test.RGB_IMAGE_TRANSFER
elif(test_number == 3):
    test_number = test.RGB_IMAGE_TRANSFER_ZLIB
elif(test_number == 4):
    test_number = test.JPEG_TRANSFER
elif(test_number == 5):
    test_number = test.SPLIT_LAYER
elif(test_number == 6):
    test_number = test.SPLIT_LAYER_ZLIB
elif(test_number == 7):
    test_number = test.SPLIT_LAYER_QUANTIZED
elif(test_number == 8):
    test_number = test.SPLIT_LAYER_QUANTIZED_ZLIB
else:
    print(test_scenarios)

if(reshape_image_size == None):
    reshape_image_size = 250

if(max_tests == None):
    max_tests = 50
elif (((max_tests % 50) == 0) and (max_tests <= 5000)):
    max_tests = max_tests
else:
    print("max_tests must be multiple of 50 and less than or equal to 5000")
    exit(1)

# print("Test scenario[%d] %s" % (test_number, test_scenarios[test_number]))


# In[ ]:


# tf.compat.v1.disable_eager_execution()
tf.get_logger().setLevel('ERROR')


# In[ ]:


Logger.set_log_level(verbose)
tk = TimeKeeper()
cfg = Config(server_ip)
client = Client(cfg)
imagesInfo = ImagesInfo(cfg)


# In[ ]:


from CaptionModel import CaptionModel


# In[ ]:


data_dir='/home/suphale/coco'
N_LABELS = 80
# split_val = "validation"
split_val = "validation[:20%]"
# split_val = "validation[:1%]"

# split_layer = 3

Logger.milestone_print("Test scenario   : [%d] %s" % (test_number, test_scenarios[test_number]))
Logger.milestone_print("Image shape     : (%d %d)" % (reshape_image_size, reshape_image_size))
Logger.milestone_print("Max tests       : %d" % (max_tests))
if ((test_number == test.SPLIT_LAYER) or (test_number == test.SPLIT_LAYER_ZLIB)):
    Logger.milestone_print("split layer     : [%d]" % (split_layer))


# In[ ]:


def my_preprocess(inputs):
    image = inputs['image']
    # image = tf.image.resize(image, (image_size, image_size))
    # image = tf.cast(image, tf.float32)
    # image /= 127.5
    # image -= 1.

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


# In[ ]:


# tf.compat.v1.disable_eager_execution()


# In[ ]:


if(test_number in [test.STANDALONE]):
    model_path = cfg.saved_model_path + '/iv3_full_model'
    Logger.event_print("Loading model : from %s" % (model_path))
    model = tf.keras.models.load_model(model_path, compile=False)
    # model = tf.keras.Model(inputs=model.inputs,outputs=[ 
    #                             model.layers[310].output, 
    #                             model.layers[313].output])
    print("Finished loading")
    captionModel = CaptionModel(image_size=reshape_image_size)
if(test_number in [test.JPEG_TRANSFER, test.RGB_IMAGE_TRANSFER, test.RGB_IMAGE_TRANSFER_ZLIB]):
    # head_model = tf.keras.models.load_model(cfg.saved_model_path + '/model', compile=False)
    send_json_dict = {}
    send_json_dict['request'] = 'load_model_request'
    send_json_dict['model'] = 'model'
    send_json_dict['model_path'] = '/iv3_full_model'
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')

    send_json_dict = {}
    send_json_dict['request'] = 'load_model_request'
    send_json_dict['model'] = 'captionModel'
    send_json_dict['model_path'] = '/caption_i_%d' % (reshape_image_size)
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')


if(test_number in [test.SPLIT_LAYER, test.SPLIT_LAYER_ZLIB, test.SPLIT_LAYER_QUANTIZED ,test.SPLIT_LAYER_QUANTIZED_ZLIB]):
    model_path = cfg.saved_model_path + '/iv3_head_model_%d' % (split_layer)
    Logger.event_print("Loading model : from %s" % (model_path))
    head_model = tf.keras.models.load_model(model_path, compile=False)
    print("Finished loading")
    send_json_dict = {}
    send_json_dict['request'] = 'load_model_request'
    send_json_dict['model'] = 'tail_model'
    send_json_dict['model_path'] = '/iv3_tail_model_%d' % (split_layer)
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')

    send_json_dict = {}
    send_json_dict['request'] = 'load_model_request'
    send_json_dict['model'] = 'captionModel'
    send_json_dict['model_path'] = '/caption_i_%d' % (reshape_image_size)
    app_json = json.dumps(send_json_dict)
    response = client.send_load_model_request(str(app_json))
    assert(response == 'OK')


# In[ ]:


def preprocess_image(image):
    image = tf.squeeze(image,[0])
    image = tf.image.resize(image, (reshape_image_size, reshape_image_size))
    image = tf.cast(image, tf.float32)
    image /= 127.5
    image -= 1.
    image = tf.expand_dims(image, 0)
    return image


# In[ ]:


# @tf.function
def handle_test_STANDALONE(sample_img_batch, img_path):
    sample_img_batch = preprocess_image(sample_img_batch)
    
    features, result = model(sample_img_batch)

    reshape_layer_size = get_reshape_size(reshape_image_size)
    features = tf.reshape(features, [sample_img_batch.shape[0], reshape_layer_size*reshape_layer_size, 2048])
    caption_tensor = captionModel.evaluate(features)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, 0)

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    predictions, predictions_prob = get_predictions(cfg, result)

    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, 0)

    return predictions, predictions_prob, caption_tensor


# In[ ]:


def handle_test_JPEG_TRANSFER(sample_img_batch, file_name):
    image = tf.squeeze(sample_img_batch,[0]) 
    image_np_array = image.numpy()
    image_tensor = tf.io.encode_jpeg(image)
    byte_buffer_to_send = image_tensor.numpy()

    send_json_dict = {}
    send_json_dict['request'] = 'jpeg_buffer'
    send_json_dict['file_name'] = file_name
    send_json_dict['data_size'] = (len(byte_buffer_to_send))
    send_json_dict['data_shape'] = image_np_array.shape
    send_json_dict['reshape_image_size'] = reshape_image_size

    app_json = json.dumps(send_json_dict)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    response = client.send_data(str(app_json), byte_buffer_to_send)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    response = json.loads(response)

    predictions = response['predictions']
    predictions_prob = response['predictions_prob']
    caption_tensor = response['predicted_captions']
    tail_model_time = response['tail_model_time']
    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

    return predictions, predictions_prob, caption_tensor


# In[ ]:


def handle_test_RGB_IMAGE_TRANSFER(sample_img_batch,file_name, zlib_compression=False):
    image_tensor = tf.cast(sample_img_batch, tf.float32)
    image_np_array = image_tensor.numpy()
    byte_buffer_to_send = image_np_array.tobytes()
    if(zlib_compression == True):
        byte_buffer_to_send = zlib.compress(byte_buffer_to_send)

    send_json_dict = {}
    send_json_dict['request'] = 'rgb_buffer'
    send_json_dict['file_name'] = file_name
    send_json_dict['data_size'] = (len(byte_buffer_to_send))
    send_json_dict['data_shape'] = image_np_array.shape
    send_json_dict['reshape_image_size'] = reshape_image_size
    if(zlib_compression == True):
        send_json_dict['zlib_compression'] = 'yes'
    else:
        send_json_dict['zlib_compression'] = 'no'

    app_json = json.dumps(send_json_dict)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    response = client.send_data(str(app_json), byte_buffer_to_send)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    response = json.loads(response)

    predictions = response['predictions']
    predictions_prob = response['predictions_prob']
    tail_model_time = response['tail_model_time']
    caption_tensor = response['predicted_captions']
    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

    return predictions, predictions_prob, caption_tensor


# In[ ]:


import pprint

def handle_test_SPLIT_LAYER(sample_img_batch,file_name, quantized=False, zlib_compression=False):

    sample_img_batch = preprocess_image(sample_img_batch)
    intermediate_tensor = head_model(sample_img_batch)
    if(quantized == True):
        image_np_array = tf.cast(intermediate_tensor, dtype=tf.int8).numpy()
    else:
        image_np_array = intermediate_tensor.numpy()
    byte_buffer_to_send = image_np_array.tobytes()

    if(zlib_compression == True):
        byte_buffer_to_send = zlib.compress(byte_buffer_to_send)

    send_json_dict = {}
    send_json_dict['request'] = 'intermediate_tensor'
    send_json_dict['file_name'] = file_name
    send_json_dict['data_size'] = (len(byte_buffer_to_send))
    send_json_dict['data_shape'] = image_np_array.shape
    send_json_dict['reshape_image_size'] = reshape_image_size
    if(zlib_compression == True):
        send_json_dict['zlib_compression'] = 'yes'
    else:
        send_json_dict['zlib_compression'] = 'no'
    if(quantized == True):
        send_json_dict['quantized'] = 'yes'
    else:
        send_json_dict['quantized'] = 'no'

    app_json = json.dumps(send_json_dict)

    tk.logInfo(img_path, tk.I_BUFFER_SIZE, len(byte_buffer_to_send))

    tk.logTime(img_path, tk.E_START_COMMUNICATION)

    response = client.send_data(str(app_json), byte_buffer_to_send)

    tk.logTime(img_path, tk.E_STOP_COMMUNICATION)

    response = json.loads(response)

    predictions = response['predictions']
    predictions_prob = response['predictions_prob']
    caption_tensor = response['predicted_captions']
    tail_model_time = response['tail_model_time']
    tk.logInfo(img_path, tk.I_TAIL_MODEL_TIME, tail_model_time)

    return predictions, predictions_prob, caption_tensor


# In[ ]:


ds_val, ds_info = tfds.load(name="coco/2017", split=split_val, data_dir=data_dir, shuffle_files=False, download=False, with_info=True)
ds_val = ds_val.map(functools.partial(my_preprocess), num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.map(expand_dims_for_single_batch, num_parallel_calls=tf.data.experimental.AUTOTUNE)
ds_val = ds_val.prefetch(tf.data.experimental.AUTOTUNE)


# In[ ]:


count = 0
max_test_images = max_tests

# coco_image_dir = '/home/suphale/snehal_bucket/coco/raw-data/val2017/'

# df_result = pd.from_csv(cfg.temp_path + '/temp/results.csv')


total_time = 0.0

total_time = 0.0
df = pd.DataFrame(columns=['img_path','ground_truth', 'top_predict', 'Prediction', 'accuracy', 'top_1_accuracy', 'top_5_accuracy', 'precision', 'recall', 'time'])
ds_val = ds_val.take(max_tests)
for sample_img_batch, ground_truth, img_path in tqdm(ds_val):
# for sample_img_batch, ground_truth, img_path in ds_val:
    count += 1
    img_path = img_path.numpy().decode()

    tk.startRecord(img_path)
    tk.logTime(img_path, tk.E_START_CLIENT_PROCESSING)

    tensor_shape = len(ground_truth.get_shape().as_list())
    if(tensor_shape > 1):
        ground_truth = tf.squeeze(ground_truth,[0])

    ground_truth = list(set(ground_truth.numpy()))

    if(test_number == test.STANDALONE):
        predictions,predictions_prob, caption_tensor = handle_test_STANDALONE(sample_img_batch, img_path)
    if(test_number == test.JPEG_TRANSFER):
        predictions,predictions_prob, caption_tensor = handle_test_JPEG_TRANSFER(sample_img_batch, img_path)
    if(test_number == test.RGB_IMAGE_TRANSFER):
        predictions,predictions_prob, caption_tensor = handle_test_RGB_IMAGE_TRANSFER(sample_img_batch, img_path)
    if(test_number == test.RGB_IMAGE_TRANSFER_ZLIB):
        predictions,predictions_prob, caption_tensor = handle_test_RGB_IMAGE_TRANSFER(sample_img_batch, img_path,zlib_compression=True)
    if(test_number == test.SPLIT_LAYER):
        predictions,predictions_prob, caption_tensor = handle_test_SPLIT_LAYER(sample_img_batch, img_path)
    if(test_number == test.SPLIT_LAYER_ZLIB):
        predictions,predictions_prob, caption_tensor = handle_test_SPLIT_LAYER(sample_img_batch, img_path,zlib_compression=True)
    if(test_number == test.SPLIT_LAYER_QUANTIZED):
        predictions,predictions_prob, caption_tensor = handle_test_SPLIT_LAYER(sample_img_batch, img_path, quantized=True)
    if(test_number == test.SPLIT_LAYER_QUANTIZED_ZLIB):
        predictions,predictions_prob, caption_tensor = handle_test_SPLIT_LAYER(sample_img_batch, img_path, quantized=True, zlib_compression=True)

    tk.logTime(img_path, tk.E_STOP_CLIENT_PROCESSING)

    accuracy, top_1_accuracy,top_5_accuracy,precision,recall, top_predictions, predictions_str = process_predictions(cfg, imagesInfo, ground_truth,predictions, predictions_prob)
    bleu_score,real_caption,pred_caption = process_caption_predictions(caption_tensor, img_path, imagesInfo)

    df = df.append(
        {'image':img_path, 
        'ground_truth':(str(imagesInfo.get_segmentation_texts(ground_truth))),
        'top_predict':str(top_predictions),
        'Prediction':predictions_str,
        'accuracy':accuracy,
        'top_1_accuracy':top_1_accuracy,
        'top_5_accuracy':top_5_accuracy,
        'precision':precision,
        'recall':recall,
        'BLEU':bleu_score,
        'real_caption':real_caption,
        'pred_caption':pred_caption,
        'time':0,
        },
        ignore_index = True)
    truth_str = ' '.join([str(elem) for elem in imagesInfo.get_segmentation_texts(ground_truth)])

    tk.finishRecord(img_path)

from pathlib import Path
Path(cfg.temp_path + '/results/' + result_folder).mkdir(parents=True, exist_ok=True)

fname = cfg.temp_path + '/results/' + result_folder + '/test_%d_i_%d_s_%d.csv' % (test_number, reshape_image_size, split_layer)
os.makedirs(os.path.dirname(fname),exist_ok=True)

df.to_csv(fname)

print(df['real_caption'].iloc[0])
print(df['pred_caption'].iloc[0])

av_column = df.mean(axis=0)
Logger.milestone_print("----------------:")
Logger.milestone_print("Test scenario   : %d %s" % (test_number, test_scenarios[test_number]))
Logger.milestone_print("Image shape     : (%d %d)" % (reshape_image_size, reshape_image_size))
Logger.milestone_print("Max tests       : %d" % (max_tests))
Logger.milestone_print("accuracy        : %.2f" % (av_column.accuracy))
Logger.milestone_print("top_1_accuracy  : %.2f" % (av_column.top_1_accuracy))
Logger.milestone_print("top_5_accuracy  : %.2f" % (av_column.top_5_accuracy))
Logger.milestone_print("precision       : %.2f" % (av_column.precision))
Logger.milestone_print("recall          : %.2f" % (av_column.recall))
Logger.milestone_print("BLEU            : %.2f" % (av_column.BLEU))
Logger.milestone_print("time            : %.2f" % (av_column.time))

# tk.printAll()
tk.summary()
    


# In[ ]:


fname = cfg.temp_path + '/results/' + result_folder + '/results.csv'
if(os.path.isfile(fname) == True):
    df_result = pd.read_csv(fname,dtype={
                     'test_number': int,
                     'image_size': int,
                     'split_layer': int,
                     'accuracy': float,
                     'top_1_accuracy': float,
                     'top_5_accuracy': float,
                     'precision': float,
                     'recall': float,
                     'BLEU': float,
                     'total_time': float,
                     'head_time': float,
                     'network_time': float,
                     'tail_time': float,
                     'nw_payload': float})
else:
    print("Results in %s" % fname)
    df_result = pd.DataFrame(columns=[  'test_number',
                                        'image_size',
                                        'split_layer',
                                        'accuracy', 
                                        'top_1_accuracy', 
                                        'top_5_accuracy', 
                                        'precision', 
                                        'recall', 
                                        'BLEU', 
                                        'total_time',
                                        'head_time',
                                        'network_time',
                                        'tail_time',
                                        'nw_payload'])

new_row = { 'test_number': test_number,
            'image_size': reshape_image_size,
            'split_layer': split_layer,
            'accuracy': float(av_column.accuracy),
            'top_1_accuracy': float(av_column.top_1_accuracy),
            'top_5_accuracy': float(av_column.top_5_accuracy),
            'precision': float(av_column.precision),
            'recall': float(av_column.recall),
            'BLEU': float(av_column.BLEU),
            'total_time': float(tk.get_average_inference_time()),
            'head_time': float(tk.get_average_head_model_time()),
            'network_time': float(tk.get_average_communication_time()),
            'tail_time': float(tk.get_average_tail_model_time()),
            'nw_payload': float(tk.get_average_communication_payload())}
df_result = df_result.append(new_row, ignore_index=True)
df_result.to_csv(fname,index = False)


# In[ ]:


# ds_info


# In[ ]:




