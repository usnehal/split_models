# Split Models
This repository contains scripts for spliting a tensorflow model into head and tail model and evaluate it.

## Description
Image segmentation and video analytics are resource hungry. Many of these models can't be run on customer devices due to resource-constraint or unacceptable inference time. In such cases, we may wish to split the AI models into head and tail models which can be run on customer device and cloud backend respectively. Some benefits of this architecture are improved inference time, reduced network payload etc.
This repository contains python code to split a tensorflow model (as example InceptionV3 model).

## Important files
### split.ipynb
split an InceptionV3 model based on a give split layer

### train_captions.ipynb
train a caption generation model using InceptionV3 features.

### client.y and server.py
client and server scripts to evalulate the performance of head and tail models. Tested on Raspberry Pi 4 and GCP instance.
