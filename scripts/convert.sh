#!/bin/bash

set -e

cd /home/suphale/WorkSpace

jupyter nbconvert server.ipynb --to python
jupyter nbconvert client.ipynb --to python
#jupyter nbconvert inference.ipynb --to python
jupyter nbconvert train.ipynb --to python
jupyter nbconvert split.ipynb --to python

chmod 777 server.py
chmod 777 client.py
#chmod 777 inference.py
chmod 777 train.py
chmod 777 split.py

