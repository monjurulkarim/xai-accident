import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import os.path as osp
import pandas as pd
import shutil

destination_negative = 'train/negative/'
destination_positive = 'train/positive/'

anno_file = 'train.txt'

with open(anno_file, 'r') as f:
    for line in f.readlines():
        filename = line.split('.')[0]
        file_type = filename.split('/')[0]
        if file_type == 'negative':
            shutil.move(filename, destination_negative)
        else:
            shutil.move(filename, destination_positive)
