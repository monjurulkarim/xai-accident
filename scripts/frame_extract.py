"""
Extracts frames from given video.
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import time
import os
import os.path as osp
import pandas as pd

currentDirectory = os.getcwd()
print(currentDirectory)

video_directory = 'Normal/'
destination = 'negative/'

video_list = os.listdir(video_directory)

count = 0
for i in range(len(video_list)):
    filename = os.path.join(video_directory,video_list[i])
    video_list[i]
    vname_slice = video_list[i].split('.')
    print(vname_slice[0])

    cap = cv2.VideoCapture(filename)
    fps = round(cap.get(cv2.CAP_PROP_FPS))
    print(fps)
#     print(fps)
    images =[]


#     check if capture was successful
    if not cap.isOpened():
        print("Could not open!")
    else:
        print("Video read successful!")
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        print('Extracting frames from: ', video_list[i])
        for loop in range(total_frames):
            cap = cv2.VideoCapture(filename)
            cap.set(1,loop)
#             cap.set(cv2.CAP_PROP_POS_MSEC,(count*55.1))
# #             cv2.CAP_PROP_POS_MSEC,(count*1000)

            success = cap.grab()
            ret, image = cap.retrieve()
            try:
                image = cv2.resize(image,(224,224))
            except:
                continue

            frame_name = vname_slice[0] +'_frame_%d.jpg' %loop
            images.append(frame_name)
            saved_path = '/'+ vname_slice[0]
            destination_2 = destination + saved_path
            if not os.path.exists(destination_2):
                os.makedirs(destination_2)
#                 print(destination_2)

            destination_dir = osp.join(destination_2, frame_name)
            cv2.imwrite(destination_dir,image)
#             if count % hop == 0:
#                 cv2.imwrite(destination_dir,image)
    count = count + 1
    cap.release()
    cv2.destroyAllWindows()
    print('Finished')
print('+++++++++++++++++++++++++++++')
print('Completed : ', count)
