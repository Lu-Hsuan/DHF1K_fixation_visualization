import os
import sys
import numpy as np 
import cv2
import time
import random
import matplotlib.pyplot as plt
from utilty import *

v_path = 'video/564.AVI'
video_p = int(v_path.split('/')[1].split('.')[0])
print(video_p)
load = 1
mean_cal = 0
people = 17

if(load == 1):
    point_list = np.load(f'record_npy/record_{str(video_p).zfill(4)}_f60_p{people}.npy',allow_pickle=True)
else:
    point_list = genrecord_npy(video_p,people,rate=60,save=1)
    
po = np.moveaxis(point_list,0,1)
print(po.shape)
if(mean_cal == 1):
    mean_list , label_draw , count_c = genmean(po,th=10)
    np.save(f'record_npy/mean_list_{str(video_p).zfill(4)}_f60_p{people}.npy',np.array(mean_list))
    np.save(f'record_npy/mean_count_{str(video_p).zfill(4)}_f60_p{people}.npy',np.array(count_c))
else:
    mean_list = np.load(f'record_npy/mean_list_{str(video_p).zfill(4)}_f60_p{people}.npy',allow_pickle=True)
    count_c = np.load(f'record_npy/mean_count_{str(video_p).zfill(4)}_f60_p{people}.npy',allow_pickle=True)
    th = 10
    label_draw = np.where(count_c > len(mean_list)-th,True,False)
#'''

draw_attention_p(v_path,po,people,f'out_video/{str(video_p).zfill(4)}_{10}_90fps_bei_pick_mean_test',mean=True,mean_list = mean_list,label_draw=label_draw)
#'''