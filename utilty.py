import sys
import numpy as np 
import cv2
import time
import random
import matplotlib.pyplot as plt
from sklearn.cluster import MeanShift,estimate_bandwidth
p_num = 10
rat = 90//p_num
print(rat)

def genrecord_npy(video_p=None,per_num=17,rate = 30,save = 0):
    if(video_p == None or video_p < 1 or video_p > 700):
        assert('Waring 1~700 video pick')
        return 0
    gazeFile = './exportdata_train'
    video_res_x = 640
    video_res_y = 360

    screen_res_x = 1440
    screen_res_y = 900
    

    a=video_res_x/screen_res_x
    b=(screen_res_y-video_res_y/a)/2

    frame_list = np.load('frame_list.npy')
    video_num = video_p
    video_list = []
    for person in range(1,per_num+1):
        person_list = []
        file_ = f'{gazeFile}/P{str(person).zfill(2)}/P{str(person).zfill(2)}_Trail{str(video_num).zfill(3)}.txt'
        fd = open(file_,'r')    
        time,model,trialnum,diax, diay, x_screen,y_screen,event = np.loadtxt(fd,
                delimiter='\t',
                dtype={'names': ('col1', 'col2', 'col3', 'col4', 'col5', 'col6', 'col7', 'col8'),
                'formats': ('float', 'S4', 'float', 'float', 'float', 'float','float','S1')},skiprows = 1,unpack=True)
        fd.close()
        #print(np.where(event == 'S'))
        print('frame_num : ',frame_list[video_num-1])
        for i in range(frame_list[video_num-1]*rate//30):
            time = time-time[0]
            eff = np.array(np.where(((i)<rate*time/1000000)&(rate*time/1000000<i+1)&(event == b'F')==1)) #np.bitwise_or
            #print(eff.shape,eff)
            #############print(i,eff,eff.size)

            if(eff.size == 0):
            #    print('none')
                person_list.append(np.array([[-1,-1]]))
                continue
            #print(eff)
            #print(type(x_screen))
            x_stimulus=np.array([a*x_screen[n] for n in eff[0]]).astype(int)
            y_stimulus=np.array([a*(y_screen[n]-b) for n in eff[0]]).astype(int)
            #print(type(x_screen),x_stimulus.dtype)
            t = np.where(np.logical_not(np.logical_or(np.logical_or(x_stimulus<=0,x_stimulus>=video_res_x) \
                ,np.logical_or( y_stimulus<=0 ,y_stimulus>=video_res_y))))[0]
            if(t.size == 0):
                person_list.append(np.array([[-1,-1]]))
                continue
            x = x_stimulus[t]
            y = y_stimulus[t]
            ###############print(x,y,event[eff])
            point = np.concatenate(([x],[y]))
            point = np.transpose(point)
            ###############print(point.shape)
            person_list.append(point)
            #all_fixation[i,y,x,...] = color_x[(person-1)%6,...]
        video_list.append(person_list)
    if(save == 1):
        np.save(f'record_npy/record_{str(video_p).zfill(4)}_f60_p{per_num}.npy',np.array(video_list))
    return np.array(video_list)
    #'''
#'''
def genmean(point_list,th=20):
    frame_num = point_list.shape[0]
    per_num = point_list.shape[1]
    print(frame_num,per_num)
    p_temp = np.zeros((per_num,2),dtype=np.int) 
    p_temp[:,0] = 320
    p_temp[:,1] = 180
    count_c = np.zeros((per_num),dtype=np.int) 
    mean_list = []
    for i in range(0,frame_num,6): # num of frame every 6 p take 1 p (60 -> 10p) 
        for j in range(per_num): #num of person
            point_p = point_list[i][j]
            #print(point_p)
            if(point_p[0][0] == -1):
                continue
            else :
                p_m = (np.mean(point_p,axis=0)).astype(int)
                p_temp[j,0] = p_m[0]
                p_temp[j,1] = p_m[1]
        #mean
        bandwidth = estimate_bandwidth(p_temp, quantile=0.7, n_samples=None)
        ms = MeanShift(bandwidth=bandwidth, bin_seeding=True)
        ms.fit(p_temp)
        labels = ms.labels_
        cluster_centers = ms.cluster_centers_
        labels_unique ,count = np.unique(labels,return_counts=True)
        max_c = np.argmax(count)
        max_c_center = cluster_centers[max_c]
        label_point = labels == max_c
        count_c[label_point] += 1
        mean_list.append([max_c_center.astype(int)])
    print(count_c)
    label_draw = np.where(count_c > len(mean_list)-th,True,False)
    return mean_list , label_draw , count_c

def tri_bezier(p1,p2,p3,p4,t):
    parm_1 = (1-t)**3
    parm_2 = 3*(1-t)**2 * t
    parm_3 = 3 * t**2 * (1-t)
    parm_4 = t**3

    px = p1[0] * parm_1 + p2[0] * parm_2 + p3[0] * parm_3 + p4[0] * parm_4
    py = p1[1] * parm_1 + p2[1] * parm_2 + p3[1] * parm_3 + p4[1] * parm_4
    return px,py
    
def xy_iter_(p_s,p_d):
    l = np.sqrt((p_d[0]-p_s[0])**2+(p_d[1]-p_s[1])**2)
    
    print(l)
    global rat
    rat_ = rat+1
    xy_p = np.zeros((rat_,2))
    if(l <= 1.0):
        xy_p[:,0] = p_d[0]
        xy_p[:,1] = p_d[1]
        return xy_p
    #r = np.random.choice([-1,1], 1, p=[0.5, 0.5])
    t = np.linspace(0,1,rat_,endpoint=True)
    th = np.arctan2(-(p_d[0]-p_s[0]),(p_d[1]-p_s[1]))
    p_c1 = (p_d*0.4 + p_s*0.6)
    p_c1[0] = p_c1[0]+np.log1p(l)*np.cos(th)
    p_c1[1] = p_c1[1]+np.log1p(l)*np.sin(th)
    p_c2 = (p_d*0.7 + p_s*0.3)
    p_c2[0] = p_c2[0]+np.log1p(l)*np.cos(th)
    p_c2[1] = p_c2[1]+np.log1p(l)*np.sin(th)
    
    #if(p_d[0]-p_s[0] >= 0 and p_d[1]-p_s[1] >= 0):

    px , py = tri_bezier(p_s,p_c1,p_c2,p_d,t)
    xy_p[:,0] = px
    xy_p[:,1] = py
    return xy_p
def per_xy_iter(f_point_list,meanp_list,per_num,p_temp,p_mean=None):
    xy_iter_list = []
    mean = np.zeros((2),dtype=np.int)
    #person
    for j in range(per_num):
        point_p = f_point_list[j]
        print(point_p)
        if(point_p[0][0] == -1):
            list_temp = xy_iter_(p_temp[j],p_temp[j])
            #print('blink')
            #continue
        else :
            p_m = (np.mean(point_p,axis=0)).astype(int)
            list_temp = xy_iter_(p_temp[j],p_m)
            p_temp[j,0] = p_m[0]
            p_temp[j,1] = p_m[1]
        print(list_temp)
        xy_iter_list.append(list_temp)
    #mean
    #'''
    mean = meanp_list[0]
    list_temp = xy_iter_(p_mean[0],mean)
    xy_iter_list.append(list_temp)
    p_mean[0,0] = mean[0]
    p_mean[0,1] = mean[1]
    #'''
    return xy_iter_list

def draw_iter(xy_iter_list,per_num,img,color_table,videoWrite,i,mean=False,label_point=None):
    point_num = 0
    if(mean == True):
        per_num_ = per_num + 1
    else:
        per_num_ = per_num
    for index in range(len(xy_iter_list)):
        if(xy_iter_list[index] != []):
            point_num += 1 
    point_num *= 10
    if(point_num == 10):
        length = 8
        n = 8
        m = 0
    elif(point_num == 20):
        length = 12  
        n = 8
        m = 1
    elif(point_num == 30):
        length = 24
        n = 1
        m = 2
    #"""
    for iter_ in range(3):
        img_ = img.copy()
        k = 0 
        for j in range(per_num_):
            if(j != per_num and label_point[j] == False and mean== True):
                continue
            if(j == per_num):
                col = color_table[-1]
            else:
                k += 1
                if(k >= 8):
                    continue
                col = color_table[k]
            start_p = int(max(0,(i*3+iter_)/n+m)) # 0-1 1-2 2-8-10
            end_p = int(min(start_p+length,point_num-1)) # 8-9 or 13-14 or 22-30
            
            for p in range(start_p,end_p):
                x_i1 = xy_iter_list[p//10][j][p%10,0]
                y_i1 = xy_iter_list[p//10][j][p%10,1]
                x_i2 = xy_iter_list[(p+1)//10][j][(p+1)%10,0]
                y_i2 = xy_iter_list[(p+1)//10][j][(p+1)%10,1]
                col_ = [max(int(col[n]-3.5*point_num+3.5*p),0) for n in range(3)]
                #col_ = col
                cv2.line(img_, (int(x_i1), int(y_i1)), \
                               (int(x_i2), int(y_i2)),col_, p//4+1)

            cv2.circle(img_,(int(x_i2),int(y_i2)),5,col,-1)
        cv2.imshow('name',img_)
        cv2.waitKey(1)
        videoWrite.write(img_)
    #"""
def draw_attention_p(video_path,point_list,per_num,output_='record_temp',mean=False,mean_list = None,label_draw=None):
    #img = np.zeros((video_res_y,video_res_x,3),dtype=np.uint8)
    cap = cv2.VideoCapture(video_path)
    rows = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cols = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    fps_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    p_temp = np.zeros((per_num,2),dtype=np.int)
    p_mean = np.zeros((1,2),dtype=np.int)
    p_temp[:,0] = cols//2
    p_temp[:,1] = rows//2
    p_mean[:,0] = cols//2
    p_mean[:,1] = rows//2
    color_table = [ [200,0,0],
                    [0,200,0],
                    [0,0,200],
                    [200,0,200],
                    [100,100,200],
                    [200,140,140],
                    [200,69,0],
                    [200,200,0],
                    [0,255,255]]
    print('frame : ',point_list.shape[0], fps_count)
    if(point_list.shape[0] != fps_count*3):
        print('wrong video')
    videoWrite = cv2.VideoWriter(output_ + '.avi',\
                                 cv2.VideoWriter_fourcc('X', 'V', 'I', 'D'), 90, (cols,rows))
    i = 0
    global rat
    xy_iter_list = [[]]*3
    print(xy_iter_list)
    while(cap.isOpened()):
            ret, frame = cap.read()
            if ret == True:
                if(i%3==0):
                    xy_list = per_xy_iter(point_list[2*i],mean_list[i//3],per_num,p_temp,p_mean)
                    if(xy_iter_list[0] == []):
                        xy_iter_list[0] = xy_list
                    elif(xy_iter_list[1] == []):
                        xy_iter_list[1] = xy_list
                    elif(xy_iter_list[2] == []):
                        xy_iter_list[2] = xy_list
                    else:
                        xy_iter_list[0] = xy_iter_list[1]
                        xy_iter_list[1] = xy_iter_list[2]
                        xy_iter_list[2] = xy_list
                print(len(xy_iter_list),xy_iter_list[0][0].shape)
                #point_access(point_list[i],per_num,frame,p_temp,color_table,videoWrite)
                draw_iter(xy_iter_list,per_num,frame,color_table,videoWrite,i%3,mean=mean,label_point=label_draw)
                #videoWrite.write(frame_temp)
                i += 1
                if(i >= point_list.shape[0]):
                    break
            else :
                break
    cap.release()
    videoWrite.release()
    return