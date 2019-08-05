# load .t7 file and save as .pkl data

import torchfile
import cv2
import numpy as np
import scipy.io as sio
import pickle
import time

data_path = './data/test_PC/'

# panoContext
#img_tr = torchfile.load('./data/panoContext_img_train.t7')
#print(img_tr.shape)
#lne_tr = torchfile.load('./data/panoContext_line_train.t7')
#print(lne_tr.shape)
#edg_tr = torchfile.load('./data/panoContext_edge_train.t7')
#print(edg_tr.shape)
#junc_tr = torchfile.load('./data/panoContext_cor_train.t7')
#print(junc_tr.shape)
#print('done')

#img_tr = torchfile.load('./data/panoContext_img_val.t7')
#print(img_tr.shape)
#lne_tr = torchfile.load('./data/panoContext_line_val.t7')
#print(lne_tr.shape)
#edg_tr = torchfile.load('./data/panoContext_edge_val.t7')
#print(edg_tr.shape)
#junc_tr = torchfile.load('./data/panoContext_cor_val.t7')
#print(junc_tr.shape)
#print('done')

img_tr = torchfile.load('./data/panoContext_img_test.t7')
print(img_tr.shape)
lne_tr = torchfile.load('./data/panoContext_line_test.t7')
print(lne_tr.shape)
edg_tr = torchfile.load('./data/panoContext_edge_test.t7')
print(edg_tr.shape)
junc_tr = torchfile.load('./data/panoContext_cor_test.t7')
print(junc_tr.shape)
print('done')

# stanford
#img_tr = torchfile.load('./data/stanford2d-3d_img_area_5.t7')
#print(img_tr.shape)
#lne_tr = torchfile.load('./data/stanford2d-3d_line_area_5.t7')
#print(lne_tr.shape)
#edg_tr = torchfile.load('./data/stanford2d-3d_edge_area_5.t7')
#print(edg_tr.shape)
#junc_tr = torchfile.load('./data/stanford2d-3d_cor_area_5.t7')
#print(junc_tr.shape)
#print('done')

gt_txt_path = './data/panoContext_testmap.txt'
gt_path = './data/layoutnet_dataset/test/label_cor/'

# Load data
namelist = []
id_num = []
with open(gt_txt_path, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        id_num0 = line.split()
        id_num0 = int(id_num0[1])
        id_num.append(id_num0)
        namelist.append(line)
id_num = np.array(id_num)  

cnt = 0

for num in range(img_tr.shape[0]):
    print(num)
    
    image = img_tr[num]
    image = np.transpose(image, (1,2,0))#*255.0
    line = lne_tr[num]
    line = np.transpose(line, (1,2,0))
    edge = edg_tr[num]
    edge = np.transpose(edge, (1,2,0))
    junc = junc_tr[num]
    junc = np.transpose(junc, (1,2,0))
    # corner gt
    idn = np.where(id_num == num)
    idn = idn[0][0]
    filename = namelist[idn]
    filename = filename.split()
    filename = gt_path+filename[0][:-4]+'.txt'#'.mat'
  
    cnt+=1
    cor = np.loadtxt(filename)
    cor_sum = 0
    for cor_num in range(cor.shape[0]):
        cor_sum+=junc[int(cor[cor_num,1]),int(cor[cor_num,0]),0]
    #print(cor_sum)
    #time.sleep(0.5)

#   pickle.dump({'image':image, 'line':line, 'edge':edge, 'junc':junc, 'cor':cor, 'filename':filename[:-4]}, open(data_path+'PC_'+"{:04d}".format(num)+'.pkl', "wb" ) )    
    pickle.dump({'image':image, 'line':line, 'edge':edge, 'junc':junc, 'cor':cor, 'filename':filename[:-4]}, open(data_path+'PCts_'+"{:04d}".format(num)+'.pkl', "wb" ) )
#   pickle.dump({'image':image, 'line':line, 'edge':edge, 'junc':junc, 'cor':cor, 'filename':filename[:-4]}, open(data_path+'PCval_'+"{:04d}".format(num)+'.pkl', "wb" ) ) 
#    pickle.dump({'image':image, 'line':line, 'edge':edge, 'junc':junc, 'cor':cor, 'filename':filename[:-4]}, open(data_path+'area5_'+"{:04d}".format(num)+'.pkl', "wb" ) )

