from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
print("PyTorch Version: ",torch.__version__)

import pickle
import os
import scipy.io as sio
from model import *

from pano import get_ini_cor
from pano_opt import optimize_cor_id
from utils_eval import eval_PE, eval_3diou

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
test_path = './data/test_PC/'

weight_path = './model/resnet34_PC.pth'

save_path = './result/'

# Pre-trained models to choose from [resnet18, resnet34, resnet50]
#model_name = "resnet18"
model_name = "resnet34"
#model_name = "resnet50"
num_classes = 1024

print("Load Models...")
# Define the encoder
encoder = initialize_encoder(model_name, num_classes,use_pretrained=True)
# Full model
model_ft = SegNet(encoder, num_classes)
model_ft.load_state_dict(torch.load(weight_path))

# Detect if we have a GPU available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# Send the model to GPU
model_ft = model_ft.to(device)

# evaluation mode
model_ft.eval()

# Load data
gt_txt_path = './gt/panoContext_test.txt'

namelist = []
with open(gt_txt_path, 'r') as f:
    while(True):
        line = f.readline().strip()
        if not line:
            break
        namelist.append(line)

criterion = nn.BCELoss()
criterion2 = nn.BCELoss()

cnt = 0
num = 0
loss_cor = 0.0
loss_pe = 0.0
loss_3d = 0.0
loss_sum = 0.0

for file_list in namelist:
    
    #file_list = np.random.choice(namelist, 1)  
    #file_list = file_list[0]

    #print(file_list)
    
    pkl_path = test_path+'PCts_'+ "{:04d}".format(cnt) + '.pkl'
    
    pkl = pickle.load(open(pkl_path, 'rb'))
    img = pkl['image'].astype('float32')
    label = pkl['edge'].astype('float32')
    label2 = pkl['junc'].astype('float32')
    mask = pkl['line'].astype('float32')

    # lr flip
    img2 = np.fliplr(img).copy()
    mask2 = np.fliplr(mask).copy()

    image = torch.tensor(img).to(device).float()
    labels = torch.tensor(label).to(device).float()
    labels2 = torch.tensor(label2).to(device).float()
    masks = torch.tensor(mask).to(device).float()     
    
    inputs = image.permute(2,0,1)
    inputs = inputs.unsqueeze(0)
    masks = masks.permute(2,0,1)
    masks = masks.unsqueeze(0)
    inputs = torch.cat((inputs,masks),1)
    labels = labels.permute(2,0,1)
    labels = labels.unsqueeze(0)
    labels2 = labels2.permute(2,0,1)
    labels2 = labels2.unsqueeze(0)

    image2 = torch.tensor(img2).to(device).float()
    masks2 = torch.tensor(mask2).to(device).float()

    inputs2 = image2.permute(2,0,1)
    inputs2 = inputs2.unsqueeze(0)
    masks2 = masks2.permute(2,0,1)
    masks2 = masks2.unsqueeze(0)
    inputs2 = torch.cat((inputs2,masks2),1)

    inputs = torch.cat((inputs, inputs2),0)

    outputs, outputs2 = model_ft(inputs)

    # lr flip and take mean
    outputs1 = outputs[1]
    outputs22 = outputs2[1]

    inv_idx = torch.arange(outputs1.size(2)-1, -1, -1).to(device).long()
    outputs1 = outputs1.index_select(2, inv_idx)
    outputs = torch.mean(torch.cat((outputs[0].unsqueeze(0), outputs1.unsqueeze(0)), 0), 0, True)

    outputs22 = outputs22.index_select(2, inv_idx)
    outputs2 = torch.mean(torch.cat((outputs2[0].unsqueeze(0), outputs22.unsqueeze(0)), 0), 0, True)

    loss = criterion(outputs, labels) + criterion(outputs2, labels2)
    
    loss_sum += loss.data.cpu().numpy()

    labels = labels.squeeze(0).permute(1,2,0)
    outputs = outputs.squeeze(0).permute(1,2,0)
    labels2 = labels2.squeeze(0).squeeze(0)
    outputs2 = outputs2.squeeze(0).squeeze(0)

    inputs = inputs[0].permute(1,2,0)

    #gradient ascent refinement
    cor_img = outputs2.data.cpu().numpy()
    edg_img = outputs.data.cpu().numpy()
    
    # load gt
    path = os.path.join('./data/layoutnet_dataset/test', 'label_cor', '%s.txt' % file_list[:-4])
    with open(path) as f:
        gt = np.array([line.strip().split() for line in f], np.float64)

    # sort gt, since it's box shape
    gt_id = np.argsort(gt[:,0])
    gt = gt[gt_id,:]
    for row in range(0,gt.shape[0],2):
        gt_id = np.argsort(gt[row:row+2,1])
        gt[row:row+2,:] = gt[row:row+2,gt_id]
    
    # initial corners
    cor_id = get_ini_cor(cor_img, 21, 3)

    # refinement
    cor_id = optimize_cor_id(cor_id, edg_img, cor_img, num_iters=100, verbose=False)

    # sort cor_id, since it's box
    cor_idd = np.argsort(cor_id[:,0])
    cor_id = cor_id[cor_idd,:]
    for row in range(0,cor_id.shape[0],2):
        cor_idd = np.argsort(cor_id[row:row+2,1])
        cor_id[row:row+2,:] = cor_id[row:row+2,cor_idd]

    cor_error = ((gt - cor_id) ** 2).sum(1) ** 0.5
    cor_error /= np.sqrt(cor_img.shape[0] ** 2 + cor_img.shape[1] ** 2)
    cor_error = cor_error.mean()
   
    # rotate variations 
    cor_id2 = np.concatenate((cor_id[2:,:], cor_id[:2,:]), axis=0)
    cor_id2[6:,0] = 1024+cor_id2[6:,0]
    cor_error2 = ((gt - cor_id2) ** 2).sum(1) ** 0.5
    cor_error2 /= np.sqrt(cor_img.shape[0] ** 2 + cor_img.shape[1] ** 2)
    cor_error2 = cor_error2.mean()
    
    cor_id3 = np.concatenate((cor_id[6:,:], cor_id[:6,:]), axis=0)
    cor_id3[:2,0] = cor_id3[:2,0]-1024
    cor_error3 = ((gt - cor_id3) ** 2).sum(1) ** 0.5
    cor_error3 /= np.sqrt(cor_img.shape[0] ** 2 + cor_img.shape[1] ** 2)
    cor_error3 = cor_error3.mean()
    if cor_error2 <= cor_error and cor_error <= cor_error3:
        cor_error = cor_error2
    if cor_error3 <= cor_error and cor_error <= cor_error2:
        cor_error = cor_error3
    
    # pixel error
    pe_error, surface, surface_gt = eval_PE(cor_id[0::2], cor_id[1::2], gt[0::2], gt[1::2])
    # 3D IoU
    iou3d = eval_3diou(cor_id[1::2], cor_id[0::2], gt[1::2], gt[0::2])
    
    loss_cor += cor_error
    loss_pe += pe_error
    loss_3d += iou3d
    
    
    # save
    print(save_path+file_list[:-3]+'mat')
    #sio.savemat(save_path+file_list[:-3]+'mat',{'image':inputs.data.cpu().numpy(), 'pred2':cor_img, 'pred':edg_img, 'cor_id':cor_id})
#    sio.savemat(save_path+'PCts_'+ "{:04d}".format(cnt)+'.mat',{'image':inputs.data.cpu().numpy(), 'pred2':cor_img, 'pred':edg_img, 'cor_id':cor_id})
    
    torch.cuda.empty_cache()
    del outputs1, outputs, outputs2, outputs22, labels, labels2, inputs, inputs2, loss

    cnt += 1
    num += 1
    
    print('No. {}, cor Loss: {:.6f}, pc Loss: {:.6f}, 3d Loss: {:.6f}'.format(cnt,loss_cor/cnt,loss_pe/cnt,loss_3d/cnt))

print('Total No. {}, cor Loss: {:.6f}, pc Loss: {:.6f}, 3d Loss: {:.6f}'.format(cnt,loss_cor/cnt, loss_pe/cnt, loss_3d/cnt))
