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

import cv2

from model import *

from pano import get_ini_cor
from pano_opt_gen import optimize_cor_id
import post_proc2 as post_proc
from shapely.geometry import Polygon
from scipy.ndimage.filters import maximum_filter
from scipy.ndimage import convolve
import scipy.signal
import sys

from sklearn.metrics import classification_report

# general case

# Top level data directory. Here we assume the format of the directory conforms 
#   to the ImageFolder structure
test_path = './data/matterport/mp3d_align/'

weight_path = './model/resnet34_matterport.pth'

save_path = './result_gen/'
depth_path = './result_gen_depth/'
depth_path_gt = './data/matterport/share_depth/'

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

def find_N_peaks(signal, r=29, min_v=0.05, N=None):
    max_v = maximum_filter(signal, size=r, mode='wrap')
    pk_loc = np.where(max_v == signal)[0]
    pk_loc = pk_loc[signal[pk_loc] > min_v]
    # check for odd case, remove one 
    if (pk_loc.shape[0]%2)!=0:
        pk_id = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[pk_id[:-1]]
        pk_loc = np.sort(pk_loc)
    if N is not None:
        order = np.argsort(-signal[pk_loc])
        pk_loc = pk_loc[order[:N]]
        pk_loc = pk_loc[np.argsort(pk_loc)]
    return pk_loc, signal[pk_loc]

def find_N_peaks_conv(signal, prominence, distance, N=4):
    locs, _ = scipy.signal.find_peaks(signal,
                                      prominence=prominence,
                                      distance=distance)
    pks = signal[locs]
    pk_id = np.argsort(-pks)
    pk_loc = locs[pk_id[:min(N, len(pks))]]
    pk_loc = np.sort(pk_loc)
    return pk_loc, signal[pk_loc]

def get_ini_cor(cor_img, d1=21, d2=3):
    cor = convolve(cor_img, np.ones((d1, d1)), mode='constant', cval=0.0)
    cor_id = []
    cor_ = cor_img.sum(0)
    cor_ = (cor_-np.amin(cor_))/np.ptp(cor_)
    
    min_v = 0.25#0.05
    xs_ = find_N_peaks(cor_, r=26, min_v=min_v, N=None)[0]
    
    # spetial case for too less corner
    if xs_.shape[0] < 4:
        xs_ = find_N_peaks(cor_, r=26, min_v=0.05, N=None)[0]
        if xs_.shape[0] < 4:
            xs_ = find_N_peaks(cor_, r=26, min_v=0, N=None)[0]

    X_loc = xs_
    for x in X_loc:
        x_ = int(np.round(x))

        V_signal = cor[:, max(0, x_-d2):x_+d2+1].sum(1)
        y1, y2 = find_N_peaks_conv(V_signal, prominence=None,
                              distance=20, N=2)[0]
        cor_id.append((x, y1))
        cor_id.append((x, y2))

    cor_id = np.array(cor_id, np.float64)

    return cor_id

def test_general(dt_cor_id, gt_cor_id, w, h, losses):

    dt_floor_coor = dt_cor_id[1::2]
    dt_ceil_coor = dt_cor_id[0::2]
    gt_floor_coor = gt_cor_id[1::2]
    gt_ceil_coor = gt_cor_id[0::2]
    assert (dt_floor_coor[:, 0] != dt_ceil_coor[:, 0]).sum() == 0
    assert (gt_floor_coor[:, 0] != gt_ceil_coor[:, 0]).sum() == 0

    # Eval 3d IoU and height error(in meter)
    N = len(dt_floor_coor)
    ch = -1.6
    dt_floor_xy = post_proc.np_coor2xy(dt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    gt_floor_xy = post_proc.np_coor2xy(gt_floor_coor, ch, 1024, 512, floorW=1, floorH=1)
    dt_poly = Polygon(dt_floor_xy)
    gt_poly = Polygon(gt_floor_xy)

    area_dt = dt_poly.area
    area_gt = gt_poly.area
    
    if area_dt < 1e-05:
        print('too small room')
        # Add a result
        n_corners = len(gt_floor_coor)
        n_corners = str(n_corners) if n_corners < 14 else '14+'
        losses[n_corners]['2DIoU'].append(0)
        losses[n_corners]['3DIoU'].append(0)
        losses['overall']['2DIoU'].append(0)
        losses['overall']['3DIoU'].append(0)
        return
        
    area_inter = dt_poly.intersection(gt_poly).area
    
    area_union = dt_poly.union(gt_poly).area
    area_pred_wo_gt = dt_poly.difference(gt_poly).area
    area_gt_wo_pred = gt_poly.difference(dt_poly).area

    iou2d = area_inter / (area_gt + area_dt - area_inter)
    cch_dt = post_proc.get_z1(dt_floor_coor[:, 1], dt_ceil_coor[:, 1], ch, 512)
    cch_gt = post_proc.get_z1(gt_floor_coor[:, 1], gt_ceil_coor[:, 1], ch, 512)

    h_dt = abs(cch_dt.mean() - ch)
    h_gt = abs(cch_gt.mean() - ch)
    #iouH = min(h_dt, h_gt) / max(h_dt, h_gt)
    #iou3d = iou2d * iouH
    iou3d = (area_inter * min(h_dt, h_gt)) / (area_pred_wo_gt * h_dt + area_gt_wo_pred * h_gt + area_inter * max(h_dt, h_gt))

    # Add a result
    n_corners = len(gt_floor_coor)
    n_corners = str(n_corners) if n_corners < 14 else '14+'
    losses[n_corners]['2DIoU'].append(iou2d)
    losses[n_corners]['3DIoU'].append(iou3d)
    losses['overall']['2DIoU'].append(iou2d)
    losses['overall']['3DIoU'].append(iou3d)

# Load data
gt_txt_path = '/data/czou4/Layout/_final_label_v2/test.txt'
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

losses = dict([
        (n_corner, {'2DIoU': [], '3DIoU': [], 'rmse':[], 'delta_1':[]})
        for n_corner in ['4', '6', '8', '10', '12', '14+', 'overall']
    ])
# for precision recall
target_names = ['4 corners', '6 corners', '8 corners', '10 corners', '12 corners', '14 corners', '16 corners', '18 corners']
y_true = np.zeros(len(namelist))
y_pred = np.zeros(len(namelist))

for file_list in namelist:
    
    #file_list = np.random.choice(namelist, 1)  
    #file_list = file_list[0]
    
    print(file_list)
    
    file_list_sub = file_list.split(" ")
    pkl_path = os.path.join(test_path,file_list_sub[0],file_list_sub[1])
    img = cv2.imread(os.path.join(pkl_path,'aligned_rgb.png'))
    img = img.astype('float32')/255.0
    mask = cv2.imread(os.path.join(pkl_path,'aligned_line.png'))
    mask = mask.astype('float32')/255.0
    gt = np.loadtxt(os.path.join(pkl_path,'cor.txt'))

    # lr flip
    img2 = np.fliplr(img).copy()
    mask2 = np.fliplr(mask).copy()

    image = torch.tensor(img).to(device).float()
    masks = torch.tensor(mask).to(device).float()     
    
    inputs = image.permute(2,0,1)
    inputs = inputs.unsqueeze(0)
    masks = masks.permute(2,0,1)
    masks = masks.unsqueeze(0)
    inputs = torch.cat((inputs,masks),1)

    image2 = torch.tensor(img2).to(device).float()
    masks2 = torch.tensor(mask2).to(device).float()

    inputs2 = image2.permute(2,0,1)
    inputs2 = inputs2.unsqueeze(0)
    masks2 = masks2.permute(2,0,1)
    masks2 = masks2.unsqueeze(0)
    inputs2 = torch.cat((inputs2,masks2),1)

    inputs = torch.cat((inputs, inputs2),0)

    # forward
    outputs, outputs2 = model_ft(inputs)

    # lr flip and take mean
    outputs1 = outputs[1]
    outputs22 = outputs2[1]

    inv_idx = torch.arange(outputs1.size(2)-1, -1, -1).to(device).long()
    outputs1 = outputs1.index_select(2, inv_idx)
    outputs = torch.mean(torch.cat((outputs[0].unsqueeze(0), outputs1.unsqueeze(0)), 0), 0, True)

    outputs22 = outputs22.index_select(2, inv_idx)
    outputs2 = torch.mean(torch.cat((outputs2[0].unsqueeze(0), outputs22.unsqueeze(0)), 0), 0, True)

    outputs = outputs.squeeze(0).permute(1,2,0)
    outputs2 = outputs2.squeeze(0).squeeze(0)

    inputs = inputs[0].permute(1,2,0)

    #gradient ascent refinement
    cor_img = outputs2.data.cpu().numpy()
    edg_img = outputs.data.cpu().numpy()
    
    #general layout, tp view
    cor_ = cor_img.sum(0)
    cor_ = (cor_-np.amin(cor_))/np.ptp(cor_)
    min_v = 0.25#0.05
    xs_ = find_N_peaks(cor_, r=26, min_v=min_v, N=None)[0]
    # spetial case for too less corner
    if xs_.shape[0] < 4:
        xs_ = find_N_peaks(cor_, r=26, min_v=0.05, N=None)[0]
        if xs_.shape[0] < 4:
            xs_ = find_N_peaks(cor_, r=26, min_v=0, N=None)[0]
    # get ceil and floor line
    ceil_img = edg_img[:,:,1]
    floor_img = edg_img[:,:,2]
    ceil_idx = np.argmax(ceil_img, axis=0)
    floor_idx = np.argmax(floor_img, axis=0)
    # Init floor/ceil plane
    z0 = 50
    force_cuboid=False
    _, z1 = post_proc.np_refine_by_fix_z(ceil_idx, floor_idx, z0)
    # Generate general  wall-wall
    cor, xy_cor = post_proc.gen_ww(xs_, ceil_idx, z0, tol=abs(0.16 * z1 / 1.6), force_cuboid=force_cuboid)
    
    if not force_cuboid:
        # Check valid (for fear self-intersection)
        xy2d = np.zeros((len(xy_cor), 2), np.float32)
        for i in range(len(xy_cor)):
            xy2d[i, xy_cor[i]['type']] = xy_cor[i]['val']
            xy2d[i, xy_cor[i-1]['type']] = xy_cor[i-1]['val']
        if not Polygon(xy2d).is_valid:
            # actually it's not force cuboid, just assume all corners are visible, go back to original LayoutNet initialization
            #print(
            #    'Fail to generate valid general layout!! '
            #    'Generate cuboid as fallback.',
            #    file=sys.stderr)
            cor_id = get_ini_cor(cor_img, 21, 3)
            force_cuboid= True
    
    if not force_cuboid:
        # Expand with btn coory
        cor = np.hstack([cor, post_proc.infer_coory(cor[:, 1], z1 - z0, z0)[:, None]])
        # Collect corner position in equirectangular
        cor_id = np.zeros((len(cor)*2, 2), np.float32)
        for j in range(len(cor)):
            cor_id[j*2] = cor[j, 0], cor[j, 1]
            cor_id[j*2 + 1] = cor[j, 0], cor[j, 2]

    # refinement
    cor_id = optimize_cor_id(cor_id, edg_img, cor_img, num_iters=100, verbose=False)

    test_general(cor_id, gt, 1024, 512, losses)
    
    # save, uncomment to generate depth map
    #print(save_path+file_list_sub[0]+'_'+file_list_sub[1]+'.mat')
    #sio.savemat(save_path+file_list_sub[0]+'_'+file_list_sub[1]+'.mat',{'cor_id':cor_id})
    
    #load
    pred_depth = depth_path+file_list_sub[0]+'_'+file_list_sub[1]+'.mat'

    if os.path.exists(pred_depth):
        pred_depth = sio.loadmat(pred_depth)
        pred_depth = pred_depth['im_depth']
        
        #gt
        gt_depth = np.load(os.path.join(depth_path_gt, file_list_sub[0], file_list_sub[1], 'new_depth.npy'))
        pred_depth = cv2.resize(pred_depth, (gt_depth.shape[1], gt_depth.shape[0]))
       
        # rmse
        pred_depth = pred_depth[np.nonzero(gt_depth)]
        gt_depth = gt_depth[np.nonzero(gt_depth)]
        rmse = np.average((gt_depth - pred_depth) ** 2) ** 0.5
     
        # delta_1
        max_map = np.where(gt_depth/pred_depth > pred_depth/gt_depth, gt_depth/pred_depth, pred_depth/gt_depth)
        delta_1 = np.average(np.where(max_map < 1.25, 1, 0))
        
        # Add a result
        n_corners = len(gt[1::2])
        n_corners = str(n_corners) if n_corners < 14 else '14+'
        losses[n_corners]['rmse'].append(rmse)
        losses[n_corners]['delta_1'].append(delta_1)
        losses['overall']['rmse'].append(rmse)
        losses['overall']['delta_1'].append(delta_1)

    torch.cuda.empty_cache()
    #del outputs1, outputs, outputs2, outputs22, labels, labels2, inputs, inputs2, loss
    del outputs1, outputs, outputs2, outputs22, inputs, inputs2
    y_true[cnt] = int(gt.shape[0]//2//2-2)
    y_pred[cnt] = int(cor_id.shape[0]//2//2-2)
    
    cnt += 1
    num += 1

    iou2d = np.array(losses['overall']['2DIoU'])
    iou3d = np.array(losses['overall']['3DIoU'])
    rmse = np.array(losses['overall']['rmse'])
    delta_1 = np.array(losses['overall']['delta_1'])
    print('No. {}, 2d Loss: {:.6f}, 3d Loss: {:.6f}, rmse: {:.6f}, delta_1: {:.6f}'.format(cnt,iou2d.mean() * 100,iou3d.mean() * 100, rmse.mean() *100, delta_1.mean()*100))


for k, result in losses.items():
    iou2d = np.array(result['2DIoU'])
    iou3d = np.array(result['3DIoU'])
    rmse = np.array(result['rmse'])
    delta_1 = np.array(result['delta_1'])
    if len(iou2d) == 0:
        continue
    print('GT #Corners: %s  (%d instances)' % (k, len(iou2d)))
    print('    2DIoU: %.2f' % ( iou2d.mean() * 100))
    print('    3DIoU: %.2f' % ( iou3d.mean() * 100))
    print('    RMSE: %.2f' % ( rmse.mean()*100))
    print('    Delta_1: %.2f' % ( delta_1.mean()*100))

print(classification_report(y_true, y_pred, target_names=target_names))
