from __future__ import print_function
from __future__ import division
import numpy as np
from torchvision import transforms
import time
import os
import pickle
from torch.utils import data
import scipy.io as sio
import scipy.ndimage
import cv2
import random
from skimage import exposure
import panostretch
from pano import draw_boundary_from_cor_id
from scipy.ndimage import gaussian_filter

# Data augmentation and normalization for training
# Just normalization for validation
data_transforms = {
    'train': transforms.Compose([
        transforms.ToTensor(),
    ]),
    'val': transforms.Compose([
        transforms.ToTensor(),
    ]),
}

def cor2xybound(cor):
    ''' Helper function to clip max/min stretch factor '''
    corU = cor[0::2]
    corB = cor[1::2]
    zU = -50
    u = panostretch.coorx2u(corU[:, 0])
    vU = panostretch.coory2v(corU[:, 1])
    vB = panostretch.coory2v(corB[:, 1])

    x, y = panostretch.uv2xy(u, vU, z=zU)
    c = np.sqrt(x**2 + y**2)
    zB = c * np.tan(vB)
    xmin, xmax = x.min(), x.max()
    ymin, ymax = y.min(), y.max()

    S = 3 / abs(zB.mean() - zU)
    dx = [abs(xmin * S), abs(xmax * S)]
    dy = [abs(ymin * S), abs(ymax * S)]

    return min(dx), min(dy), max(dx), max(dy)

# Data generator
class ShapeNetDataset(data.Dataset):
    def __init__(self, root_dir, train_type, transform=None):
        print(root_dir)
        self.namelist = next(os.walk(root_dir))[2]
        self.root_dir = root_dir
        self.transform = transform
        self.train_type = train_type
        self.max_stretch = 2.0
        self.im_w = 1024
        self.im_h = 512

    def __len__(self):
        return len(self.namelist)

    def __getitem__(self, idx):
        pkl_path = self.root_dir+self.namelist[idx]
        pkl = pkl = pickle.load(open(pkl_path, 'rb'))
        img = pkl['image'].astype('float32')
        label = pkl['edge'].astype('float32')
        mask = pkl['line'].astype('float32')
        cor = pkl['cor'].astype('float32')
        
        # data augmentation
        if self.train_type == 'train':
            # random streching
            xmin, ymin, xmax, ymax = cor2xybound(cor)
            kx = np.random.uniform(1.0, self.max_stretch)
            ky = np.random.uniform(1.0, self.max_stretch)
            if np.random.randint(2) == 0:
                kx = max(1 / kx, min(0.5 / xmin, 1.0))
            else:
                kx = min(kx, max(10.0 / xmax, 1.0))
            if np.random.randint(2) == 0:
                ky = max(1 / ky, min(0.5 / ymin, 1.0))
            else:
                ky = min(ky, max(10.0 / ymax, 1.0))
            img, mask, cor = panostretch.pano_stretch(img, mask, cor, kx, ky)

            # random rotation
            random.seed()
            h = img.shape[0]
            w = img.shape[1]
            rot = int(np.floor(np.random.random()*w))
            img = np.concatenate((img[:,rot:,:],img[:,:rot,:]), axis=1)
            mask = np.concatenate((mask[:,rot:,:],mask[:,:rot,:]), axis=1)
            cor[:,0] = cor[:,0] - rot
            id = cor[:,0]<0
            cor[id,0] = cor[id,0]+1023

        # generate line label
        # sort gt
        cor_id = np.argsort(cor[:,0])
        cor = cor[cor_id,:]
        for row in range(0,cor.shape[0],2):
            cor_id = np.argsort(cor[row:row+2,1])
            cor[row:row+2,:] = cor[row:row+2,cor_id]
        # wall
        kpmap_w = np.zeros((self.im_h, self.im_w))
        bg = np.zeros_like(img)
        for l_id in range(0,cor.shape[0],2):
            panoEdgeC = draw_boundary_from_cor_id(cor[l_id:l_id+2,:],bg)
            # add gaussian
            panoEdgeC = panoEdgeC.astype('float32')/255.0
            panoEdgeC = gaussian_filter(panoEdgeC[:,:,1], sigma=20)
            panoEdgeC = panoEdgeC/np.max(panoEdgeC)
            kpmap_w = np.maximum(kpmap_w, panoEdgeC)
        # ceil
        kpmap_c = np.zeros((self.im_h, self.im_w))
        cor_all = cor[[0,2,2,4,4,6,6,0],:]
        for l_id in range(0,cor_all.shape[0],2):
            panoEdgeC = draw_boundary_from_cor_id(cor_all[l_id:l_id+2,:],bg)
            # add gaussian
            panoEdgeC = panoEdgeC[:,:,1].astype('float32')/255.0
            panoEdgeC[int(np.amax(cor_all[l_id:l_id+2,1]))+5:,:] = 0
            panoEdgeC = gaussian_filter(panoEdgeC, sigma=20)
            panoEdgeC = panoEdgeC/np.max(panoEdgeC)
            kpmap_c = np.maximum(kpmap_c, panoEdgeC)
        # floor
        kpmap_f = np.zeros((self.im_h, self.im_w))
        cor_all = cor[[1,3,3,5,5,7,7,1],:]
        for l_id in range(0,cor_all.shape[0],2):
            panoEdgeC = draw_boundary_from_cor_id(cor_all[l_id:l_id+2,:],bg)
            # add gaussian
            panoEdgeC = panoEdgeC[:,:,1].astype('float32')/255.0
            panoEdgeC[:int(np.amin(cor_all[l_id:l_id+2,1]))-5,:] = 0
            panoEdgeC = gaussian_filter(panoEdgeC, sigma=20)
            panoEdgeC = panoEdgeC/np.max(panoEdgeC)
            kpmap_f = np.maximum(kpmap_f, panoEdgeC)
        label = np.stack((kpmap_w, kpmap_c, kpmap_f), axis=-1)

        # generate corner label
        label2 = np.zeros((self.im_h, self.im_w))
        for l_id in range(cor.shape[0]):
            panoEdgeC = np.zeros((self.im_h, self.im_w))
            hh = int(np.round(cor[l_id,1]))
            ww = int(np.round(cor[l_id,0]))
            panoEdgeC[hh-1:hh+2, ww]=1.0
            panoEdgeC[hh, ww-1:ww+2]=1.0
            # add gaussian
            panoEdgeC = gaussian_filter(panoEdgeC, sigma=20)
            panoEdgeC = panoEdgeC/np.max(panoEdgeC)
            label2 = np.maximum(label2, panoEdgeC)
        label2 = np.expand_dims(label2, axis=2)

        if self.train_type == 'train':
            # gamma
            random.seed()
            g_prob = np.random.random()*1+0.5
            img = exposure.adjust_gamma(img, g_prob)
            # intensity
            random.seed()
            g_prob = np.random.random()*127
            img = exposure.rescale_intensity(img*255.0, in_range=(g_prob, 255))
            # color channel
            random.seed()
            g_prob = np.random.random()*0.4+0.8
            img[:,:,0] = img[:,:,0]*g_prob
            random.seed()
            g_prob = np.random.random()*0.4+0.8
            img[:,:,1] = img[:,:,1]*g_prob
            random.seed()
            g_prob = np.random.random()*0.4+0.8
            img[:,:,2] = img[:,:,2]*g_prob

            # random flip
            if random.uniform(0, 1) > 0.5:
                img = np.fliplr(img).copy()
                mask = np.fliplr(mask).copy()
                label = np.fliplr(label).copy()
                label2 = np.fliplr(label2).copy()

        # reshape
        np.clip(img, 0.0, 1.0 , out=img)
        np.clip(label, 0.0, 1.0 , out=label)
        np.clip(label2, 0.0, 1.0 , out=label2)
        np.clip(mask, 0.0, 1.0 , out=mask)
        
        img = np.concatenate((img, mask), axis=2)

        # permute dim
        if self.transform:
            if self.train_type == 'train':
                img = data_transforms['train'](img).float()
                label = data_transforms['train'](label).float()
                label2 = data_transforms['train'](label2).float()
            else:
                img = data_transforms['val'](img).float()
                label = data_transforms['val'](label).float()
                label2 = data_transforms['val'](label2).float()

        return img, label, label2
