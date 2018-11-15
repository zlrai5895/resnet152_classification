#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 15 16:59:29 2018

@author: ws
"""

import argparse
import torch
import torch.nn as nn
import numpy as np
from data_loader import get_test_loader
from torchvision import transforms
from torchvision.models import resnet152
import os
from tqdm import tqdm




############################相关参数#########################################
parser=argparse.ArgumentParser()
parser.add_argument('--test_img_dir',type=str,default='/home/ws/3z_gan/dataset/val',help='test img path')
parser.add_argument('--checkpoint_dir',type=str,default='./checkpoint',help='the path to save model')
parser.add_argument('--pre_dir',type=str,default='./pred',help='the path to save the predicted results')
parser.add_argument('--batchsize',type=int,default=8,help='batchsize')
parser.add_argument('--num_workers',type=int,default=12,help='num_workers')
parser.add_argument('--num_classes',type=int,default=6,help='num_classes')
parser.add_argument('--num_epoches',type=int,default=30,help='num_epochs')
parser.add_argument('--crop_size',type=int,default=224,help='crop_size')
parser.add_argument('--pretrained_moedl_path',type=str,default='best_para.pkl',help='the path where  stored the pretrained model ')
args=parser.parse_args()




######################首先建立数据加载器#######################################
transform = transforms.Compose([ 
    transforms.RandomResizedCrop(args.crop_size ),
    transforms.RandomVerticalFlip(),
    transforms.ColorJitter(),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(),                         #将image转化为tensor
    transforms.Normalize(( 0.485, 0.456, 0.406 ), 
                         ( 0.229, 0.224, 0.225 ))])#三个通道的均值和方差

test_loader   = get_test_loader(args.test_img_dir,  transform,args.batchsize, shuffle=False, num_workers=args.num_workers)

#############################建立模型##########################################
net = resnet152(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, args.num_classes)

######################GPU加速##################################################
net=net.cuda()
net=torch.nn.DataParallel(net)


#############################加载模型##########################################
state=torch.load(os.path.join(args.checkpoint_dir,args.pretrained_moedl_path))
net.load_state_dict(state['state_dict'])


def test(args):
    net.eval()
    final_preds=[]
    for test_batch_idx, test_images in enumerate(tqdm(test_loader)):
        test_images= test_images.cuda()
        test_outputs=net(test_images)
        _, test_preds = torch.max(test_outputs.data, 1)
        final_preds.extend(test_preds)
    final_preds=np.array(final_preds)
    np.save(os.path.join(args.pre_dir,'pred.npy'),final_preds)
    print('Predict done!')