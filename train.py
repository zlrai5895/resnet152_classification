# -*- coding: utf-8 -*-
"""
Created on Wed Oct 17 22:21:44 2018

@author: ws
"""
#coding:utf-8

######################此代码用于完成图像分类任务##############################

import argparse
import torch
import torch.nn as nn
from data_loader import get_train_loader, get_val_loader
from torch.autograd import Variable 
from torchvision import transforms
from tensorboardX import SummaryWriter
from torchvision.models import resnet152
from torch.optim import lr_scheduler
import os
from tqdm import tqdm

############################相关参数#########################################
parser=argparse.ArgumentParser()
parser.add_argument( '--train_csv_file', type=str, default='/home/ws/3z_gan/data/train_2000.csv',help='train csv dir')
parser.add_argument( '--val_csv_file', type=str, default='/home/ws/3z_gan/data/val_csv_file.csv',help='val csv dir')
parser.add_argument( '--train_img_dir', type=str, default='/home/ws/3z_gan/data/merge_data',help='train_img_dir')
parser.add_argument('--val_img_dir',type=str,default='/home/ws/3z_gan/data/merge_data',help='val img path')
parser.add_argument('--test_img_dir',type=str,default='/home/ws/3z_gan/dataset/val',help='test img path')
parser.add_argument('--checkpoint_dir',type=str,default='./checkpoint',help='the path to save model')
parser.add_argument('--log_dir',type=str,default='./log',help='the path to save model')
parser.add_argument('--pre_dir',type=str,default='./pred/',help='the path to save the predicted results')
parser.add_argument('--batchsize',type=int,default=8,help='batchsize')
parser.add_argument('--num_workers',type=int,default=12,help='num_workers')
parser.add_argument('--num_classes',type=int,default=6,help='num_classes')
parser.add_argument('--learning_rate',type=float,default=0.0001,help='learning rate at start')
parser.add_argument('--num_epoches',type=int,default=30,help='num_epochs')
parser.add_argument('--crop_size',type=int,default=224,help='crop_size')
parser.add_argument('--eval_every_steps',type=int,default=100,help='how many steps to do an test on val data')
parser.add_argument('--save_every_steps',type=int,default=100,help='how many steps to save')
parser.add_argument('--pretrained',type=bool,default=True,help='use pretrained model or not')
parser.add_argument('--pretrained_moedl_path',type=str,default='epoch_1_batch_idx249.pkl',help='the path where  stored the pretrained model ')
parser.add_argument('--start_decay_epoch',type=int,default=10,help='the epoch when to start decay learning rate')
parser.add_argument('--att_factor',type=float,default=0.9,help='the decay factor')
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

train_loader = get_train_loader(args.train_csv_file,args.train_img_dir,transform, args.batchsize, shuffle=True,  num_workers=args.num_workers)
val_loader   = get_val_loader(args.val_csv_file,args.train_img_dir,  transform,args.batchsize, shuffle=False, num_workers=args.num_workers)



#########################建立模型##############################################
net = resnet152(pretrained=True)
net.fc = nn.Linear(net.fc.in_features, args.num_classes)


######################GPU加速##################################################
if torch.cuda.is_available():
    net=net.cuda()
    net=torch.nn.DataParallel(net)


######################优化器和损失函数#########################################
optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate,weight_decay=0)
epoch_lr_scheduler = lr_scheduler.StepLR(optimizer,step_size=2,gamma=0.5)#权重衰减
criterion = nn.CrossEntropyLoss()



#####################是否要加载之前的模型######################################
if args.pretrained:
    print('==> Resuming from checkpoint..')
    assert os.path.isdir(args.checkpoint_dir), 'Error: no checkpoint directory found!'
    loaded_state=torch.load(os.path.join(args.checkpoint_dir,args.pretrained_moedl_path))
    net.load_state_dict(loaded_state['state_dict'])
    best_val_acc=loaded_state['best_val_accuracy']
    epoch_start=loaded_state['epoch']
    optimizer.load_state_dict(loaded_state['optimizer'])
else:
    print('==> Using model unpretrained..')
    best_val_acc=0
    epoch_start=0
#    net = CNN(args.num_classes)
    
    
#######################记录日志################################################
writer = SummaryWriter(log_dir=args.log_dir)





##############################开始训练#########################################
def train(args):
    net.train()     
    all_batch_ind=0
    try:
        for epoch in range(epoch_start,args.num_epoches):
            epoch_lr_scheduler.step()
            train_pred=[]
            train_targets=[]
            train_correct = 0
            train_total = 0
            total_train_loss=0
            batch_idx=0
            learning_rate=epoch_lr_scheduler.get_lr()[0]
            print('epoch:{},learning rate:{}'.format(epoch,learning_rate))
            val_accracy_per_epoch=[]
            for batch_idx, (images, targets,_) in enumerate(tqdm(train_loader)):
                all_batch_ind=all_batch_ind+1        #总的bacth_ind +1
                # 将数据移到GPU上
                images, targets = images.cuda(), targets.cuda()
                # 先将optimizer梯度先置为0
                optimizer.zero_grad()
                # 图计算的开始处。图的leaf variable
                images, targets = Variable(images), Variable(targets)
                # 模型输出
                outputs = net(images)
                train_loss = criterion(outputs, targets)#计算交叉熵损失
                total_train_loss=total_train_loss+train_loss
                
                
                _, temp_train_preds = torch.max(outputs.data, 1)
                train_total += targets.size(0)
                train_correct += temp_train_preds.eq(targets.data).cpu().numpy().sum()
                train_pred.extend(temp_train_preds)
                train_targets.extend(targets)#保存每个批次预测的结果和真实值
                
                # 反向传播，计算梯度
                train_loss.backward()
                # 更新参数
                optimizer.step()
                
                writer.add_scalar('data/train_loss', train_loss.data[0].item(), all_batch_ind)
                writer.add_scalar('data/learning_rate', learning_rate, all_batch_ind)
                
                if all_batch_ind%args.eval_every_steps==0:
                    net.eval()
                    val_correct = 0
                    val_total = 0
                    for val_batch_idx, (val_images, val_targets,_) in enumerate(val_loader):
                        val_images ,val_targets= val_images.cuda(), val_targets.cuda()
                        val_outputs=net(val_images)
                        _, val_preds = torch.max(val_outputs.data, 1)
                        val_total += val_targets.size(0)
                        val_correct += val_preds.eq(val_targets.data).cpu().numpy().sum()
                    val_accuracy=val_correct/val_total
                    val_accracy_per_epoch.append(round(val_accuracy,4))
                    writer.add_scalar('data/val_accuracy', val_accuracy, all_batch_ind)
                    net.train()
                    
                    if val_accuracy>best_val_acc:
                        best_epoch=epoch
                        state={
                                'epoch': epoch,
                                'state_dict': net.state_dict(),
                                'best_val_accuracy': val_accuracy,
                                'optimizer' : optimizer.state_dict(),
                                }
                        torch.save(state,os.path.join(args.checkpoint_dir,'best_para.pkl'))
    
        
                if all_batch_ind%args.save_every_steps==0:
                        state={
                                'epoch': epoch,
                                'state_dict': net.state_dict(),
                                'best_val_accuracy': val_accuracy,
                                'optimizer' : optimizer.state_dict(),
                                }
                        torch.save(state,os.path.join(args.checkpoint_dir,'epoch_{}_batch_idx{}.pkl'.format(epoch,batch_idx)))
    
                    
                    
             
            train_accracy=train_correct/train_total
            writer.add_scalar('data/train_accuracy', train_accracy, all_batch_ind)
            print('The val accracy during this epoch is:{}'.format(val_accracy_per_epoch))
            print('\n\n\n')
        print('\n\n\n')
        print('best  epoch is:{},the val accracy is:{}'.format(best_epoch, best_val_acc))
        writer.close()
    except  KeyboardInterrupt:
        print('The program is terminated, saving the current weight')
        state={
                'epoch': epoch,
                'state_dict': net.state_dict(),
                'best_val_accuracy': val_accuracy,
                'optimizer' : optimizer.state_dict(),
                }
        torch.save(state,os.path.join(args.checkpoint_dir,'before_KeyboardInterrupt_epoch_{}_batch_idx{}_all_indx{}.pkl'.format(epoch,batch_idx,all_batch_ind)))





    
def main(args):
    train(args)
    
    
    
if __name__ == '__main__': 
    main(args)







