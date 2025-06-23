import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import pytorch_ssim
from net.model import SS_UIE_model
from utils.utils import *
from utils.LAB import *
from utils.LCH import *
from utils.FDL import *
import cv2
import time as time
import datetime
import sys
from torchvision.utils import save_image
import csv
import random
import torch.utils.data as dataf
import torch.nn.functional as F
import argparse
from torch.cuda.amp import autocast, GradScaler


# 解析命令行参数
def parse_args():
    parser = argparse.ArgumentParser(description='SS-UIE Training')
    parser.add_argument('--use_amp', action='store_true', default=False,
                        help='Enable Automatic Mixed Precision training')
    parser.add_argument('--gpu_ids', type=str, default='0',
                        help='GPU IDs to use (e.g., "0" for single GPU, "0,1" for multi-GPU)')
    parser.add_argument('--batch_size', type=int, default=4,
                        help='Batch size for training')
    parser.add_argument('--epochs', type=int, default=600,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='Learning rate')
    return parser.parse_args()

# 配置设备和GPU
def setup_device(gpu_ids_str):
    """Setup device configuration for single/multi-GPU training"""
    if not torch.cuda.is_available():
        print("CUDA is not available. Using CPU.")
        return torch.device('cpu'), None, False
    
    # 解析GPU IDs
    gpu_ids = [int(id.strip()) for id in gpu_ids_str.split(',') if id.strip()]
    
    # 设置CUDA_VISIBLE_DEVICES
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_ids_str
    
    # 重新映射device_ids（因为CUDA_VISIBLE_DEVICES会重新编号）
    device_ids = list(range(len(gpu_ids)))
    
    device = torch.device(f"cuda:{device_ids[0]}")
    is_multi_gpu = len(device_ids) > 1
    
    print(f"Using GPU(s): {gpu_ids_str}")
    print(f"Device IDs after mapping: {device_ids}")
    print(f"Main device: {device}")
    print(f"Multi-GPU training: {is_multi_gpu}")
    
    return device, device_ids, is_multi_gpu

dtype = 'float32'

# 如果脚本被直接运行，解析参数；否则使用默认值（兼容Jupyter）
if __name__ == "__main__":
    args = parse_args()
else:
    # Jupyter notebook兼容模式
    class Args:
        use_amp = False
        gpu_ids = '0'
        batch_size = 4
        epochs = 600
        lr = 0.0002
    args = Args()

# 设置设备
device, device_ids, is_multi_gpu = setup_device(args.gpu_ids)
torch.set_default_tensor_type(torch.FloatTensor)



def sample_images(batches_done, SS_UIE, x_test, Y_test, device, use_amp=False):
    """Saves a generated sample from the validation set"""
    SS_UIE.eval()
    i = random.randrange(1, min(500, len(x_test)))
    
    with torch.no_grad():
        real_A = x_test[i,:,:,:].to(device).unsqueeze(0)
        real_B = Y_test[i,:,:,:].to(device).unsqueeze(0)
        
        if use_amp:
            with autocast():
                fake_B = SS_UIE(real_A)
        else:
            fake_B = SS_UIE(real_A)
        
        imgx = fake_B.data
        imgy = real_B.data
        x = imgx[:,:,:,:]
        y = imgy[:,:,:,:]
        img_sample = torch.cat((x,y), -2)
        save_image(img_sample, "images/%s/%s.png" % ('results', batches_done), nrow=5, normalize=True)
    
    SS_UIE.train()
    x=imgx[:,:,:,:]
    y=imgy[:,:,:,:]
    img_sample = torch.cat((x,y), -2)
    save_image(img_sample, "images/%s/%s.png" % ('results', batches_done), nrow=5, normalize=True)#要改




training_x=[]
path='./data/LSUI/SS-UIE/train/input/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    training_x.append(imgx)   

X_train = []
for features in training_x:
    X_train.append(features)

X_train = np.array(X_train)
X_train=X_train.astype(dtype)
X_train= torch.from_numpy(X_train)
X_train=X_train.permute(0,3,1,2)
#X_train=X_train.unsqueeze(1)
X_train=X_train/255.0
print("input shape:",X_train.shape)


training_y=[]
path='./data/LSUI/SS-UIE/train/GT/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    training_y.append(imgx)


y_train = []
for features in training_y:
    y_train.append(features)

y_train = np.array(y_train)
y_train=y_train.astype(dtype)
y_train= torch.from_numpy(y_train)
y_train=y_train.permute(0,3,1,2)
y_train=y_train/255.0
print("output shape:",y_train.shape)


test_x=[]
path='./data/LSUI/SS-UIE/test/input/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    test_x.append(imgx)


x_test = []
for features in test_x:
    x_test.append(features)

x_test = np.array(x_test)
x_test=x_test.astype(dtype)
x_test= torch.from_numpy(x_test)
x_test=x_test.permute(0,3,1,2)
x_test=x_test/255.0
print("test input shape:",x_test.shape)

test_Y=[]
path='./data/LSUI/SS-UIE/test/GT/'#要改
path_list = os.listdir(path)
path_list.sort(key=lambda x:int(x.split('.')[0]))
for item in path_list:
    impath=path+item
    #print("开始处理"+impath)
    imgx= cv2.imread(path+item)
    imgx = cv2.cvtColor(imgx, cv2.COLOR_BGR2RGB)
    imgx=cv2.resize(imgx,(256,256))
    test_Y.append(imgx)


Y_test = []
for features in test_Y:
    Y_test.append(features)

Y_test = np.array(Y_test)
Y_test=Y_test.astype(dtype)
Y_test= torch.from_numpy(Y_test)
Y_test=Y_test.permute(0,3,1,2)
Y_test=Y_test/255.0
print("test output shape:",Y_test.shape)





dataset = dataf.TensorDataset(X_train, y_train)
loader = dataf.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)

# 初始化模型
SS_UIE = SS_UIE_model(in_channels=3, channels=16, num_resblock=4, num_memblock=4)
SS_UIE = SS_UIE.to(device)

# 根据GPU数量决定是否使用DataParallel
if is_multi_gpu and torch.cuda.is_available():
    SS_UIE = torch.nn.DataParallel(SS_UIE, device_ids=device_ids)
    print(f"Using DataParallel with {len(device_ids)} GPUs")
else:
    print("Using single GPU or CPU")

# 初始化损失函数
MSE = nn.L1Loss(reduction='sum').to(device)
SSIM = pytorch_ssim.SSIM().to(device)
L_lab = lab_Loss().to(device)
L_lch = lch_Loss().to(device)
FDL_loss = FDL(loss_weight=1.0, alpha=2.0, patch_factor=4, ave_spectrum=True, log_matrix=True, batch_matrix=True).to(device)

# 初始化AMP scaler
scaler = GradScaler() if args.use_amp else None
if args.use_amp:
    print("Using Automatic Mixed Precision (AMP)")

optimizer = torch.optim.Adam(SS_UIE.parameters(), lr=args.lr, betas=(0.5, 0.999), weight_decay=5e-5)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=200, gamma=0.4)

use_pretrain = False
if use_pretrain:
    # Load pretrained models
    start_epoch = 967
    checkpoint = torch.load("saved_models/SS_UIE_%d.pth" % (start_epoch), map_location=device)
    if is_multi_gpu:
        SS_UIE.load_state_dict(checkpoint)
    else:
        # 如果当前是单GPU但保存的是多GPU模型，需要处理module.前缀
        if list(checkpoint.keys())[0].startswith('module.'):
            checkpoint = {k[7:]: v for k, v in checkpoint.items()}
        SS_UIE.load_state_dict(checkpoint)
    print('successfully loading epoch {} 成功！'.format(start_epoch))
else:
    start_epoch = 0
    print('No pretrain model found, training will start from scratch！')

# ----------
#  Training
# ----------

# Create directories for saving images and logs if they don't exist
os.makedirs("images/results", exist_ok=True)
os.makedirs("saved_models", exist_ok=True)

f1 = open('psnr.csv','w',encoding='utf-8')#要改
csv_writer1 = csv.writer(f1)
f2 = open('SSIM.csv','w',encoding='utf-8')#要改
csv_writer2 = csv.writer(f2)

f2 = open('SSIM.csv','w',encoding='utf-8')#要改
csv_writer2 = csv.writer(f2)

checkpoint_interval = 5
epochs = start_epoch
n_epochs = args.epochs
sample_interval = 200

# ignored when opt.mode=='S'
psnr_max = 0
psnr_list = [] 
prev_time = time.time()

for epoch in range(epochs, n_epochs):
    psnr_list = []
    SS_UIE.train()
    
    for i, batch in enumerate(loader):
        # Model inputs
        Input = batch[0].to(device).contiguous() 
        GT = batch[1].to(device).contiguous()

        # ------------------
        #  Train 
        # ------------------
        optimizer.zero_grad()

        if args.use_amp:
            # 使用AMP训练
            with autocast():
                output = SS_UIE(Input)
                loss_RGB = MSE(output, GT) / (GT.size()[2] ** 2)
                loss_lab = (L_lab(output, GT) + L_lab(output, GT) + L_lab(output, GT) + L_lab(output, GT)) / 4.0
                loss_lch = (L_lch(output, GT) + L_lch(output, GT) + L_lch(output, GT) + L_lch(output, GT)) / 4.0    
                loss_ssim = 1 - SSIM(output, GT)
                fdl_loss = FDL_loss(output, GT)
                loss_final = loss_ssim * 10 + loss_RGB * 10 + loss_lch + loss_lab * 0.0001 + fdl_loss * 10000
            
            # 使用scaler进行反向传播
            scaler.scale(loss_final).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            # 常规训练
            output = SS_UIE(Input)
            loss_RGB = MSE(output, GT) / (GT.size()[2] ** 2)
            loss_lab = (L_lab(output, GT) + L_lab(output, GT) + L_lab(output, GT) + L_lab(output, GT)) / 4.0
            loss_lch = (L_lch(output, GT) + L_lch(output, GT) + L_lch(output, GT) + L_lch(output, GT)) / 4.0    
            loss_ssim = 1 - SSIM(output, GT)
            fdl_loss = FDL_loss(output, GT)
            loss_final = loss_ssim * 10 + loss_RGB * 10 + loss_lch + loss_lab * 0.0001 + fdl_loss * 10000
            
            loss_final.backward()
            optimizer.step()

        # 计算指标
        ssim_value = -(loss_ssim.item() - 1)

        # --------------
        #  Log Progress
        # --------------

        # Determine approximate time left
        batches_done = epoch * len(loader) + i
        batches_left = n_epochs * len(loader) - batches_done
        out_train = torch.clamp(output, 0., 1.) 
        psnr_train = batch_PSNR(out_train, GT, 1.)
        time_left = datetime.timedelta(seconds=batches_left * (time.time() - prev_time))
        prev_time = time.time()

        # Print log
        if batches_done % 100 == 0:
            sys.stdout.write(
                "\r[Epoch %d/%d] [Batch %d/%d][PSNR: %2f] [SSIM: %2f][loss: %2f][loss_lch: %2f][loss_lab: %2f][fdl_loss: %2f] ETA: %2s"
                % (
                    epoch,
                    n_epochs,
                    i,
                    len(loader),
                    psnr_train,
                    ssim_value,
                    loss_final.item(),
                    loss_lch.item(),
                    loss_lab.item() * 0.0001,
                    fdl_loss.item() * 5000, 
                    time_left,
                )
            )

        # If at sample interval save image
        if batches_done % sample_interval == 0:
            sample_images(batches_done, SS_UIE, x_test, Y_test, device, args.use_amp)
            csv_writer1.writerow([str(psnr_train)])
            csv_writer2.writerow([str(ssim_value)])
        psnr_list.append(psnr_train)

    PSNR_epoch = np.array(psnr_list)
    if PSNR_epoch.mean() > psnr_max:
        # 保存模型时根据是否多GPU决定保存方式
        model_state_dict = SS_UIE.module.state_dict() if is_multi_gpu else SS_UIE.state_dict()
        torch.save(model_state_dict, "saved_models/SS_UIE_%d.pth" % (epoch))
        psnr_max = PSNR_epoch.mean()
        print("")
        print('A checkpoint Saved PSNR= %f' % (psnr_max))

    scheduler.step()

# 关闭CSV文件
f1.close()
f2.close()

print("Training completed!")

if __name__ == "__main__":
    print('''
使用说明:

1. 单GPU训练 (默认使用GPU 0):
   python train.py

2. 单GPU训练 (指定GPU):
   python train.py --gpu_ids 1

3. 多GPU训练:
   python train.py --gpu_ids 0,1,2,3

4. 启用AMP混合精度训练:
   python train.py --use_amp

5. 完整参数示例:
   python train.py --use_amp --gpu_ids 0,1 --batch_size 8 --epochs 600 --lr 0.0002

6. 后台运行训练:
   nohup python train.py --use_amp --gpu_ids 0,1 > "log/train_$(date +%Y%m%d_%H%M%S).log" 2>&1 &

参数说明:
--use_amp: 启用自动混合精度训练，可以节省显存并加速训练
--gpu_ids: 指定要使用的GPU ID，多个GPU用逗号分隔
--batch_size: 批次大小
--epochs: 训练轮数
--lr: 学习率
    ''')