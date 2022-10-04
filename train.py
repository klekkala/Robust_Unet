import argparse
import logging
import os
import sys
import math

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
import pandas as pd

from eval import eval_net
from unet import UNet

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import BasicDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F

dir_img = 'data/imgs/'
dir_mask = 'data/masks/'
dir_checkpoint = 'checkpoints/'

# ==== Bernoulli's ====
"""
def BBFC_loss(Y, X, beta):
    term1 = (1 / beta)
    term2 = (X * torch.pow(Y, beta)) + (1 - X) * torch.pow((1 - Y), beta)
    term2 = torch.prod(term2, dim=1) - 1
    term3 = torch.pow(Y, (beta + 1)) + torch.pow((1 - Y), (beta + 1))
    term3 = torch.prod(term3, dim=1) / (beta + 1)
    loss1 = torch.sum(-term1 * term2 + term3)

    if torch.isnan(loss1):
        print('NaN Loss')

    #print("Loss",loss1)
    return loss1


# Reconstruction + KL divergence losses summed over all elements and batch
def beta_loss_function(pred_mask, true_mask, beta):
    
    w = pred_mask.shape[2]
    h = pred_mask.shape[3]
    
    pred_mask = pred_mask.view(-1, w*h) 
    
    if beta > 0:
        # If beta is nonzero, use the beta entropy
        BBCE = BBFC_loss(pred_mask, true_mask.view(-1, w*h), beta)
    else:
        # if beta is zero use binary cross entropy
        BBCE = F.binary_cross_entropy(pred_mask, true_mask.view(-1, w*h), reduction='sum')

    return BBCE
"""

# Gaussian

# MSE loss
def MSE_loss(Y, X):
    ret = (X - Y)**2
    ret = torch.sum(ret)
    
    return ret
      
# Beta loss
def SE_loss(Y,X):
    ret = (X - Y)**2
    ret = torch.sum(ret,1)
    
    return ret
                  
def Gaussian_CE_loss(Y, X, beta, sigma=1):
    Dim = Y.shape[1]
    const1 = -((1 + beta) / beta)
    const2 = 1 / pow((2 * math.pi * (sigma**2)), (beta * Dim / 2))
    SE = SE_loss(Y, X)
    term1 = torch.exp(-(beta / (2 * (sigma**2))) * SE)
    loss = torch.sum(const1 * (const2* term1 - 1))
    
    return loss
                                
def beta_loss_function(pred_mask, true_mask, beta):
    w = pred_mask.shape[2]
    h = pred_mask.shape[3]
    if beta > 0:
        BBCE = Gaussian_CE_loss(pred_mask.view(-1, w*h), true_mask.view(-1, w*h), beta)
    else:
        BBCE = MSE_loss(pred_mask.view(-1, w*h), true_mask.view(-1, w*h))
    
    return BBCE


def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              beta=0.0,
              outlier=0.0):

    dataset = BasicDataset(dir_img, dir_mask, img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    dataset.set_outlier_indices(train.indices, outlier)
    logging.info(f'Beta {beta}')
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=8, pin_memory=True, drop_last=True)
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0

    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Validation size: {n_val}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images scaling:  {img_scale}
    ''')
    s_model = torch.nn.Softmax(dim=1)
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.n_classes > 1 else 'max', patience=2)
    """
    if net.n_classes > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()
    """
    train_loss = []

    val_scores = []

    for epoch in range(epochs):
        net.train()

        epoch_loss = []

        #with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
        for batch in train_loader:
            imgs = batch['image']
            true_masks = batch['mask']

            assert imgs.shape[1] == net.n_channels, \
                f'Network has been defined with {net.n_channels} input channels, ' \
                f'but loaded images have {imgs.shape[1]} channels. Please check that ' \
                'the images are loaded correctly.'

            imgs = imgs.to(device=device, dtype=torch.float32)
            mask_type = torch.float32 if net.n_classes == 1 else torch.long
            true_masks = true_masks.to(device=device, dtype=mask_type)
            masks_pred = net(imgs)
            loss = beta_loss_function(masks_pred, true_masks, beta)
            epoch_loss.append(loss.item())
            #writer.add_scalar('Loss/train', loss.item(), global_step)

            #pbar.set_postfix(**{'loss (batch)': loss.item()})

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_value_(net.parameters(), 0.1)
            optimizer.step()

            #pbar.update(imgs.shape[0])
            global_step += 1
            if global_step % (n_train // (10 * batch_size)) == 0:
                for tag, value in net.named_parameters():
                    tag = tag.replace('.', '/')
                    #writer.add_histogram('weights/' + tag, value.data.cpu().numpy(), global_step)
                    #writer.add_histogram('grads/' + tag, value.grad.data.cpu().numpy(), global_step)
                val_score = eval_net(net, val_loader, device)
                val_scores.append(val_score)
                scheduler.step(val_score)
                #writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)

                """
                if net.n_classes > 1:
                    logging.info('Validation cross entropy: {}'.format(val_score))
                    writer.add_scalar('Loss/test', val_score, global_step)
                else:
                    logging.info('Validation Dice Coeff: {}'.format(val_score))
                    writer.add_scalar('Dice/test', val_score, global_step)
                """
                #writer.add_images('images', imgs, global_step)
                if net.n_classes == 1:
                    writer.add_images('masks/true', true_masks, global_step)
                    writer.add_images('masks/pred', torch.sigmoid(masks_pred) > 0.5, global_step)
        train_loss.append(sum(epoch_loss) / len(epoch_loss))
        
        if epoch == 50:
            train_pd = pd.DataFrame({'train': train_loss})
            val_pd = pd.DataFrame({'val': val_scores})
            train_pd.to_csv('./plots/train_loss')
            val_pd.to_csv('./plots/val_scores')

        if save_cp:
            try:
                os.mkdir(dir_checkpoint)
                logging.info('Created checkpoint directory')
            except OSError:
                pass
            torch.save(net.state_dict(),
                       dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
            logging.info(f'Checkpoint {epoch + 1} saved !')
   
    # == Train Loss Curves ==
    plt.figure(figsize=(20,10))
    plt.subplot(2, 1, 1)
    plt.title('Training loss')
    plt.plot(train_loss, '-o')
    plt.xlabel('Epoch')
    plt.savefig('./plots/train_loss.png')
    
    # == Validation Loss Curves ==
    plt.figure(figsize=(20,10))
    plt.subplot(2, 1, 1)
    plt.title('Validation Dice Coefficient')
    plt.plot(val_scores, '-o')
    plt.xlabel('Total Batch size/10')
    plt.savefig('./plots/val_score.png')

    writer.close()


def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-e', '--epochs', metavar='E', type=int, default=5,
                        help='Number of epochs', dest='epochs')
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=1,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-l', '--learning-rate', metavar='LR', type=float, nargs='?', default=0.0001,
                        help='Learning rate', dest='lr')
    parser.add_argument('-f', '--load', dest='load', type=str, default=False,
                        help='Load model from a .pth file')
    parser.add_argument('-s', '--scale', dest='scale', type=float, default=0.5,
                        help='Downscaling factor of the images')
    parser.add_argument('-v', '--validation', dest='val', type=float, default=10.0,
                        help='Percent of the data that is used as validation (0-100)')
    parser.add_argument('-be', '--beta', dest='beta', type=float, default=0.0,
                        help='Beta Value for Robust Unet')
    parser.add_argument('-o', '--outlier', dest='outlier', type=float, default=0.0,
                        help='Outlier Percentage for Robust Unet')

    return parser.parse_args()


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    # Change here to adapt to your data
    # n_channels=3 for RGB images
    # n_classes is the number of probabilities you want to get per pixel
    #   - For 1 class and background, use n_classes=1
    #   - For 2 classes, use n_classes=1
    #   - For N > 2 classes, use n_classes=N

    net = UNet(n_channels=3, n_classes=1, bilinear=True)
    # net = torch.hub.load('milesial/Pytorch-UNet', 'unet_carvana')
    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n'
                 f'\t{"Bilinear" if net.bilinear else "Transposed conv"} upscaling')

    if args.load:
        net.load_state_dict(
            torch.load(args.load, map_location=device)
        )
        logging.info(f'Model loaded from {args.load}')

    net.to(device=device)
    # faster convolutions, but more memory
    # cudnn.benchmark = True

    try:
        train_net(net=net,
                  epochs=args.epochs,
                  batch_size=args.batchsize,
                  lr=args.lr,
                  device=device,
                  img_scale=args.scale,
                  val_percent=args.val / 100,
                  beta=args.beta,
                  outlier=args.outlier)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
