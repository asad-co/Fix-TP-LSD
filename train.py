# train.py
import torch
assert torch.cuda.is_available(), "CUDA is required for training"

import os
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import time
import datetime
from config.train_config import TrainParameters
from utils.utils import load_model, save_model, optimizer_define, AverageMeter
from custom_dataset import LineSegmentDataset
import cv2 as cv

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
    
    def forward(self, pred, target):
        pos_inds = target.eq(1).float()
        neg_inds = target.lt(1).float()
        
        pos_loss = -((1 - pred) ** self.gamma) * torch.log(pred + 1e-8) * pos_inds
        neg_loss = -(pred ** self.gamma) * torch.log(1 - pred + 1e-8) * neg_inds
        
        num_pos = pos_inds.float().sum()
        pos_loss = pos_loss.sum()
        neg_loss = neg_loss.sum()
        
        if num_pos == 0:
            loss = neg_loss
        else:
            loss = (pos_loss + neg_loss) / num_pos
        return loss

def update_learning_rate(optimizer, epoch, lr_decay_epochs, lr_decay_rate, initial_lr):
    """
    Update learning rate based on epochs
    """
    lr = initial_lr
    for decay_epoch in lr_decay_epochs:
        if epoch >= decay_epoch:
            lr *= lr_decay_rate
    
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * lr / initial_lr
    
    return lr

def compute_loss(outputs, targets, params):
    """
    Compute the combined loss
    """
    device = outputs[-1]['center'].device
    
    # Get targets
    center_target = targets["center"].to(device)
    dis_target = targets["dis"].to(device)
    line_target = targets["line"].to(device)
    
    # Initialize losses
    center_loss = torch.zeros(1, device=device)
    dis_loss = torch.zeros(1, device=device)
    line_loss = torch.zeros(1, device=device)
    
    # Define loss functions
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    mse_loss = nn.MSELoss(reduction='mean')
    
    # Compute losses for each output in stack
    for i, output in enumerate(outputs):
        # Center loss (focal loss)
        center_pred = torch.sigmoid(output['center'])
        center_loss += focal_loss(center_pred, center_target)
        
        # Line loss (focal loss)
        line_pred = torch.sigmoid(output['line'])
        line_loss += focal_loss(line_pred, line_target)
        
        # Distance loss (MSE loss with mask)
        dis_pred = output['dis']
        mask = line_target > 0.5
        if mask.sum() > 0:
            dis_loss += mse_loss(dis_pred * mask.expand_as(dis_pred), dis_target * mask.expand_as(dis_target))
        
    # Weighted sum of losses
    total_loss = (params.center_weight * center_loss + 
                  params.dis_weight * dis_loss + 
                  params.line_weight * line_loss)
    
    return total_loss, center_loss, dis_loss, line_loss

def validate(model, val_loader, params):
    """
    Validation function
    """
    model.eval()
    val_loss = AverageMeter()
    
    with torch.no_grad():
        for i, data in enumerate(val_loader):
            # Get data and move to correct device
            inputs = data["input"].cuda()
            targets = {
                "center": data["center"].cuda(),
                "dis": data["dis"].cuda(),
                "line": data["line"].cuda()
            }
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss, _, _, _ = compute_loss(outputs, targets, params)
            
            # Update metrics
            val_loss.update(loss.item(), inputs.size(0))
    
    return val_loss.avg

def train(params):
    """
    Main training function
    """
    # Check CUDA availability
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available, but training requires GPU")
    
    # Set device - use the standard PyTorch approach
    device = torch.device("cuda:0")
    torch.cuda.set_device(device)
    print(f"Using CUDA device {torch.cuda.current_device()}: {torch.cuda.get_device_name(0)}")
    
    # Create datasets and data loaders
    train_dataset = LineSegmentDataset(
        params.train_file, params.image_dir, params.label_dir, params, is_train=True
    )
    val_dataset = LineSegmentDataset(
        params.val_file, params.image_dir, params.label_dir, params, is_train=False
    )
    
    train_loader = DataLoader(
        train_dataset, batch_size=params.batch_size, shuffle=True,
        num_workers=params.num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=params.batch_size, shuffle=False,
        num_workers=params.num_workers, pin_memory=True
    )
    
    # Create the model on demand
    model = params.create_model()
    model = load_model(model, params.load_model_path, params.resume, params.selftrain)
    if torch.cuda.is_available():
        model = model.cuda()
        model = nn.DataParallel(model)
    
    # Initialize optimizer
    optimizer = optimizer_define(model, params.optim_weight, params.learning_rate)
    
    # Training variables
    start_epoch = 0
    best_val_loss = float('inf')
    current_lr = params.learning_rate
    
    # Resume training if needed
    if params.resume and os.path.isfile(params.load_model_path):
        checkpoint = torch.load(params.load_model_path)
        start_epoch = checkpoint.get('epoch', 0)
        current_lr = checkpoint.get('lr', params.learning_rate)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint.get('optimizer', optimizer.state_dict()))
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resuming from epoch {start_epoch}")
    
    # Training loop
    for epoch in range(start_epoch, params.epochs):
        # Update learning rate
        current_lr = update_learning_rate(optimizer, epoch, params.lr_decay_epochs, params.lr_decay_rate, params.learning_rate)
        
        # Training phase
        model.train()
        epoch_loss = AverageMeter()
        center_loss_meter = AverageMeter()
        dis_loss_meter = AverageMeter()
        line_loss_meter = AverageMeter()
        batch_time = AverageMeter()
        data_time = AverageMeter()
        
        end = time.time()
        for i, data in enumerate(train_loader):
            # Measure data loading time
            data_time.update(time.time() - end)
            
            # Get data
            inputs = data["input"].cuda()
            targets = {
                "center": data["center"].cuda(),
                "dis": data["dis"].cuda(),
                "line": data["line"].cuda()
            }
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss, center_loss, dis_loss, line_loss = compute_loss(outputs, targets, params)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update metrics
            epoch_loss.update(loss.item(), inputs.size(0))
            center_loss_meter.update(center_loss.item(), inputs.size(0))
            dis_loss_meter.update(dis_loss.item(), inputs.size(0))
            line_loss_meter.update(line_loss.item(), inputs.size(0))
            
            # Measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()
            
            # Print progress
            if i % 10 == 0:
                print(f'Epoch: [{epoch+1}/{params.epochs}][{i+1}/{len(train_loader)}] '
                      f'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s) '
                      f'Data {data_time.val:.3f}s ({data_time.avg:.3f}s) '
                      f'Loss {epoch_loss.val:.4f} ({epoch_loss.avg:.4f}) '
                      f'C {center_loss_meter.avg:.4f} D {dis_loss_meter.avg:.4f} L {line_loss_meter.avg:.4f}')
        
        # Validation phase
        val_loss = validate(model, val_loader, params)
        
        # Print epoch summary
        print(f'Epoch: {epoch+1}/{params.epochs} '
              f'Training Loss: {epoch_loss.avg:.4f} '
              f'Validation Loss: {val_loss:.4f} '
              f'LR: {current_lr:.8f}')
        
        # Save checkpoint
        is_best = val_loss < best_val_loss
        best_val_loss = min(val_loss, best_val_loss)
        
        # Save model
        save_path = os.path.join(params.save_path, f'model_epoch_{epoch+1}.pth')
        best_model_path = os.path.join(params.save_path, 'model_best.pth')
        
        save_dict = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'lr': current_lr
        }
        
        if (epoch + 1) % params.save_interval == 0:
            torch.save(save_dict, save_path)
            print(f'Saved checkpoint to {save_path}')
        
        if is_best:
            torch.save(save_dict, best_model_path)
            print(f'Saved best model to {best_model_path}')

if __name__ == '__main__':
    print("Starting Training: Gathering Train Parameters")
    params = TrainParameters()
    print("Parameters acquired! starting training")
    train(params)