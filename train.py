import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

import os
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import matplotlib.pyplot as plt

import config
from model.UNet import UNet
from dataset.train_dataset import TrainDataset
from utils import logger, metrics, common
from utils.loss import MixLoss, DiceLoss

def val(model, val_loader, loss_func, n_labels):
    model.eval()
    val_loss = metrics.LossAverage()
    val_dice = metrics.DiceAverage(n_labels)
    with torch.no_grad():
        for idx,(data, target) in tqdm(enumerate(val_loader),total=len(val_loader)):
            data, target = data.float(), target.long()
            target = common.to_one_hot_3d(target, n_labels)
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss=loss_func(output, target)
            
            val_loss.update(loss.item(),data.size(0))
            val_dice.update(output, target)
    val_log = OrderedDict({'Val_Loss': val_loss.avg, 'Val_Dice': val_dice.avg[1], 'Val_Precision': val_dice.precision, 'Val_Recall': val_dice.recall, 'Val_F1': val_dice.F1})
    return val_log

def train(model, train_loader, optimizer, loss_func, n_labels, alpha):
    print("=======Epoch:{}=======lr:{}".format(epoch,optimizer.state_dict()['param_groups'][0]['lr']))
    model.train()
    train_loss = metrics.LossAverage()
    train_dice = metrics.DiceAverage(n_labels)

    for idx, (data, target) in tqdm(enumerate(train_loader),total=len(train_loader)):
        data, target = data.float(), target.long()
        target = common.to_one_hot_3d(target,n_labels)
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        output = model(data)
        loss = loss_func(output, target)

        loss.backward()
        optimizer.step()
        
        train_loss.update(loss.item(),data.size(0))
        train_dice.update(output, target)
        # train_dice.update(output, target)

    val_log = OrderedDict({'Train_Loss': train_loss.avg, 'Train_Dice': train_dice.avg[1], 'Train_Precision': train_dice.precision, 'Train_Recall': train_dice.recall, 'Train_F1': train_dice.F1})
    return val_log


if __name__ == '__main__':
    args = config.args
    save_path = os.path.join('./runs', args.save_path)
    if not os.path.exists(save_path): os.makedirs(save_path)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # data info
    ds_train = TrainDataset(args, args.train_image_dir, args.train_label_dir)
    train_loader = DataLoader(dataset=ds_train, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=common.train_collate_fn)
    # train_loader = TrainDataset.get_dataloader(ds_train, args.batch_size, False, args.workers)
    ds_val = TrainDataset(args, args.val_image_dir, args.val_label_dir)
    val_loader = DataLoader(dataset=ds_val, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, collate_fn=common.train_collate_fn)
    # val_loader = TrainDataset.get_dataloader(ds_val, args.batch_size, False, args.workers)

    # model info
    model = UNet(1, args.n_labels).to(device)
    model = torch.nn.DataParallel(model, device_ids=args.gpu_id)  # multi-GPU
    if args.weight is not None:
        checkpoint = torch.load(args.weight)

        model.load_state_dict(checkpoint['net'])

        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        optimizer.load_state_dict(checkpoint['optimizer'])

        start_epoch = checkpoint['epoch'] + 1
        log = logger.Train_Logger(save_path,"train_log",init=os.path.join(save_path,"train_log.csv"))
    else:
        log = logger.Train_Logger(save_path,"train_log")
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
        start_epoch = 1
    common.print_network(model)
 
    loss = MixLoss(nn.BCEWithLogitsLoss(), 0.5, DiceLoss(), 1)
    
    if log.log is not None:
        best = [log.log.idxmax()['Val_Dice']+1, log.log.max()['Val_Dice']]
    else:
        best = [0,0]
    trigger = 0  # early stop 计数器
    alpha = 0.4 # 深监督衰减系数初始值
    for epoch in range(start_epoch, start_epoch + args.epochs):
        common.adjust_learning_rate(optimizer, epoch, args)
        train_log = train(model, train_loader, optimizer, loss, args.n_labels, alpha)
        val_log = val(model, val_loader, loss, args.n_labels)
        log.update(epoch,train_log,val_log)

        # Save checkpoint.
        state = {'net': model.state_dict(),'optimizer':optimizer.state_dict(),'epoch': epoch}
        torch.save(state, os.path.join(save_path, 'latest_model.pth'))
        trigger += 1
        if val_log['Val_Dice'] > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_path, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_log['Val_Dice']
            trigger = 0
        print('Best performance at Epoch: {} | {}'.format(best[0],best[1]))

        # 深监督系数衰减
        if epoch % 30 == 0: alpha *= 0.8

        # early stopping
        if args.early_stop is not None:
            if trigger >= args.early_stop:
                print("=> early stopping")
                break
        torch.cuda.empty_cache()

        ax = log.log.plot(x='epoch', y='Val_Dice', grid=True, title='Val_Dice')
        fig = ax.get_figure()
        fig.savefig(os.path.join(save_path, 'dice.png'))
        # plt.show()
        ax = log.log.plot(x='epoch', y='Val_Loss', grid=True, title='Val_Loss')
        fig = ax.get_figure()
        fig.savefig(os.path.join(save_path, 'loss.png'))
        # plt.show()
        ax = log.log.plot(x='epoch', y='Val_Recall', grid=True, title='Val_Recall')
        fig = ax.get_figure()
        fig.savefig(os.path.join(save_path, 'recall.png'))
        # plt.show()