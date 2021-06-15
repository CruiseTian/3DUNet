import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.ticker as ticker
import os, sys, copy
import numpy as np
import pandas as pd

from pylab import rcParams
import matplotlib.pylab as pylab

rcParams['legend.numpoints'] = 1
mpl.style.use('seaborn')
# plt.rcParams['axes.facecolor']='binary'
# print(rcParams.keys())
params = {
    'font.family': 'sans-serif',
    'font.sans-serif': 'Times New Roman',
    'font.weight': 'bold'
}
pylab.rcParams.update(params)

def plot(ex_id):
    fig, ax = plt.subplots(1, 3, figsize=(15, 6), tight_layout=True)
    # ax = ax.ravel()
    s = ['train loss', 'val loss', 'train recall', 'val recall', 'train dice', 'val dice']

    filename = 'ex'+str(ex_id)+'_train_log.csv'
    data = pd.read_csv(filename, delimiter=',')

    ax[0].plot(data.epoch, data.Train_Loss, '-', c='#e41b1b', label=s[0], linewidth=1.5)
    ax[0].plot(data.epoch, data.Val_Loss, '-', c='#377eb8', label=s[1], linewidth=1.5)
    ax[0].set_xlabel('Epoch', fontweight='bold', fontsize=16)  # x轴字体大小
    ax[0].set_ylabel('Loss', fontweight='bold', fontsize=16)  # y轴字体大小
    ax[0].tick_params(labelsize=13)
    ax[0].set_title('train loss and val loss',fontweight='bold',fontsize=18)
    ax[0].legend(loc='best',fancybox=True, framealpha=0,fontsize=16)
    ax[0].grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度

    ax[1].plot(data.epoch, data.Train_Recall, '-', c='#e41b1b', label=s[2], linewidth=1.5)
    ax[1].plot(data.epoch, data.Val_Recall, '-', c='#377eb8', label=s[3], linewidth=1.5)
    ax[1].set_xlabel('Epoch', fontweight='bold', fontsize=16)  # x轴字体大小
    ax[1].set_ylabel('Recall', fontweight='bold', fontsize=16)  # y轴字体大小
    ax[1].tick_params(labelsize=13)
    ax[1].set_title('train recall and val recall',fontweight='bold',fontsize=18)
    ax[1].legend(loc='best',fancybox=True, framealpha=0,fontsize=16)
    ax[1].grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度

    ax[2].plot(data.epoch, data.Train_dice_frac, '-', c='#e41b1b', label=s[4], linewidth=1.5)
    ax[2].plot(data.epoch, data.Val_dice_frac, '-', c='#377eb8', label=s[5], linewidth=1.5)
    ax[2].set_xlabel('Epoch', fontweight='bold', fontsize=16)  # x轴字体大小
    ax[2].set_ylabel('Dice', fontweight='bold', fontsize=16)  # y轴字体大小
    ax[2].tick_params(labelsize=13)
    ax[2].set_title('train dice and val dice',fontweight='bold',fontsize=18)
    ax[2].legend(loc='best',fancybox=True, framealpha=0,fontsize=16)
    ax[2].grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度
    
    plt.savefig('ex{}_curve.png'.format(ex_id), format='png', dpi=400)
    plt.show()

plot(6)