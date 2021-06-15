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
    fig, ax = plt.subplots(1, 1, figsize=(9, 6), tight_layout=True)
    # ax = ax.ravel()
    s = ['train loss', 'val loss']

    filename = 'ex'+str(ex_id)+'_train_log.csv'
    data = pd.read_csv(filename, delimiter=',')

    ax.plot(data.epoch, data.Train_Loss, '-', c='#e41b1b', label=s[0], linewidth=1.5)
    ax.plot(data.epoch, data.Val_Loss, '-', c='#377eb8', label=s[1], linewidth=1.5)
    plt.xticks(fontsize=14)  # x轴字体大小
    plt.yticks(fontsize=14)  # y轴字体大小
    ax.set_title('train loss and val loss',fontweight='bold',fontsize=18)
    ax.legend(loc='best',fancybox=True, framealpha=0,fontsize=16)
    ax.grid(True, linestyle='dotted')  # x坐标轴的网格使用主刻度
    plt.savefig('ex{}_loss.png'.format(ex_id), format='png', dpi=400)
    plt.show()

plot(3)