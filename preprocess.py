import torch
from os.path import join
import config
from torch.utils.data import DataLoader
from dataset.dataset_lits_val import Val_Dataset
from dataset.dataset_lits_train import Train_Dataset

def storeData(args):
    # data info
    train_loader = DataLoader(dataset=Train_Dataset(args),batch_size=args.batch_size,shuffle=False)
    # val_loader = DataLoader(dataset=Val_Dataset(args),batch_size=1,shuffle=False)
    store_path = join(args.dataset_path, "process")
    torch.save(train_loader, join(store_path, 'train_loader.pth'))
    print("Done!")


if __name__ == '__main__':
    args = config.args
    storeData(args)
