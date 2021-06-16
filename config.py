import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--workers', type=int, default=4,help='number of threads for data loading')
parser.add_argument('--gpu_id', type=list,default=[0,1,2,3], help='multi-GPU')

# Preprocess parameters
parser.add_argument('--weight', type=str, default=None, help='model init weight')
parser.add_argument('--n_labels', type=int, default=2,help='number of classes')
parser.add_argument('--upper', type=int, default=1000, help='')
parser.add_argument('--lower', type=int, default=-200, help='')

# data in/out and dataset
parser.add_argument("--train_image_dir", default = '../unet/datasets/ribfrac-train-images/',
    help="The training image nii directory.")
parser.add_argument("--train_label_dir", default = '../unet/datasets/ribfrac-train-labels/',
    help="The training label nii directory.")
parser.add_argument("--val_image_dir", default = '../unet/datasets/ribfrac-val-images/',
    help="The validation image nii directory.")
parser.add_argument("--val_label_dir", default = '../unet/datasets/ribfrac-val-labels/',
    help="The validation label nii directory.")
parser.add_argument('--save_path',default='ex1',help='save path of trained model')
parser.add_argument('--batch_size', type=list, default=8,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',help='learning rate (default: 0.001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=64)

# test
parser.add_argument('--gpu', type=int,default=0, help='predict GPU')
parser.add_argument('--postprocess', type=bool, default=True, help='post process')
parser.add_argument('--model_path', default='./runs/ex3/best_model.pth',help='test model path')
parser.add_argument("--pred_dir", default='./predict', help="The directory for saving predictions.")
parser.add_argument('--test_data_path',default = '../unet/datasets/ribfrac-test-images/',help='Testset path')
parser.add_argument("--prob_thresh", default=0.1, type=float, help="Prediction probability threshold.")
parser.add_argument("--bone_thresh", default=300, type=int, help="Bone binarization threshold.")
parser.add_argument("--size_thresh", default=200, type=int, help="Prediction size threshold.")


args = parser.parse_args()


