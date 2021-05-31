import argparse

parser = argparse.ArgumentParser(description='Hyper-parameters management')

# Hardware options
parser.add_argument('--seed', type=int, default=2021, help='random seed')

# Preprocess parameters
parser.add_argument('--weight', type=str, default=None, help='model init weight')
parser.add_argument('--n_labels', type=int, default=2,help='number of classes')
parser.add_argument('--upper', type=int, default=1000, help='')
parser.add_argument('--lower', type=int, default=-200, help='')
parser.add_argument('--xy_down_scale', type=float, default=0.5, help='')
parser.add_argument('--slice_down_scale', type=float, default=1.0, help='')

# data in/out and dataset
parser.add_argument('--dataset_path',default = '/content/gdrive/Shareddrives/课程实验/datasets/',help='fixed trainset root path')
parser.add_argument('--save_path',default='/content/gdrive/Shareddrives/课程实验/',help='save path of trained model')
parser.add_argument('--batch_size', type=int, default=2,help='batch size of trainset')

# train
parser.add_argument('--epochs', type=int, default=200, metavar='N',help='number of epochs to train (default: 200)')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',help='learning rate (default: 0.0001)')
parser.add_argument('--early-stop', default=30, type=int, help='early stopping (default: 30)')
parser.add_argument('--crop_size', type=int, default=64)

# test
parser.add_argument('--postprocess', type=bool, default=True, help='post process')
parser.add_argument('--model_path', default='/content/gdrive/Shareddrives/课程实验/runs/best_model.pth',help='test model path')
parser.add_argument("--pred_dir", required=True, default='/content/gdrive/Shareddrives/课程实验/predict', help="The directory for saving predictions.")
parser.add_argument('--test_data_path',default = '/content/gdrive/Shareddrives/课程实验/datasets/ribfrac-test-images/',help='Testset path')
parser.add_argument("--prob_thresh", default=0.1, help="Prediction probability threshold.")
parser.add_argument("--bone_thresh", default=300, help="Bone binarization threshold.")
parser.add_argument("--size_thresh", default=100, help="Prediction size threshold.")


args = parser.parse_args()


