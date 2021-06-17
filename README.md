# Code Structure
* [`dataset/`](dataset/):
    * [`train_dataset.py`](dataset/train_dataset.py): The training and valid dataset script.
    * [`test_dataset.py`](dataset/test_dataset.py): The test dataset script.
    * [`transforms.py`](dataset/transforms.py): Useful transforms for dataset processing.
* [`model/`](model/):
    * [`UNet.py`](model/UNet.py): The 3D U-Net model.
* [`utils/`](utils/): The useful scripts.
* [`result/`](result/):
    * [`best_model.pth`](result/best_model.pth): The model used for predict.
    * [`test_predict.zip`](result/test_predict.zip): The test dataset prediction results.
    * [`val_predict.zip`](result/val_predict.zip): The val dataset prediction results.
* [`train.py`](train.py): The training script.
* [`predict.py`](predict.py): The script to predict the label.
* [`config.py`](config.py): The script to get the parameters required for training.
* [`plot.py`](plot.py): The script to plot the needed curve.

## How to Run

### Training

```bash
python train.py --lr 0.001 --gpu_id=[0,1,2,3] --batch_size 8 --train_image_dir 'datasets/ribfrac-train-images/' --train_label_dir 'datasets/ribfrac-train-labels/' --val_image_dir 'datasets/ribfrac-val-images/' --val_label_dir 'datasets/ribfrac-val-labels/' 
```

+ `lr:` learning rate (default: 0.001)
+ `gpu_id:` gpu id list used for training.
+ `batch_size:` batch size of trainset
+ `train_image_dir:` The folder where the training images are saved.
+ `train_label_dir:` The folder where the training labels are saved.
+ `val_image_dir:` The folder where the val images are saved.
+ `val_label_dir:` The folder where the val labels are saved.

For more parameter information, please check the [`config.py`](config.py) script.

### Predict

```bash
python predict.py --gpu 0 --model_path ./result/best_model.pth --pred_dir ./predict --test_data_path ./datasets/ribfrac-test-images/
```

+ `gpu:` gpu id used for predict.
+ `model_path:` The model path used for predict.
+ `pred_dir:` The folder to store the predict result.
+ `test_data_path:` The folder where the test images are saved.

For more parameter information, please check the [`config.py`](config.py) script.

