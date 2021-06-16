# Code Structure
* [`dataset/`](dataset/):
    * [`train_dataset.py`](dataset/train_dataset.py): The training and valid dataset script.
    * [`test_dataset.py`](dataset/test_dataset.py): The test dataset script.
    * [`transforms.py`](dataset/transforms.py): Useful transforms for dataset processing.
* [`model/`](model/):
    * [`UNet.py`](model/UNet.py): The 3D U-Net model.
* [`utils/`](utils/): The useful scripts.
* [`result/`](result/):
    * [`test_predict.zip`](result/test_predict.zip): The test dataset prediction results.
    * [`val_predict.zip`](result/val_predict.zip): The val dataset prediction results.
* [`train.py`](train.py): The training script.
* [`predict.py`](predict.py): The script to predict the label.
* [`config.py`](config.py): The script to get the parameters required for training.
* [`plot.py`](plot.py): The script to plot the needed curve.