# Week 5

## 1. Create dataset
Create train and validation splits for training and inference.

Check the config file to specify which sequences to use.

WARNING: Around 30gb of data will be generated if you select all the sequences and it will
take some time.
````
python create_dataset.py --config=create_dataset.yaml
````


## 2. Train detection model (optional)
Train a Yolov8 model using the previously created dataset (train and validation) in case you
do not have a pretrained Yolov8 model for this task.

Check the config file to specify where are located the different splits.
````
python train_detection.py --config=train_detection.yaml
````

## 3. Track multiple cameras (without Re-ID)
Track multiple cameras from one sequence using a Yolov8 model and SORT.
It will track all the cameras that are found in the path (data_path) specified in the config file.

No Re-ID is implemented in this version.
````
python track_multi_camera.py --config=track.yaml
````