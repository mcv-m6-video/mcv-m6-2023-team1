# Week 5

## 1. Create dataset for detection
Create train and validation splits for training and inference.

Check the config file to specify which sequences to use.

WARNING: Around 30gb of data will be generated if you select all the sequences and it will
take some time.
````
python create_dataset_detection.py --config=configs/create_dataset_detection.yaml
````

## 2. Train detection model (optional)
Train a Yolov8 model using the previously created dataset (train and validation) in case you
do not have a pretrained Yolov8 model for this task.

Check the config file to specify where are located the different splits.
````
python train_detection.py --config=configs/train_detection.yaml
````

## 3. Track singles cameras
Track single cameras from one sequence using a Yolov8 model and SORT.

It will use Yolov8 model defined in the config variable `model_path`.

It will generate detections from Yolov8 in `data/detection_{model_path}` if they do not exist, and use them
in order to do tracking.

Every cam tracking will be evaluated (IDF1 and HOTA) and saved as a video file in folder `tracking_videos`.

````
python track_multi_camera.py --config=configs/track.yaml
````

## 4. Create dataset for reid
Save detected car frames in ground truth organized in folder for every different car.
````
python create_dataset_reid.py --config=configs/create_dataset_reid.yaml
````

## 5. Metric learning
Metric learning using the previously created datset for reid

````
python train_reid.py --config=configs/train_reid.yaml
````

## 6. Evaluate metric learning
Evaluate validation dataset (Precision, Recall, F1Score)

````
python evaluate_reid.py --config=configs/evaluate_reid.yaml
````