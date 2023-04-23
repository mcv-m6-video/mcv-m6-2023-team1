# MCV M6 - Video Analysis
# Project: Video Surveillance for Road

## Team 1 Members
- Rachid Boukir ([email](mailto:rachid.boukir@autonoma.cat))
- Josep Bravo ([email](mailto:pepbravo11@gmail.com))
- Alex Martin ([email](mailto:alex.martin@midokura.com))
- Guillem Martinez ([email](mailto:guillemmarsan@gmail.com))
- Miquel Romero ([email](mailto:miquel.robla@gmail.com))

## Introduction
The goal of this project is to learn the basic concepts and techniques related to video sequences mainly for surveillance applications. In this case, we will focus on road traffic monitoring taking as the scope of the project the following techniques/tasks:
- Use of statistical models to estimate the background information of the video sequence.
- Use of deep learning techniques to detect the foreground.
- Use Optical Flow estimations and compensations.
- Track detections with tracking algorithms.
- Analyze system performance evaluation.

In the first stage of the project, we will focus on background and foreground estimation using simple statistical models as adaptaive and non-adaptive Gaussians, and we will compare them with more complex models as MOG2 (Mixture Of Gaussian), LSBP (Local SVD Binary Pattern), KNN (K-Nearest Neighbours), etc.
Then in the second stage we will use the foreground to perform object detections and tracking with and without using Optical Flow in a single camera system.
Finally, in the last stage we will use our best detection and tracking method found in the previous stages to apply them in this stage in a multi-camera tracking system.
 
## Installation
### Requirements
- Python ≥ 3.7
- Pytorch ≥ 1.8 and and torchvision that matches the PyTorch installation. Install them together at [pytorch.org](https://pytorch.org/).
- Detectron2 is required for week 3 tasks. You can install it following the instructions in its official [github repository](https://github.com/facebookresearch/detectron2).
- The other needed packages can be installed using the [requirements.txt](https://github.com/mcv-m6-video/mcv-m6-2023-team1/blob/main/requirements.txt) file.
```
pip install -r requirements.txt
```
### Folder Structure
In this project you will need to download the [AI City Challenge 22](https://www.aicitychallenge.org/2022-data-and-evaluation/) dataset, also the [KITTI Stereo/Optical flow 2012 dataset](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_stereo_flow.zip) and organize the folders as follows:
```        
├── mcv-m6-2023-team1
   ├── dataset
       ├── AICity_data
          ├── train
             ├── S03
       ├── aic19-track1-mtmc-train
          ├── train
             ├── S01
             ├── S03
             ├── S04
       ├── data_stereo_flow
           ├── testing
           ├── training
       ├── results_opticalflow_kitti
           ├── results
   ├── src  
   ├── week1
      ...
   ├── week5
```
But you have to take into account that you can change the path of the dataset in the .yaml files provided for each task.

## Week 1
On the first week, the goal is to implement and understand the evaluation metrics that will be used in the next weeks. The tasks performed during this week are the following:
- Task 1: Detection metrics.
    - mIoU (Mean Intersection Over Union).
    - mAP (Mean Average Precision).
- Task 2: Temporal analysis using detection metrics.
- Task 3: Optical flow evaluation metrics.
    - MSEN (Mean Square Error in Non-occluded areas).
    - PEPN (Percentage of Erroneous Pixels in Non-occluded areas).
- Task 4: Visual representation of the optical flow.

The source code and the report of this week can be found in the following links:
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week1)
- [Presentation](https://docs.google.com/presentation/d/1tblLMqS2rwEGRwNK6NIIsIg-r4ZXQbJeovDwc3PPRDc/edit?usp=sharing)

In order to generate the results from the different task one can simply run the corresponding scripts of the task.
Take into account that some script have configs where you can decide on some options and task 1 and 2 use the same `task1.yaml`. 
For example in order to run task 1 from the /week1 folder run :
```
python task1.py 
```

## Week 2
On the second week, the main goal is to implement different background estimation models from scratch and compare them with more complex models from state-of-the-art. The tasks performed during this week are the following:
- Task 1: Implement a Gaussian model and evaluate it.
- Task 2: Implement an adaptive Gaussian model and compare it with non-adaptive model.
- Task 3: Compare the Gaussian models with state-of-the-art models.
- Task 4: Implement the Gaussian model in different color spaces.

The source code and the report of this week can be found in the following links:
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week2)
- [Presentation](https://docs.google.com/presentation/d/1Vzk87VFi-S48UVvC9IeGrX8msQPG8p626TiSDeO3Lig/edit?usp=sharing)

In order to generate the results from the different task one can simply run the corresponding scripts of the task.
Take into account that some script have configs where you can decide on some options. The config files can be found in /week2/configs/taskx.yaml, 
where "x" is the task number.
For example in order to run task 1 from the /week2 folder run :
```
python task1.py 
```
The config files for the task 1 and 2, have the following parameters that can be adjusted:
- estimator: "single_gaussian" or "adaptive_single_gaussian"
- alpha: float number
- rho: float number

In the task 3 config file can be adjusted only the "estimator" parameter, that can be: "MOG2", "KNN", "LSBP", "GSOC", or "CNT". The finality of this task is to evaluate each single gaussian background substractor method, without training it.

Finally, in the task 4 config file can be fine tuned the following parameters:
- estimator: "single_gaussian", "adaptive_single_gaussian", or "multi_gaussian"
- method: "adaptative" or "non-adaptative"
- color_space: "RGB", "HSV", "YUV", "XYZ"

## Week 3
On the third week, the main goal is to implement and train a model for object detection, and test different algorithms for object tracking.The tasks performed during this week are the following:
- Task 1: Object Detection.
    - Test different Off-the-shelf DL networks for object detection.
    - Test the different available tools used to perform the ground truth annotations.
    - Fine-tune one of the off-the-shelf DL network to our data.
    - Test K-Fold Cross-validation.
- Task 2: Object Tracking.
    - Implement a tracking algorithm by Overlap.
    - Implement a tracking algorithm with Kalman Filter.
    - Evaluate the trackers with IDF1 and HOTA scores.

The source code and the report of this week can be found in the following links:
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week3)
- [Presentation](https://docs.google.com/presentation/d/1bTaPiW5-V4t5nyi4mDJ3oiAD_aPqXAO3lbOn-PdhqBw/edit#slide=id.g2238731dbee_0_53)

The dataset created in 1.2. can be found in YOLO format (a txt for each frame of the sequence) in the folder week3/S05_c010_ownGT/*
In the same folder a yaml can be found, which contains the information of the own GT. 
The yaml file can be opened with the python package yaml, and consist of a List (for each frame) of Lists (for each object in the frame) of the different bboxes.
Moreover, a visual GIF of the GT over the sequence can be found in the slides.

Task 1 can be simply used by running from root:
```
python3 week3/task1.py 
```
Task 2 can be ran the same way but the week3/configs/task2.yaml has to be modified correspondingly in order to use it 
for visualization of results, or saving the tracking results in MOT format. 

In order to evaluate the tracking results with the TrackEval , first clone the TrackEval repository locally, then copy
data folder in the week3 directory in TrackEval and then run:
```
python3 TrackEval/scripts/run_mot_challenge.py --BENCHMARK week3 --TRACKERS_TO_EVAL overlap kalman --METRICS HOTA Identity --DO_PREPROC False
```


## Week 4
On the week 4, the main goal is to estimate the optical flow of a video sequence and try to improve an object tracking algorithm using the optical flow.The tasks performed during this week are the following:
- Task 1: Optical Flow.
    - Estimate the Optical Flow with block matching.
    - Estimate the Optical Flow with off-the-shelf method.
    - Improve the object tracking algorithm with Optical Flow.
- Task 2: Multi-Target Single-Camera tracking (MTSC).
    - Evaluate our best tracking algorithm in SEQ3 of AI City Challenge.
    - Evaluate the tracking using IDF1 and HOTA scores.

The source code and the report of this week can be found in the following links:
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week4)
- [Presentation - task1](https://docs.google.com/presentation/d/1FTtwSulFm87SZkPYsDEbVqKK0ixPBFlQ0KzYLtEYOio/edit)
- [Presentation - task2](https://docs.google.com/presentation/d/1i7jyIbeC1bf1TXjsiLCqS8t2q83O2DqegVrfY0NBrIU/edit)


## Week 5
On the week 5, the main goal is to put everything from previous weeks together and perform the Multi-target multi-camera tracking (MTMC). The source code (with the instructions of how to run the code), the final presentation, and the report of this week can be found in the following links:
- [Source Code with instructions](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week5)
- [Presentation](https://docs.google.com/presentation/d/1aG9oz3pBwZF4IP2PRvAQEiDEt2cDAEpn3adElMmn16w/edit?usp=sharing)
- [Report](https://www.overleaf.com/read/wvddxwynvhzz)
