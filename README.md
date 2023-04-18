# MCV M6 - Video Analysis
# Project: Video Surveillance for Road

## Team 1 Members
- Rachid Boukir ([email](mailto:rachid.boukir@autonoma.cat))
- Josep Bravo ([email](mailto:pepbravo11@gmail.com))
- Alex Martin ([email](mailto:alex.martin@midokura.com))
- Guillem Martinez ([email](mailto:guillemmarsan@gmail.com))
- Miquel Romero ([email](mailto:miquel.robla@gmail.com))

## Introduction
 PENDING
 
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
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week1)
- [Presentation](https://docs.google.com/presentation/d/1tblLMqS2rwEGRwNK6NIIsIg-r4ZXQbJeovDwc3PPRDc/edit?usp=sharing)

In order to generate the results from the different task one can simply run the corresponding scripts of the task.
Take into account that some script have configs where you can decide on some options and task 1 and 2 use the same `task1.yaml`. 
For example in order to run task 1 from the /week1 folder run :
```
python task1.py 
```

## Week 2
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

- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week3)
- [Presentation](https://docs.google.com/presentation/d/1bTaPiW5-V4t5nyi4mDJ3oiAD_aPqXAO3lbOn-PdhqBw/edit#slide=id.g2238731dbee_0_53)
- 
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
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week4)
- [Presentation - task1](https://docs.google.com/presentation/d/1FTtwSulFm87SZkPYsDEbVqKK0ixPBFlQ0KzYLtEYOio/edit)
- [Presentation - task2](https://docs.google.com/presentation/d/1i7jyIbeC1bf1TXjsiLCqS8t2q83O2DqegVrfY0NBrIU/edit)


## Week 5
- [Source Code](https://github.com/mcv-m6-video/mcv-m6-2023-team1/tree/main/week5)
- [Presentation](https://docs.google.com/presentation/d/1aG9oz3pBwZF4IP2PRvAQEiDEt2cDAEpn3adElMmn16w/edit?usp=sharing)
- [Report](https://www.overleaf.com/read/wvddxwynvhzz)