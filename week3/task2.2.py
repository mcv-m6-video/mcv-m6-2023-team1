import sys, os, cv2, time, argparse
sys.path.append(os.path.join(os.path.realpath(os.path.dirname(__file__)), "../")) # comment if you are not using Visual Studio Code
from typing import Dict
import numpy as np
from tqdm import tqdm
from skimage import io

from src.in_out import extract_frames_from_video, get_frames_paths_from_folder, extract_rectangles_from_xml
from src.utils import open_config_yaml, load_bboxes_from_file, draw_img_with_ids, draw_bboxes_trajectory

#Import Sort for tracking using Kalman filter
from sort import Sort


def task2_2(cfg: Dict):
    """
    For this task, you will implement a background estimation algorithm using a single Gaussian model.
    """
    paths = cfg["paths"]

    #create instance of SORT
    mot_tracker = Sort(max_age=1, 
                       min_hits=3,
                       iou_threshold=0.3) #create instance of the SORT tracker

    # get detections
    gt_labels = extract_rectangles_from_xml(paths["annotations"])
    if cfg['model']['use_gt']:
        gt_bboxes = [*gt_labels.values()]
        bboxes = gt_bboxes[int(len(dataset)*0.25):]
    else:
        bboxes = load_bboxes_from_file(paths['detected_bboxes'])

    track_bbs_ids = []
    # update SORT
    for frame_bboxes in bboxes:  
        # np array where each row contains a valid bounding box and track_id (last column)  
        track_bbs_ids.append(mot_tracker.update(np.array(frame_bboxes)))


    #Obtain all frames of the sequence
    extract_frames_from_video(video_path=paths["video"], output_path=paths["extracted_frames"])
    frames = get_frames_paths_from_folder(input_path=paths["extracted_frames"])    
    dataset = [(key, frames[key])for key in gt_labels.keys()]

    #keep only frames of selected range
    #dataset = dataset[first_frame:last_frame]
    dataset = dataset[int(len(dataset)*0.25):]
    print("Number of frames: ", len(dataset))

    #Draw the bounding boxes and the trajectory
    overlay = np.zeros_like(cv2.imread(dataset[0][1]))
    for i, frame in tqdm(enumerate(dataset)):
        if i > cfg['model']['first_frame'] and i < cfg['model']['last_frame']:
            img = cv2.imread(frame[1])
            #Draw the trajectory lines
            overlay = draw_bboxes_trajectory(overlay, track_bbs_ids[i], track_bbs_ids[i-1])
            #Draw the bounding boxes with ID number
            img_out = draw_img_with_ids(img, track_bbs_ids[i])
            #Fuse both images
            img_out = cv2.addWeighted(img_out,1,overlay,1,0)

            #Show the frame with the trajectory and bounding boxes IDs
            if cfg['visualization']['show_detection']:
                cv2.imshow('frame2', img_out)
                k = cv2.waitKey(30) & 0xff
                if k == 27:
                    break
            
            #Save the frame
            if cfg['visualization']['save']:
                os.makedirs(paths['output'], exist_ok=True)
                cv2.imwrite(os.path.join(paths['output'], str(i)+'.png'), img_out, [int(cv2.IMWRITE_PNG_COMPRESSION), 9])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    #parser.add_argument("--config", default="configs/task1.yaml")
    parser.add_argument("--config", default="week3/configs/task2_2.yaml") #comment if you are not using Visual Studio Code
    args = parser.parse_args(sys.argv[1:])

    config = open_config_yaml(args.config)

    task2_2(config)