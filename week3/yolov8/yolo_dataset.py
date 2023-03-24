from src.in_out import extract_rectangles_from_xml_detection, extract_frames_from_video
from tqdm import tqdm
import glob
import os
from sklearn.model_selection import KFold
import shutil


# Convert Coco bb to Yolo
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]


def generate_rectangles_yolo(rectangles_dict, dataset_path):
    voc_to_yolo_labels = {
        "car": 0,
        "bike": 1
    }
    img_width = 1920
    img_height = 1080
    for frame, rectangles in tqdm(rectangles_dict.items()):
        with open(f'{dataset_path}/frame_{str(frame).zfill(4)}.txt', 'w') as f:
            for rectangle in rectangles:
                label = rectangle[1]
                label_id = voc_to_yolo_labels[label]
                x1 = rectangle[2]
                y1 = rectangle[3]
                x2 = rectangle[4]
                y2 = rectangle[5]
                width = x2 - x1
                height = y2 - y1
                x_center, y_center, width, height = coco_to_yolo(x1, y1, width, height, img_width, img_height)

                f.write(f"{label_id} {x_center} {y_center} {width} {height}")
                f.write("\n")


def generate_yolo_dataset(dataset_path):
    os.makedirs(dataset_path, exist_ok=True)
    # extract frames
    extract_frames_from_video("../../dataset/AICity_data/train/S03/c010/vdo.avi", output_path=dataset_path)
    # extract rectangles
    rectangles = extract_rectangles_from_xml_detection("../../dataset/ai_challenge_s03_c010-full_annotation.xml")
    generate_rectangles_yolo(rectangles, dataset_path)


def generate_k_folds(dataset_path, splits_path, random_fold=False):
    os.makedirs(f"{splits_path}", exist_ok=True)

    # get image paths and sort them
    img_paths = glob.glob(f"{dataset_path}/*.jpg")
    img_paths = [path.replace("\\", "/") for path in img_paths]
    img_paths = sorted(img_paths, key=lambda path: int(path.split("_")[-1].split(".")[0]))
    # get annotations paths and sort them
    anns_paths = glob.glob(f"{dataset_path}/*.txt")
    anns_paths = [path.replace("\\", "/") for path in anns_paths]
    anns_paths = sorted(anns_paths, key=lambda path: int(path.split("_")[-1].split(".")[0]))

    indices = list(range(len(img_paths)))

    # Set the number of folds for K-fold cross-validation
    k = 4
    kf = KFold(n_splits=k, shuffle=random_fold, random_state=42)

    for i, (val_indices, train_indices) in enumerate(kf.split(indices)):
        train_img_paths = [img_paths[j] for j in train_indices]
        train_anns_paths = [anns_paths[j] for j in train_indices]

        val_img_paths = [img_paths[j] for j in val_indices]
        val_anns_paths = [anns_paths[j] for j in val_indices]

        if random_fold:
            fold_path = f"{splits_path}/randomFold{i}"
            os.makedirs(fold_path, exist_ok=True)
        else:
            fold_path = f"{splits_path}/Fold{i}"
            os.makedirs(fold_path, exist_ok=True)

        os.makedirs(f"{fold_path}/train", exist_ok=True)
        os.makedirs(f"{fold_path}/val", exist_ok=True)

        for img_path, anns_path in zip(train_img_paths, train_anns_paths):
            img_filename = img_path.split("/")[-1]
            anns_filename = anns_path.split("/")[-1]
            shutil.copy(img_path, f"{fold_path}/train/{img_filename}")
            shutil.copy(anns_path, f"{fold_path}/train/{anns_filename}")

        for img_path, anns_path in zip(val_img_paths, val_anns_paths):
            img_filename = img_path.split("/")[-1]
            anns_filename = anns_path.split("/")[-1]
            shutil.copy(img_path, f"{fold_path}/val/{img_filename}")
            shutil.copy(anns_path, f"{fold_path}/val/{anns_filename}")


if __name__ == "__main__":
    dataset_path = "../../dataset/yolo/yolo_dataset"
    splits_path = "../../dataset/yolo/splits"
    generate_yolo_dataset(dataset_path)
    generate_k_folds(dataset_path, splits_path, random_fold=False)
