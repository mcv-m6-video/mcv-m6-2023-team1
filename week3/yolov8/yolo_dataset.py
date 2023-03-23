from src.in_out import extract_rectangles_from_xml_yolo
from tqdm import tqdm
import os


# Convert Coco bb to Yolo
def coco_to_yolo(x1, y1, w, h, image_w, image_h):
    return [((2*x1 + w)/(2*image_w)) , ((2*y1 + h)/(2*image_h)), w/image_w, h/image_h]


def generate_rectangles_yolo(rectangles_dict):
    dataset_path = "../../dataset/yolo/yolo_dataset"
    os.makedirs(dataset_path, exist_ok=True)
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


rectangles = extract_rectangles_from_xml_yolo("../../dataset/ai_challenge_s03_c010-full_annotation.xml")
generate_rectangles_yolo(rectangles)
