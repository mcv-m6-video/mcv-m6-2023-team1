from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="../../dataset/yolo/splits/yolo_Fold0.yaml", epochs=20, device=0, imgsz=640, workers=2, val=False)
    metrics = model.val()


if __name__ == "__main__":
    main()
