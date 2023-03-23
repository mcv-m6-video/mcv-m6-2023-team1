from ultralytics import YOLO


def main():
    model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

    # Use the model
    model.train(data="../../dataset/yolo/yolo_dataset.yaml", epochs=3, device=0, imgsz=640, workers=0)  # train the model


if __name__ == "__main__":
    main()
