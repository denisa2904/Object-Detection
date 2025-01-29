from ultralytics import YOLO

if __name__ == '__main__':
    model = YOLO('yolov8l.pt')

    model.train(
        data='data.yaml',  
        epochs=50,
        imgsz=640,
        batch=16,
        workers=8
    )
