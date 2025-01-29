from ultralytics import YOLO
import cv2

model = YOLO('yolov8l.pt')

video_path = "examples/road.mp4"
cap = cv2.VideoCapture(video_path)

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
output = cv2.VideoWriter("results/annotated_road.mp4",
                         cv2.VideoWriter_fourcc(*'mp4v'),
                         30, (frame_width, frame_height))

results = model.predict(source="road.mp4", save=True)
