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

allowed_classes = ['car', 'truck', 'person', 'traffic light']

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, conf=0.3)
    detected_objects = []

    for result in results:
        for box in result.boxes.data:
            confidence = float(box[4])
            class_id = int(box[5])
            class_name = result.names[class_id]

            if class_name in allowed_classes:
                detected_objects.append((confidence, class_name, box[:4]))

    detected_objects.sort(reverse=True, key=lambda x: x[0])
    for obj in detected_objects[:4]:
        conf, class_name, (x1, y1, x2, y2) = obj
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(frame, f"{class_name} {conf:.2f}", (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    output.write(frame)
    cv2.imshow("Annotated Video", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
output.release()
cv2.destroyAllWindows()

