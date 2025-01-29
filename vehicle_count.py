from ultralytics import YOLO
import os
import cv2

image_folder = "images"
output_folder = "output_from_trained"
os.makedirs(output_folder, exist_ok=True)


def count_cars(model):
    frame_results = {}

    for idx, img_name in enumerate(sorted(os.listdir(image_folder))):
        img_path = os.path.join(image_folder, img_name)
        frame = cv2.imread(img_path)

        results = model(img_path, conf=0.5)

        class_counts = {}
        for result in results:
            for box in result.boxes.data:
                class_id = int(box[5])
                class_name = result.names[class_id]

                if class_name in ['cars', 'truck']:
                    class_counts[class_name] = class_counts.get(class_name, 0) + 1

                x1, y1, x2, y2 = map(int, box[:4])
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(frame, class_name, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        output_path = os.path.join(output_folder, img_name)
        cv2.imwrite(output_path, frame)

        frame_results[img_name] = class_counts
        print(f"Frame {idx + 1}: {class_counts}")

    print("\nVehicle Counts Per Frame:")
    for frame, counts in frame_results.items():
        print(f"{frame}: {counts}")


def main():
    # model = YOLO('yolov8l.pt')
    model = YOLO('train3/weights/best.pt')
    count_cars(model)


if __name__ == '__main__':
    main()
