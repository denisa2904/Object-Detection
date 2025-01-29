import cv2
import os

boxes_folder = "boxes"


def select_vehicle():
    first_frame_path = os.path.join(boxes_folder, sorted(os.listdir(boxes_folder))[0])
    first_frame = cv2.imread(first_frame_path)
    cv2.imshow("Select Vehicle", first_frame)
    selected_bbox = cv2.selectROI("Select Vehicle", first_frame, False)
    vehicle_template = first_frame[int(selected_bbox[1]):int(selected_bbox[1] + selected_bbox[3]),
                       int(selected_bbox[0]):int(selected_bbox[0] + selected_bbox[2])]
    cv2.destroyAllWindows()
    return vehicle_template


def track_vehicle(vehicle_template):
    entering_frame = None
    exiting_frame = None

    for idx, img_name in enumerate(sorted(os.listdir(boxes_folder))):
        img_path = os.path.join(boxes_folder, img_name)
        frame = cv2.imread(img_path)

        result = cv2.matchTemplate(frame, vehicle_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val >= 0.8:
            top_left = max_loc
            bottom_right = (top_left[0] + vehicle_template.shape[1], top_left[1] + vehicle_template.shape[0])

            cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)
            cv2.putText(frame, f"Match {max_val:.2f}", (top_left[0], top_left[1] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            if entering_frame is None:
                entering_frame = img_name
            exiting_frame = img_name

        else:
            exiting_frame = img_name
            break
        cv2.imshow("Tracking", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    if entering_frame and exiting_frame:
        print(f"Vehicle entered at frame: {entering_frame}")
        print(f"Vehicle exited at frame: {exiting_frame}")
    else:
        print("Vehicle not found.")

    cv2.destroyAllWindows()


def main():
    vehicle_template = select_vehicle()
    track_vehicle(vehicle_template)


if __name__ == "__main__":
    main()

