import os
import shutil
import random

base_dir = "train_dataset"
images_dir = os.path.join(base_dir, "images")
labels_dir = os.path.join(base_dir, "labels")

train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")

for split in ["train", "test"]:
    os.makedirs(os.path.join(train_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(train_dir, "labels"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "images"), exist_ok=True)
    os.makedirs(os.path.join(test_dir, "labels"), exist_ok=True)

image_files = [f for f in os.listdir(images_dir) if f.endswith(".jpg") or f.endswith(".png")]

random.shuffle(image_files)

split_index = int(len(image_files) * 0.7)
train_files = image_files[:split_index]
test_files = image_files[split_index:]


def move_files(file_list, dest_images, dest_labels):
    for file in file_list:
        src_image = os.path.join(images_dir, file)
        dst_image = os.path.join(dest_images, file)
        shutil.copy(src_image, dst_image)

        label_file = file.replace('.jpg', '.txt').replace('.png', '.txt')
        src_label = os.path.join(labels_dir, label_file)
        dst_label = os.path.join(dest_labels, label_file)

        if os.path.exists(src_label):
            shutil.copy(src_label, dst_label)


move_files(train_files, os.path.join(train_dir, "images"), os.path.join(train_dir, "labels"))
move_files(test_files, os.path.join(test_dir, "images"), os.path.join(test_dir, "labels"))

print("Dataset split complete!")
print(f"Train images: {len(train_files)}, Test images: {len(test_files)}")
