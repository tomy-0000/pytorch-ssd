import glob
import json
import re
import os
import numpy as np
import cv2


class COCODataset:

    def __init__(self, root, transform=None, target_transform=None, is_test=False, keep_difficult=False, label_file=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        if is_test:
            image_sets_file = os.path.join(self.root, "images/val2017")
            annotation_file = os.path.join(self.root, "annotations/val2017.json")
        else:
            image_sets_file = os.path.join(self.root, "images/train2017")
            annotation_file = os.path.join(self.root, "annotations/train2017.json")
        self.ids = COCODataset._read_image_ids(image_sets_file)
        with open(annotation_file) as f:
            annotation = json.load(f)
        self.annotation = annotation
        self.keep_difficult = keep_difficult

        self.class_names = ("BACKGROUND", "person")

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __getitem__(self, index):
        image_file = self.ids[index]
        image_id = re.findall("0*(\d+).jpg", image_file.split("/")[-1])[0]
        boxes, labels, is_difficult = self._get_annotation(image_id)
        if not self.keep_difficult:
            boxes = boxes[is_difficult == 0]
            labels = labels[is_difficult == 0]
        image = self._read_image(image_file)
        if self.transform:
            image, boxes, labels = self.transform(image, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)
        return image, boxes, labels

    def get_image(self, index):
        image_file = self.ids[index]
        image = self._read_image(image_file)
        if self.transform:
            image, _ = self.transform(image)
        return image

    def get_annotation(self, index):
        image_file = self.ids[index]
        image_id = re.findall("0*(\d+).jpg", image_file.split("/")[-1])[0]
        return image_id, self._get_annotation(image_id)

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def _read_image_ids(image_sets_file):
        ids = glob.glob(os.path.join(image_sets_file, "*"))
        return ids

    def _get_annotation(self, image_id):
        objects = self.annotation[image_id]
        boxes = []
        labels = []
        is_difficult = []
        for box in objects["bbox"]:
            x1 = box[0]
            y1 = box[1]
            x2 = x1 + box[2]
            y2 = y1 + box[3]
            boxes.append([x1, y1, x2, y2])

            labels.append(self.class_dict["person"])
            is_difficult.append(objects["iscrowd"])
        return (np.array(boxes, dtype=np.float32),
                np.array(labels, dtype=np.int64),
                np.array(is_difficult, dtype=np.uint8))

    def _read_image(self, image_file):
        image = cv2.imread(image_file)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image
