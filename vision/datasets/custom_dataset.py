import glob
import random
from PIL import Image
import os
import numpy as np
import cv2
import torch
import torchvision.transforms as transforms

class CustomDataset:

    def __init__(self, root, end, transform=None, target_transform=None, is_test=False):
        self.root = root
        self.cnt = 0
        self.end = end

        self.transform1 = transforms.Compose([
            transforms.RandomAffine(180, shear=30),
            transforms.RandomPerspective(0.7),
        ])
        self.transform2 = transforms.Compose([
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
        ])
        self.transform3 = transforms.Compose([
            transforms.RandomPosterize(3),
            transforms.RandomAdjustSharpness(6),
        ])
        self.transform = transform
        self.target_transform = target_transform

        if is_test:
            image_sets_file = os.path.join(self.root, "test")
        else:
            image_sets_file = os.path.join(self.root, "train")
        self.ids_hand, self.ids_background = CustomDataset._read_image_ids(image_sets_file)

        self.class_names = ("BACKGROUND", "GUU", "PAA")

        self.class_dict = {class_name: i for i, class_name in enumerate(self.class_names)}

    def __iter__(self):
        return self

    def __next__(self):
        if self.cnt >= self.end:
            self.cnt = 0
            raise StopIteration()
        background_path = torch.randint(0, len(self.ids_background))
        background = Image.open(background_path)
        background_w, background_h = background.size
        num = random.randint(1, 5)
        img_path_chosen = random.choices(self.ids_hand, k=num)
        boxes = []
        labels = []
        for img_path in img_path_chosen:
            img = Image.open(img_path)
            if "GUU" in img_path:
                label = self.class_dict["GUU"]
            else:
                label = self.class_dict["PAA"]
            transparent = Image.new("RGBA", (2000, 2000), (0, 0, 0, 0))
            img_w, img_h = img.size
            transparent.paste(img, (1000 - img_w//2, 1000 - img_h//2), mask=img)
            img = self.transform1(transparent)
            img_arr = np.array(img)
            img_sum = img_arr.sum(axis=-1)
            x_idx = np.where(0 < img_sum.sum(axis=1))[0]
            y_idx = np.where(0 < img_sum.sum(axis=0))[0]
            x1 = min(x_idx)
            x2 = max(x_idx)
            y1 = min(y_idx)
            y2 = max(y_idx)
            img_arr = img_arr[x1:x2 + 1, y1:y2 + 1]
            img = Image.fromarray(img_arr)
            img = self.transform2(img)
            img2 = self.transform3(img.convert("RGB"))
            img.paste(img2, mask=img)
            img_w, img_h = img.size
            rand_w = random.randint(0, background_w - img_w)
            rand_h = random.randint(0, background_h - img_h)
            background.paste(img, (rand_w, rand_h), mask=img)
            boxes.append(rand_w, rand_h, rand_w + img_w, rand_h + img_h)
            labels.append(label)
        boxes = np.array(boxes, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)
        img = self.pil2cv(background)
        if self.transform:
            img, boxes, labels = self.transform(img, boxes, labels)
        if self.target_transform:
            boxes, labels = self.target_transform(boxes, labels)

        return img, boxes, labels

    def pil2cv(self, image):
        ''' PIL型 -> OpenCV型 '''
        new_image = np.array(image, dtype=np.uint8)
        if new_image.ndim == 2:  # モノクロ
            pass
        elif new_image.shape[2] == 3:  # カラー
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGB2BGR)
        elif new_image.shape[2] == 4:  # 透過
            new_image = cv2.cvtColor(new_image, cv2.COLOR_RGBA2BGRA)
        return new_image

    def _read_image_ids(self, image_sets_file):
        ids_hand = glob.glob(os.path.join(image_sets_file, "*.PNG"))
        ids_background = glob.glob(os.path.join(self.root, "background", "*"))
        return ids_hand, ids_background
