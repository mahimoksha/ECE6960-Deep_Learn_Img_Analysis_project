import os
import numpy as np
import torch
from PIL import Image
import json 
import cv2
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
# class PennFudanDataset(torch.utils.data.Dataset):
#     def __init__(self, root_dir, root_image_dir, type, transform = None):
#         self.json_root_dir = root_dir + type + "/"
#         self.img_dir = root_image_dir + type +"/"
        
#         json_files = os.listdir(self.json_root_dir)
#         image_file = []
#         c = 0
#         all_frames = []
#         all_classes = []
#         for i in json_files:
#             data = json.load(open(self.json_root_dir+i))
#             for frame in data['frames']:
#                 single_frame = []
#                 single_class = []
#                 for j in frame['labels']:
#                     single_frame.append(j['box2d'])
#                     single_class.append(j['category'])
#                 all_frames.append(single_frame)
#                 all_classes.append(single_class)
#                 image_file.append(self.img_dir+frame['videoName']+"/"+frame["name"])
#         self.image_files = image_file
#         self.bounding_box = all_frames
#         self.classes = all_classes
#         self.transform = transform
#         self.class_labels = {"pedestrian":1 , "rider":2 , "car":3 , "truck":4 , "bus":5, "train":6, "motorcycle":7, "bicycle":8, "traffic light":9, "traffic sign":10}
#     def __getitem__(self, idx):
#         # load images and masks
#         img_path = os.path.join(self.image_files[idx])
#         bound_box = self.bounding_box[idx]
#         img = cv2.imread(img_path)
#         boxes = []
#         for i in bound_box:
#             boxes.append([i['x1'], i['y1'], i['x2'], i['y2']])
#         boxes = torch.FloatTensor(boxes)
#         clases = self.classes[idx]
#         final_classes = []
#         for i in clases:
#             final_classes.append(self.class_labels[i])
#         # print(final_classes)
#         # final_classes = torch.tensor(final_classes)
#         image_id = torch.tensor([idx])
#         final_classes = torch.Tensor(final_classes)
#         target = {}
#         target["boxes"] = boxes
#         # target["image_id"] = image_id
#         target["labels"] = final_classes
#         if self.transform:
#             img = self.transform(img)
#         return img, target

#     def __len__(self):
#         return len(self.image_files)


class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, root_image_dir, type, removed_dir,transform = None, sample = 1):
        self.json_root_dir = root_dir + type + "/"
        self.img_dir = root_image_dir + type +"/"
        
        json_files = os.listdir(self.json_root_dir)
        image_file = []
        c = 0
        all_frames = []
        all_classes = []
        removed_images = os.listdir(removed_dir)
        for i in json_files:
            data = json.load(open(self.json_root_dir+i))
            for frame in data['frames']:
                single_frame = []
                single_class = []
                for j in frame['labels']:
                    single_frame.append(j['box2d'])
                    single_class.append(j['category'])
                if frame["name"] not in removed_images:
                    all_frames.append(single_frame)
                    all_classes.append(single_class)
                    image_file.append(self.img_dir+frame['videoName']+"/"+frame["name"])
        sample_percent = int(len(image_file)*sample)
        self.image_files = image_file[:sample_percent]
        self.bounding_box = all_frames[:sample_percent]
        self.classes = all_classes[:sample_percent]
        self.transform = transform
        self.class_labels = {"pedestrian":1 , "rider":2 , "car":3 , "truck":4 , "bus":5, "train":6, "motorcycle":7, "bicycle":8, "traffic light":9, "traffic sign":10}
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.image_files[idx])
        bound_box = self.bounding_box[idx]
        img = Image.open(img_path).convert("RGB")
        boxes = []
        for i in bound_box:
            boxes.append([i['x1'], i['y1'], i['x2'], i['y2']])
        boxes = torch.FloatTensor(boxes)
        clases = self.classes[idx]
        final_classes = []
        for i in clases:
            final_classes.append(self.class_labels[i])
        # print(final_classes)
        # final_classes = torch.tensor(final_classes)
        image_id = torch.tensor([idx])
        final_classes = final_classes
        target = {}
        target["boxes"] = boxes
        # target["image_id"] = image_id
        target["labels"] = torch.LongTensor(final_classes)
        if self.transform:
            img = self.transform(img)
        target["boxes"] = torch.Tensor(target["boxes"])
        return img, target

    def __len__(self):
        return len(self.image_files)
    





if __name__ == '__main__':
    train_dir = "/home/sci/mkaranam/Desktop/DL_Image_Analysis_project_Detection/bdd100k/labels/seg_track_20/rles/"

    img_dir = "/home/sci/mkaranam/Desktop/DL_Image_Analysis_project_Detection/bdd100k/images/seg_track_20/"
    train_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    train = PennFudanDataset(train_dir, img_dir,"train", transforms = train_transforms)

    params = {'batch_size': 1,
        'shuffle': True}
    train_dataloader = torch.utils.data.DataLoader(train,**params)
    for image, target in train_dataloader:
        image = image.cpu().detach().squeeze().numpy()
        for bb in target["boxes"][0]:
            cv2.rectangle(image, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0, 0, 255), 2)
        plt.imshow(image)
        break