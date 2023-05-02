import os
import numpy as np
import torch
from PIL import Image
import json 
import cv2
import matplotlib.pyplot as plt

class PennFudanDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, root_image_dir, type):
        self.json_root_dir = root_dir + type + "/"
        self.img_dir = root_image_dir + type +"/"
        
        json_files = os.listdir(self.json_root_dir)
        image_file = []
        c = 0
        all_frames = []
        for i in json_files:
            data = json.load(open(self.json_root_dir+i))
            # labels = data['frames']
            for frame in data['frames']:
                single_frame = []
                for j in frame['labels']:
                    single_frame.append(j['box2d'])
                all_frames.append(single_frame)
                image_file.append(self.img_dir+frame['videoName']+"/"+frame["name"])
        self.image_files = image_file
        self.bounding_box = all_frames

    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.image_files[idx])
        bound_box = self.bounding_box[idx]
        # img_path = os.path.join(self.root, "PNGImages", self.imgs[idx])
        # mask_path = os.path.join(self.root, "PedMasks", self.masks[idx])
        # img = Image.open(img_path).convert("RGB")
        img = cv2.imread(img_path)
        boxes = []
        for i in bound_box:
            boxes.append([i['x1'], i['y1'], i['x2'], i['y2']])
        boxes = torch.FloatTensor(boxes)
        image_id = torch.tensor([idx])
        target = {}
        target["boxes"] = boxes
        target["image_id"] = image_id
        # if self.transforms is not None:
        #     img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.image_files)
    
if __name__ == '__main__':
    train_dir = "/home/sci/mkaranam/Desktop/DL_Image_Analysis_project_Detection/bdd100k/labels/seg_track_20/rles/"

    img_dir = "/home/sci/mkaranam/Desktop/DL_Image_Analysis_project_Detection/bdd100k/images/seg_track_20/"
    train = PennFudanDataset(train_dir, img_dir,"train")

    params = {'batch_size': 1,
        'shuffle': True}
    train_dataloader = torch.utils.data.DataLoader(train,**params)
    for image, target in train_dataloader:
        # image = cv2.imread(image_files[-1000])
        image = image.cpu().detach().squeeze().numpy()
        for bb in target["boxes"][0]:
            cv2.rectangle(image, (int(bb[0]),int(bb[1])), (int(bb[2]),int(bb[3])), (0, 0, 255), 2)
        plt.imshow(image)
        break