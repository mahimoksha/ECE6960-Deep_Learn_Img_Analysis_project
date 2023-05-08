import json
import pandas as pd
import numpy as np
import os 
import matplotlib.pyplot as plt
import cv2
from Dataloader import PennFudanDataset
from torchvision.models import resnet50, ResNet50_Weights, vgg19
import torchvision
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

import torchvision.transforms as transforms
import os
import numpy as np
import torch
from PIL import Image
import json 
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


def collate_fn(batch):
    return tuple(zip(*batch))


def validation(model, dataloader,device):
    torch.cuda.empty_cache()
    model.eval()
    metric = MeanAveragePrecision()
    with torch.no_grad():
        for image, target in dataloader:
            images = list(i.to(device) for i in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in target]
            outputs = model(images)
            metric.update(outputs, targets)
    result = metric.compute()
    return result['map']
def train_fn(train_dataloader, val_dataloader, device):
    torch.cuda.empty_cache()
    backbone = torchvision.models.vgg19(weights="DEFAULT").features
    backbone.out_channels = 512
    anchor_generator = AnchorGenerator(sizes=((64, 128, 512),),
                                    aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
                                                    output_size=5,
                                                    sampling_ratio=2)
    model = FasterRCNN(backbone,
                    num_classes=10,
                    rpn_anchor_generator=anchor_generator,
                    box_roi_pool=roi_pooler)
    model.to(device)
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=3,gamma=0.1)
    epochs = 25
    total_train_loss = []
    total_classifier_loss = []
    total_box_reg_loss = []
    total_val_loss = []
    total_val_classifier_loss = []
    total_val_box_reg_loss = []
    total_val_map_metric = []
    total_train_map_metric = []
    val_max_loss = np.inf
    val_min_map_metric = 0
    best_epoch = 0
    a = 0
    if not os.path.exists(os.path.join("results")):
        os.makedirs(os.path.join("results"))
    for epoch in tqdm(range(epochs)):
        model.train()
        train_loss = []
        classifier_loss = []
        box_reg_loss = []
        for image, target in train_dataloader:
            optimizer.zero_grad()
            images = list(i.to(device) for i in image)
            targets = [{k: v.to(device) for k, v in t.items()} for t in target]
            outputs = model(images, targets)
            losses = sum(loss for loss in outputs.values())
            losses.backward()
            optimizer.step()
            train_loss.append(losses.item())
            classifier_loss.append(outputs['loss_classifier'].item())
            box_reg_loss.append(outputs['loss_box_reg'].item())
        if a%3 == 0:
            train_map_metric_epoch = validation(model, train_dataloader,device)
            total_train_map_metric.append(train_map_metric_epoch)
        val_map_metric = validation(model, val_dataloader,device)
        lr_scheduler.step()
        total_train_loss.append(np.mean(train_loss))
        total_classifier_loss.append(np.mean(classifier_loss))
        total_box_reg_loss.append(np.mean(box_reg_loss))
        total_val_map_metric.append(val_map_metric)

        if val_map_metric > val_min_map_metric:
            val_min_map_metric = val_map_metric	
            best_epoch = epoch
            print("Improvement-Detected")
            torch.save(model.state_dict(), os.path.join("results",'best_model_vgg19_my.torch'))
        print("train loss {}, classifier loss {}, box reg loss {}".format(np.mean(train_loss),np.mean(classifier_loss),np.mean(box_reg_loss)))
        print("Train metric {}".format(train_map_metric_epoch))
        print("val metric {}".format(val_map_metric))
        a +=1
    print("Training Done Best Epoch is at ", best_epoch)
    fig, ax = plt.subplots()
    fig.set_size_inches(20,16)
    ax.plot(total_train_loss, label="train")
    # ax.plot(val_loss, label="validation")
    ax.legend()
    fig.savefig("train_val_loss_vgg19_my.png",transparent=True,bbox_inches='tight')  

    fig, ax = plt.subplots()
    fig.set_size_inches(20,16)
    ax.plot(total_classifier_loss, label="train classifier")
    # ax.plot(val_classifier_loss, label="validation classifier")
    ax.legend()
    fig.savefig("train_val_classifier_loss_vgg19_my.png",transparent=True,bbox_inches='tight')  

    fig, ax = plt.subplots()
    fig.set_size_inches(20,16)
    ax.plot(total_box_reg_loss, label="train regression")
    # ax.plot(val_box_reg_loss, label="validation regression")
    ax.legend()
    fig.savefig("train_val_regression_loss_vgg19_my.png",transparent=True,bbox_inches='tight')  



    fig, ax = plt.subplots()
    fig.set_size_inches(20,16)
    ax.plot(total_train_map_metric, label="train MAP")
    ax.plot(total_val_map_metric, label="validation MAP")
    ax.legend()
    fig.savefig("train_val_map_metric_loss_vgg19_my.png",transparent=True,bbox_inches='tight')  

    model.load_state_dict(torch.load(os.path.join("results",'best_model_vgg19_my.torch')))
    train_map_metric = validation(model, train_dataloader,device)
    validation_map_metric = validation(model, val_dataloader,device)
    print("VGG 19 MAP Train: ", train_map_metric)
    print("VGG 19 MAP Val: ", validation_map_metric)
if __name__=="__main__":
        
    train_dir = "bdd100k/labels/seg_track_20/rles/"

    img_dir = "bdd100k/images/seg_track_20/" 
    train_transforms = transforms.Compose([
    transforms.ToTensor()
        ])
    removed_dir = "zero_bound_box/"
    sample = 1

    train = PennFudanDataset(train_dir, img_dir,"train", removed_dir,transform = train_transforms, sample = sample)
    train_dataloader = torch.utils.data.DataLoader(train, batch_size=8, shuffle = True, collate_fn=collate_fn, pin_memory=True, num_workers=8)

    removed_dir = "val_zero_bounding_box/"

    val = PennFudanDataset(train_dir, img_dir,"val", removed_dir, transform = train_transforms)
    val_dataloader = torch.utils.data.DataLoader(val, batch_size=2, shuffle=True, collate_fn=collate_fn, pin_memory=True, num_workers=2)
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        device="cuda:0"
    else:
        device="cpu"

    train_fn(train_dataloader, val_dataloader, device)
