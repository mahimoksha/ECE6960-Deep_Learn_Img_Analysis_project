import torch
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import torchvision
import os
import cv2
import torch
import torchvision.transforms as transforms
from PIL import Image
import argparse
import time 

use_cuda = torch.cuda.is_available()
if use_cuda:
    device="cuda:0"
else:
    device="cpu"

parser = argparse.ArgumentParser(description='Short sample app')
parser.add_argument('-model'   ,type=str  , action="store", dest='model'     , default= "resnet"   )
args = parser.parse_args()
if args.model == "mobile_net":
    backbone = torchvision.models.mobilenet_v2(weights="DEFAULT").features
    backbone.out_channels = 1280
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
    model.load_state_dict(torch.load("results/best_model_mobilenetv2.torch"))
    results_dir = "test_results_mobilenetv2_final"
elif args.model == "resnet":
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights="DEFAULT")
    num_classes = 10 
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.cuda()
    model.load_state_dict(torch.load("results/best_model_resnet50.torch"))
    results_dir = "test_results_resnet50_final"
elif args.model == "vgg":
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
    model.load_state_dict(torch.load("results/best_model_vgg19_my.torch"))
    results_dir = "test_results_vgg19_final"
def collate_fn(batch):
    return tuple(zip(*batch))
class PennFudanDataset_test(torch.utils.data.Dataset):
    def __init__(self, root_image_dir , transform = None):
        test_dir = os.listdir(root_image_dir)
        images = []
        for i in test_dir:
            image_names = os.listdir(root_image_dir+i)
            for j in image_names:
                images.append(root_image_dir+i+"/"+j)
        self.all_images = images
        self.transform = transform
    def __getitem__(self, idx):
        # load images and masks
        img_path = os.path.join(self.all_images[idx])
        fname = img_path
        # print(len(self.all_images[idx]))
        img = Image.open(img_path).convert("RGB")
        # print(img.shape)
        image_id = torch.tensor([idx])
        if self.transform:
            img = self.transform(img)
        return img, fname

    def __len__(self):
        return len(self.all_images)
test_dir = "bdd100k/images/seg_track_20/test/"
test_transforms = transforms.Compose([
transforms.ToTensor()
    ])
test = PennFudanDataset_test(test_dir,transform = test_transforms)
test_dataloader = torch.utils.data.DataLoader(test, batch_size=1)#, collate_fn=collate_fn, pin_memory=True, num_workers=1
import cv2
final_class_names = {1: "pedestrian" , 2:"rider" , 3:"car" , 4:"truck" , 5:"bus", 6:"train", 7:"motorcycle", 8:"bicycle", 9:"traffic light", 10:"traffic sign"}
model.eval()
metric = MeanAveragePrecision()
count_images = 1

if not os.path.exists(os.path.join(results_dir)):
    os.makedirs(os.path.join(results_dir))
times = []
with torch.no_grad():
    for image, filename in test_dataloader:

        st = time.time()
        image = image.to(device)
        target = model(image)
        image = image.cpu().detach().squeeze().permute(1,2,0).numpy()
        image = cv2.imread(filename[0])
        for cat in target:
            count = torch.sum(cat['scores']>0.75).item()
            cat['scores'] = cat['scores'][:count]
            cat['labels'] = cat['labels'][:count]
            cat['boxes'] = cat['boxes'][:count]

        target[0]['boxes'] = target[0]['boxes']
        for aa in target:
            labels = aa['labels']
            bb = aa["boxes"]
            classes = []
            for i in labels:
                classes.append(final_class_names[i.item()])
            for bn in range(len(bb)):
                y = bb[bn][1].item() - 10 if bb[bn][1].item() - 10 > 10 else bb[bn][1].item() + 10
                startX = bb[bn][0].item()
                cv2.putText(image,classes[bn],(int(startX), int(y)),cv2.FONT_HERSHEY_SIMPLEX,0.65,(0, 255, 0),2)
                cv2.rectangle(image, (int(bb[bn][0].item()),int(bb[bn][1].item())), (int(bb[bn][2].item()),int(bb[bn][3].item())), (0, 0, 255), 2)
        cv2.imwrite(os.path.join(results_dir+"/",str(count_images)+".png"), image)
        count_images+=1
        if count_images==200:
            break
        et = time.time()

        print("total time : ", et-st)
        times.append(et-st)