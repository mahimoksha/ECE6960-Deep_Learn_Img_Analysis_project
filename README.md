# ECE6960-Deep_Learn_Img_Analysis_project

### Preprocessing
For running the model we have removed some of the images base on the following criteria:
1. If bounding box dimensions are negative.
2. If there are no bounding boxes available.
3. coordinates of bounding boxes are same.

Make sure all libraries are installed.

To run different Ablation studies:

1. Resent 50 : sh run_resnet.sh
2. VGG 10 : sh run_vgg.sh
3. MobileNet v2 : sh run_mobilenet.sh

After running the models the best model will be saved in results folder then run the following script file to get the results on test images:
1. sh run_inference.sh resnet
2. sh run_inference.sh vgg
3. sh run_inference.sh mobile_net


### YOLOv5

The YOLOv5 training is inspired from [this repo](https://github.com/williamhyin/yolov5s_bdd100k)

To run the YOLO models for training and inference, see yolov5/yolov5.ipynb
