# ECE6960-Deep_Learn_Img_Analysis_project

###PreProcessing
For running the model we have removed some of the images base on the following criteria:
1. If bounding box dimensions are negative.
2. If there are no bounding boxes available.
3. coordinates of bounding boxes are same.

To run different Ablation studies:

1. Resent 50 : sh run_resnet.sh
2. VGG 10 : sh run_vgg.sh
3. MobileNet v2 : sh run_mobilenet.sh

After running the models the best model will be saved in results folder then run the following script file to get the results on test images:
sh run_inference.sh
