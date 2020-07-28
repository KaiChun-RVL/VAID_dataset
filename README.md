
# VAID_dataset

## Introduction

VAID(Vehicle Aerial Imaging from Drone) dataset contains 6000 aerial images under different illumination conditions, viewing angles from different places in Taiwan.The images are taken with the resolution of 1137 * 640 pixels in JPG format. Our VAID fataset contains seven classes of Vehicles, namely 'sedan' , 'minibus' , 'truck' , 'pickup truck' , 'bus' , 'cement truck' , ' trailer'.  
From left to right in the figure below are labels 'sedan' , 'minibus' , 'truck' , 'pickup truck' , 'bus' , 'cement truck' , ' trailer'
![image](https://github.com/KaiChun-RVL/VAID_dataset/blob/master/images/class.PNG)
You can download VAID dataset in the website.https://vision.ee.ccu.edu.tw/aerialimage/




## Test model

We test VAID dataset on 5 common model including Faster R-CNN, Yolov4, MobileNetv3 , RefineDet and U-Net.

### Perfromance

### How to use 
 
#### Faster R-CNN
environment : tensorflow1.4 cuda9.2
1. Download the file
2. Put for_download/faster rcnn/data and for_download/faster rcnn/output into VAID_dataset/tf-faster-rcnn
3. Download dataset and put Annotations amd JPEGImages into VAID_dataset/tf-faster-rcnn/data/VOCdevkit2007/VOC2007
4. make at VAID_dataset/tf-faster-rcnn/lib and VAID_dataset/tf-faster-rcnn/data/coco/PythonAPI
command :
demo
GPU_ID=0
CUDA_VISIBLE_DEVICES=${GPU_ID} ./tools/demo.py --net res101 --dataset pascal_voc

train
./experiments/scripts/train_faster_rcnn.sh 0 pascal_voc res101

test
./experiments/scripts/test_faster_rcnn.sh 0 pascal_voc res101

#### Yolov4
#### Mobilenetv3
#### RefineDet
#### U-Net







