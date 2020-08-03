import numpy as np
import matplotlib.pyplot as plt
from pylab import *
import os
import sys
import time
import cv2
from PIL import Image
from keras.preprocessing.image import *
from keras.utils.np_utils import to_categorical
from keras.models import load_model



if __name__ == '__main__':
    nb_classes = 8   #類別
    conf_m = zeros((nb_classes, nb_classes), dtype=float)
    total = 0
    #需要的話更改路徑
    #test_label_path = 'D:/graduate_project_datasets/TRAINING_DATASETS/test_data_100/test_label2/'
    #test_label_path = 'D:/graduate_project_datasets/TRAINING_DATASETS/test_5747_7647/test_label/'
    #test_label_path = 'D:/graduate_project_datasets/TRAINING_DATASETS/test_1690_2450/test_label/'
    
    test_label_path = 'D:/unet(revised)/data_new/split/test/test_label/'    #自己標的label圖(黑)D:/unet(revised)/data/test/test_label/
    #predict_path = 'D:/Admin/Desktop/unet-rgb/test_IoU_label/plants/unet+_vgg19_residual/test_label/'
    predict_path = 'D:/unet(revised)/data/predict/predict_label/'    #預測出的圖(黑)
    #predict_path = 'D:/Admin/Desktop/unet-rgb/test_IoU_label/0320/unet_vgg16_finetune_dataaug/new/'
    dirs = os.listdir(test_label_path)
    mean_acc = 0.

    for filename in dirs:
        #
        print(filename)
        #
        total += 1
        pred = img_to_array(Image.open(predict_path + filename)).astype(int)
        label = img_to_array(Image.open(test_label_path + filename)).astype(int)
        flat_pred = np.ravel(pred)
        flat_label = np.ravel(label)
        acc = 0.
        for p, l in zip(flat_pred, flat_label):
            if l == 255:
                continue
            if l < nb_classes and p < nb_classes:
                conf_m[l, p] += 1
##            else:
##                print('Invalid entry encountered, skipping! Label: ', l,
##                    ' Prediction: ', p)
            if l==p:
                acc+=1
        acc /= flat_pred.shape[0] # TP / predictions only
        mean_acc += acc
    mean_acc /= total
    print('mean acc: %f:' % mean_acc)
    I = np.diag(conf_m)
    U = np.sum(conf_m, axis=0) + np.sum(conf_m, axis=1) - I
    IOU = I/U
    IOU = IOU.round(decimals=2)
    meanIOU = np.mean(IOU)

    print(IOU)
    print('meanIOU: %f' % meanIOU)
    print('pixel acc: %f' % (np.sum(np.diag(conf_m))/np.sum(conf_m)))
    
    fig, ax = plt.subplots(figsize=(8,6))

    # Example data
    classes = ('Background', 'sedan', 'minibus', 'truck', 'pickuptruck', 'bus', 'cement truck', 'trailer')  #選擇種類作為iou比較
    #classes = ('Background', 'Red', 'Yellow')
    y_pos = np.arange(len(classes))
    performance = IOU

    bar = ax.barh(y_pos, performance, align='center')
    bar[0].set_color('black')   #background : black  
    bar[1].set_color('brown')   #sedan : white#brown
    bar[2].set_color('blue')    #minibus : blue
    bar[3].set_color('purple')  #truck : purple
    bar[4].set_color('green')   #pickuptruck : green
    bar[5].set_color('red')     #bus : red
    bar[6].set_color('yellow')   #cementtruck : yellow
    bar[7].set_color('aqua')    #trailer : aqua
    #bar[8].set_color('y')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(classes, style='italic', fontweight='bold')
    ax.invert_yaxis()  # labels read top-to-bottom
    for i, v in enumerate(IOU):
        plt.text(v, i, " "+str(v), va='center', style='italic', fontweight='bold') 
    ax.set_xlabel('IoU', fontweight='bold')
    ax.set_title('mIoU:' +  "{:.2%}".format(meanIOU), fontweight='bold')

    plt.show()
