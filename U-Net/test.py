from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import img_to_array, load_img
import numpy as np
from keras.models import *
from keras.layers import *
from keras.optimizers import *
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as keras
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
from scipy.ndimage.measurements import label
import os
import glob
import skimage.io as io
import skimage.transform as trans
import cv2
import time
from PIL import Image
from matplotlib import pyplot as plt
from datetime import datetime
#from keras_efficientnets import EfficientNetB5
'''
Sky = [128,128,128]
Building = [128,0,0]
Pole = [192,192,128]
Road = [128,64,128]
Pavement = [60,40,222]
Tree = [128,128,0]
SignSymbol = [192,128,128]
Fence = [64,64,128]
Car = [64,0,128]
Pedestrian = [64,64,0]
Bicyclist = [0,128,192]
Unlabelled = [0,0,0]

COLOR_DICT = np.array([Sky, Building, Pole, Road, Pavement,
                          Tree, SignSymbol, Fence, Car, Pedestrian, Bicyclist, Unlabelled])
'''

def getCurrentClock():
    #return time.clock()
    return datetime.now()


def draw_bounding_box(img, ori_img, bbx_img, frame_num, filename):
    Dilate_image=[]
    write = True #要不要計算MAP
    f=open('./MAP/car/detection-results/%s.txt' %(filename[:-4]),'a')   #創一個txt檔   #'./map_txt/%d.txt' 
    print(img.shape[0])
    for i in range(img.shape[0]):
        #cv2.imshow('a',img[i])
        #cv2.waitKey(0)

        output_img  = cv2.cvtColor(img[i],cv2.COLOR_BGR2GRAY)
        
        #print(output_img.shape)
        output_img = cv2.resize(output_img, (480, 360), interpolation=cv2.INTER_CUBIC)
      
        #cv2.imshow('img' , output_img)
        #cv2.waitKey(0)        
        
        th = cv2.threshold(output_img, 244, 255, cv2.THRESH_BINARY)[1]
        
        th = cv2.erode(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=2)
        dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=2)
        #cv2.imshow('dilated' , dilated)
        #cv2.waitKey(0)
        contours, hier = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)#opencv3返回3個值
        #print(contours)
        #----------看侵蝕圖
        if i == 0:
            dilated_car = dilated
        

#        output_img = np.reshape(output_img,(360, 480,1))
#        output_img = cv2.merge((output_img,output_img,output_img))
        for c in contours:
           #print('iii:',i)
            # 获取矩形框边界坐标
            x, y, w, h = cv2.boundingRect(c)
            if i == 0:
                charac = 'sedan '
            elif i==1:
                charac ='minibus '
            elif i==2:
                charac ='truck '
            elif i==3:
                charac ='pickuptruck '
            elif i==4:
                charac ='bus '
            elif i==5:
                charac ='cementtruck '
            elif i==6:
                charac ='trailer '
            #charac = 'sedan ' if i == 0 else 'minibus'

            #寫mAP測試用的TEXT 類別/信心值/xmin/ymin/smax/yamx
            
            #-----------------------------------------------------

#            xmin = np.append(xmin, [x])
#            ymin = np.append(ymin, [y])
#            xmax = np.append(xmax, [x + w])
#            ymax = np.append(ymax, [y + h])
            
            # 计算矩形框的面积
            area = cv2.contourArea(c)
            if area > 50 and i == 0:
                text = 'sedan '
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (225, 225, 255), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ') #
                    f.write('%d ' % int(x * 1137 / 480))    #數字比例要改!    1920改成1137，1080改成640
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            elif area > 50 and i == 1:
                text = 'minibus'
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 0), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (225, 0, 0), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ')
                    f.write('%d ' % int(x * 1137 / 480))
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            elif area > 50 and i == 2:
                text = 'truck'
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 0, 255), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (225, 0, 255), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ')
                    f.write('%d ' % int(x * 1137 / 480))
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            elif area > 50 and i == 3:
                text = 'pickuptruck'
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ')
                    f.write('%d ' % int(x * 1137 / 480))
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            elif area > 50 and i == 4:
                text = 'bus'
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 0, 225), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ')
                    f.write('%d ' % int(x * 1137 / 480))
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            elif area > 50 and i == 5:
                text = 'cementtruck'
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (0, 255, 255), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (0, 255, 255), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ')
                    f.write('%d ' % int(x * 1137 / 480))
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            elif area > 50 and i == 6:
                text = 'trailer'
                cv2.putText(bbx_img, text, (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 0.8, (255, 255, 0), 1, cv2.LINE_AA)
                cv2.rectangle(bbx_img, (x, y), (x + w, y + h), (225, 255, 0), 2)
                if write == True:
                    f.write(charac)
                    f.write('1.0 ')
                    f.write('%d ' % int(x * 1137 / 480))
                    f.write('%d ' % int(y * 640 / 360))
                    f.write('%d ' % int((x + w )* 1137 / 480))
                    f.write('%d\n' % int((y + h )* 640 / 360))
            #cv2.imshow('bbx_img' , bbx_img)
            #cv2.waitKey(0) 
    #cv2.imshow('dil_car',dilated_car)
    #cv2.imshow('dil_person',dilated)

    #ori_img = cv2.addWeighted(output_img, 1, ori_img, 0, 0)    
    #ori_img = 0.5 * output_img + (1 - 0.5) * ori_img
    return bbx_img

def test_multi_images(test_path, label_save_dir, visualize_save_dir, write_visualize):
    
    initial_time = getCurrentClock()
    
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    ##要用時再開##out = cv2.VideoWriter('output.wmv',fourcc, 10.0, (int(480),int(360)))
    dirs = os.listdir(test_path)
    #print('test_path:',test_path)
    frame_num = 0
    for filename in dirs:
        print(filename)
#        start = time.time()
        frame_num += 1
        
#        print('-----frame_cnt------:',frame_num)
#        print('filename:',filename) 
        img_test_path = test_path+filename                                          #亂數
        #print('filename',filename)
#        img_test_path = os.path.join(test_path, str(frame_num)+'.jpg')               #編號1開始用法

#       testpath圖片路徑        
        img_test = load_img(img_test_path, grayscale=False, target_size=[512, 512])
#        print(type(img_test))
#        cv2.imshow('img_test',img_test)
        img_test = img_to_array(img_test)
        img_test = np.resize(img_test,(1,512,512,3))
        img_test = img_test.astype('float32')
        img_test /= 255
        
#        start_time = getCurrentClock()
        imgs_mask_test = model.predict(img_test, batch_size=1, verbose=1)
        
        if deep_supervision:
            imgs_mask_test = np.reshape(imgs_mask_test,(4, 512,512,3))
            imgs_mask_test1 = imgs_mask_test[0]
            imgs_mask_test2 = imgs_mask_test[1]
            imgs_mask_test3 = imgs_mask_test[2]
            imgs_mask_test4 = imgs_mask_test[3]
            imgs_mask_test = (imgs_mask_test1 + imgs_mask_test2 + imgs_mask_test3 + imgs_mask_test4) / 4
            imgs_mask_test = np.argmax(imgs_mask_test,axis=-1)
            imgs_mask_test = np.reshape(imgs_mask_test,(512,512,1))
            imgs_mask_test = imgs_mask_test.astype(np.uint8)
            #print(imgs_mask_test1.shape)
            #imgs_mask_test = np.sum[imgs_mask_test,axis=0]
            #print(imgs_mask_test.shape)
        else:  
            imgs_mask_test = np.reshape(imgs_mask_test,(512,512,8))#類別數+1(背景)
            imgs_mask_test = np.argmax(imgs_mask_test,axis=-1)
            
            #confidence=np.amax(imgs_mask_test,axis=-1)
            #confidence[confidence <= 0.5] = 0
            #confidence[confidence > 0.5] = 1
            #imgs_mask_test = (confidence * imgs_mask_test).astype(int)

            imgs_mask_test = np.reshape(imgs_mask_test,(512,512,1))
            imgs_mask_test = imgs_mask_test.astype(np.uint8)
        
        # save predict mask for IoU evaluation
        #save_dir = './test_IoU_label/0320/inception_resnet_v2/'

        IoU_mask = cv2.resize(imgs_mask_test, (1137, 640), interpolation=cv2.INTER_CUBIC)   #要跟原影像size一樣 不然IoU會有問題(1137, 640)(1920, 1080)
        #cv2.imshow('imgs_mask_test',imgs_mask_test)
        #cv2.waitKey(0)
        cv2.imwrite(os.path.join(label_save_dir, filename[:-4] + '.png'), IoU_mask)
        #cv2.imwrite(os.path.join(save_dir, str(frame_num) + '.png'), IoU_mask)
        #---------------------------------------------------------------------------

#       根據類別對應的labelnum
        background = np.array([0])
        sedan = np.array([1])
        minibus = np.array([2])
        truck = np.array([3])
        pickuptruck = np.array([4])
        bus = np.array([5])
        cementtruck = np.array([6])
        trailer = np.array([7])
        
#        print(imgs_mask_test.shape)

        #cv2.imwrite('mask.png',label_seg)
       
        label_seg = []

        sedan_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)         # 512 / 512
        sedan_seg[(imgs_mask_test==sedan).all(axis=2)] = [255,255,255]                      #sedan[255,255,255]
        sedan_seg = sedan_seg.astype(np.uint8)

        minibus_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)       #minibus[255,0,0]  
        minibus_seg[(imgs_mask_test==minibus).all(axis=2)] = [255,255,255]    
        minibus_seg = minibus_seg.astype(np.uint8)

        truck_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)         #truck[255,0,255] 
        truck_seg[(imgs_mask_test==truck).all(axis=2)] = [255,255,255]     
        truck_seg = truck_seg.astype(np.uint8)

        pickuptruck_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)   #pickuptruck[0,255,0]
        pickuptruck_seg[(imgs_mask_test==pickuptruck).all(axis=2)] = [255,255,255]      
        pickuptruck_seg = pickuptruck_seg.astype(np.uint8)

        bus_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)           #bus[0, 0, 225] 
        bus_seg[(imgs_mask_test==bus).all(axis=2)] = [255,255,255]      
        bus_seg = bus_seg.astype(np.uint8)
        
        cementtruck_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)   #cementtruck[0, 255, 255] 
        cementtruck_seg[(imgs_mask_test==cementtruck).all(axis=2)] = [255,255,255]     
        cementtruck_seg = cementtruck_seg.astype(np.uint8)

        trailer_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)       #trailer[255,255,0]
        trailer_seg[(imgs_mask_test==trailer).all(axis=2)] = [255,255,255]      
        trailer_seg = trailer_seg.astype(np.uint8)

        label_seg = np.concatenate(([sedan_seg], [minibus_seg], [truck_seg], [pickuptruck_seg], [bus_seg], [cementtruck_seg], [trailer_seg]),axis = 0)
        #print(label_seg)
        #os.system('pause')
        output_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)

        output_seg[(imgs_mask_test==background).all(axis=2)] = [0, 0, 0]            #background : black       
        output_seg[(imgs_mask_test==sedan).all(axis=2)] = [255,255,255]             #sedan : white
        output_seg[(imgs_mask_test==minibus).all(axis=2)] = [255,0,0]               #minibus : blue[255,0,0]##############BGR##################################
        output_seg[(imgs_mask_test==truck).all(axis=2)] = [255,0,255]               #truck : purple
        output_seg[(imgs_mask_test==pickuptruck).all(axis=2)] = [0,255,0]           #pickuptruck : green
        output_seg[(imgs_mask_test==bus).all(axis=2)] = [0, 0, 255]                 #bus : red[0, 0, 225]##############BGR##################################
        output_seg[(imgs_mask_test==cementtruck).all(axis=2)] = [0, 255, 255]       #cementtruck : yellow[0, 255, 255]##############BGR##################################
        output_seg[(imgs_mask_test==trailer).all(axis=2)] = [255,255,0]             #trailer : aqua[255,255,0]##############BGR##################################
        
        output_seg = output_seg.astype(np.uint8)
        output_seg = cv2.resize(output_seg, (640, 480), interpolation=cv2.INTER_CUBIC)
        if write_visualize:
            cv2.imwrite(os.path.join(visualize_save_dir, filename[:-4] + '.png'), output_seg)

    
        path = "./results/test/"#"./results/test/"不用看
        #img = np.zeros((img_test.shape[1], img_test.shape[2], 3), dtype=np.uint8)
        ori_img = cv2.imread(test_path+filename)
        #cv2.imshow('a',ori_img)
        #cv2.waitKey(0)
#        ori_img = cv2.imread(os.path.join(test_path, str(frame_num)+'.jpg'))
        ori_img = cv2.resize(ori_img, (480, 360), interpolation=cv2.INTER_CUBIC)
       
        
        #複製一張原影像為了畫bbx上去
        bbx_img = ori_img.copy()
        bbx_img = draw_bounding_box(label_seg, ori_img, bbx_img, frame_num, filename) #加了filename


        #out.write(bbx_img)
#        ori_img = cv2.resize(ori_img, (1360, 720), interpolation=cv2.INTER_CUBIC)
#        cv2.imshow("ori_img", ori_img)

        bbx_img = cv2.resize(bbx_img, (640, 480), interpolation=cv2.INTER_CUBIC)
        mask_and_bbx_img = np.concatenate((output_seg,bbx_img),axis = 0)
        
        #cv2.imshow('mask_and_bbx_img', mask_and_bbx_img) #可以看visual_iou圖片
        #cv2.imshow('bbx_img', bbx_img)

        #cv2.imshow('test',img)
        
        #cv2.waitKey(0) #可以看visual_iou圖片

        
        #path='E:/model_output/mobilenet_v2+unet_iou/batchsize=2bottleincept_mod_best/complex/iou/'
        #cv2.imwrite(os.path.join(path, filename[:-4] + '.png'), mask_and_bbx_img)#這行儲存visual_iou圖片
    
    nowMicro = getCurrentClock()
    print("FPS: %0.4f ---" % float(frame_num/(nowMicro - initial_time).total_seconds()))
    cv2.destroyAllWindows()

#test影片(一次丟一個影片)
'''
def test_video():
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestr = time.strftime("%Y%m%d-%H%M%S")
    out = cv2.VideoWriter(timestr + '.avi',fourcc, 10.0, (int(640),int(960)))
    dirs = os.listdir(test_path)
    initial_time = getCurrentClock()
    cap = cv2.VideoCapture('E:/畢業光碟製作/3. 論文實作目錄/輸入之影像視訊目錄/test_mix.avi')
    
    if (cap.isOpened()== False):       
        print("Error opening video stream or file")
    frame_num = 0
    # Read until video is completed
    while(cap.isOpened()):
        
        frame_num += 1
        ret, frame = cap.read()
        if ret == True:
            print('kk')
            #if frame_num  == 200:
            #    break
            frame = cv2.resize(frame, (512, 512), interpolation=cv2.INTER_CUBIC)  #先轉SIZE
            #print(type(frame))
            test_frame = Image.fromarray(cv2.cvtColor(frame,cv2.COLOR_BGR2RGB))  #NDARRAY轉PIL
            img_test = img_to_array(test_frame)
            img_test = np.resize(img_test,(1,512,512,3))
            img_test = img_test.astype('float32')
            img_test /= 255
            start = time.time()
            start_time = getCurrentClock()
            imgs_mask_test = model.predict(img_test, batch_size=1, verbose=1)
            end = time.time()
            #print('PREDICT_time:',end-start)=

            start = time.time()
            if deep_supervision:
                imgs_mask_test = np.reshape(imgs_mask_test,(4, 512,512,3))
                imgs_mask_test1 = imgs_mask_test[0]
                imgs_mask_test2 = imgs_mask_test[1]
                imgs_mask_test3 = imgs_mask_test[2]
                imgs_mask_test4 = imgs_mask_test[3]
                imgs_mask_test = (imgs_mask_test1 + imgs_mask_test2 + imgs_mask_test3 + imgs_mask_test4) / 4
                imgs_mask_test = np.argmax(imgs_mask_test,axis=-1)
                imgs_mask_test = np.reshape(imgs_mask_test,(512,512,1))
                imgs_mask_test = imgs_mask_test.astype(np.uint8)
                #print(imgs_mask_test1.shape)
                #imgs_mask_test = np.sum[imgs_mask_test,axis=0]
                #print(imgs_mask_test.shape)
            else:  
                imgs_mask_test = np.reshape(imgs_mask_test,(512,512,3))            
                imgs_mask_test = np.argmax(imgs_mask_test,axis=-1)
                imgs_mask_test = np.reshape(imgs_mask_test,(512,512,1))
                imgs_mask_test = imgs_mask_test.astype(np.uint8)
                
            
            # save predict mask for IoU evaluation
            save_dir = './test_label/'
            IoU_mask = cv2.resize(imgs_mask_test, (1920, 1080), interpolation=cv2.INTER_CUBIC)   #要跟原影像size一樣 不然IoU會有問題
            #cv2.imshow('imgs_mask_test',imgs_mask_test)
            #cv2.waitKey(0)
            #cv2.imwrite(os.path.join(save_dir, str(frame_num) + '.png'), IoU_mask)
            #---------------------------------------------------------------------------
            
    #       根據類別對應的labelnum
            background = np.array([0])
            red_fish = np.array([1])
            yellow_fish = np.array([2])
            
    #        print(imgs_mask_test.shape)

            #cv2.imwrite('mask.png',label_seg)
           
            label_seg = []
            red_fish_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)        # 512 / 512
            red_fish_seg[(imgs_mask_test==red_fish).all(axis=2)] = [255,255,255]      
            red_fish_seg = red_fish_seg.astype(np.uint8)
            
            yellow_fish_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)
            yellow_fish_seg[(imgs_mask_test==yellow_fish).all(axis=2)] = [255,255,255]      
            yellow_fish_seg = yellow_fish_seg.astype(np.uint8)
            label_seg = np.concatenate(([red_fish_seg], [yellow_fish_seg]),axis = 0)
    #可視化label mask 
            output_seg = np.zeros((img_test.shape[1],img_test.shape[2],3), dtype=np.int)
            output_seg[(imgs_mask_test==background).all(axis=2)] = [0, 0, 0]
            output_seg[(imgs_mask_test==red_fish).all(axis=2)] = [0, 0, 255]
            output_seg[(imgs_mask_test==yellow_fish).all(axis=2)] = [0, 255, 255]
            output_seg = output_seg.astype(np.uint8)
            output_seg = cv2.resize(output_seg, (640, 480), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('output_seg', output_seg)


    #        ori_img = cv2.imread(os.path.join(test_path, str(frame_num)+'.jpg'))
            frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_CUBIC)
            
            #複製一張原影像為了畫bbx上去
            bbx_img = frame.copy()
            bbx_img = draw_bounding_box(label_seg, frame, bbx_img, frame_num)
            end = time.time()
            #print('visualize_time:',end-start)
            bbx_img = cv2.resize(bbx_img, (640, 480), interpolation=cv2.INTER_CUBIC)
            #cv2.imshow('bbx_img', bbx_img)

            mask_and_bbx_img = np.concatenate((output_seg,bbx_img),axis = 0)
            out.write(mask_and_bbx_img)
            #cv2.imshow('mask_and_bbx_img', mask_and_bbx_img)
            #cv2.waitKey(0)        
            #cv2.imwrite(path, img)

        else:
            cv2.destroyAllWindows()
            break
    
    nowMicro = getCurrentClock()
    print("FPS: %0.4f ---" % float(frame_num/(nowMicro - initial_time).total_seconds()))
'''    
if __name__ == '__main__':
    def categorical_focal_loss(gamma=5., alpha=1.):
        def categorical_focal_loss_fixed(y_true, y_pred):

               # Sum the losses in mini_batch
            y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
            epsilon = K.epsilon()

           
                # Clip the prediction value to prevent NaN's and Inf's
            y_pred = K.clip(y_pred, epsilon, 1.0 - epsilon)    

                # Calculate Cross Entropy
            cross_entropy = -y_true * K.log(y_pred)

                # Calculate Focal Loss
            loss = alpha * K.pow(1 - y_pred, gamma) * cross_entropy

            return K.sum(loss, axis=1)
        return categorical_focal_loss_fixed
    
     
    def class_tversky(y_true, y_pred):
        smooth = 1

        y_true = K.permute_dimensions(y_true, (3,1,2,0))
        y_pred = K.permute_dimensions(y_pred, (3,1,2,0))

        y_true_pos = K.batch_flatten(y_true)
        y_pred_pos = K.batch_flatten(y_pred)
        true_pos = K.sum(y_true_pos * y_pred_pos, 1)
        false_neg = K.sum(y_true_pos * (1-y_pred_pos), 1)
        false_pos = K.sum((1-y_true_pos)*y_pred_pos, 1)
        alpha = 0.3
        return (true_pos + smooth)/(true_pos + alpha*false_neg + (1-alpha)*false_pos + smooth)

    def focal_tversky_loss(y_true,y_pred):
        pt_1 = class_tversky(y_true, y_pred)
        gamma = 1.33
        return K.sum(K.pow((1-pt_1), gamma))
    
    def swish(x):
        return (K.sigmoid(x) * x)
        
    def Hswish(x):
        return x * tf.nn.relu6(x + 3) / 6
        
    deep_supervision = False
    
    modelFile = 'D:/unet(revised)/model/bilunet_0727crop_dataAugment.hdf5' #要測試的model
    
    #modelFile = './model/dataaug/unet_vgg16_finetune_dataaugment_ce_10_swish.hdf5'
    #model = load_model(modelFile, custom_objects={ 'categoriclal_focal_loss_fixed': categorical_focal_loss(gamma=5., alpha=1.)})
    #model = load_model(modelFile, custom_objects={ 'focal_tversky_loss' : focal_tversky_loss, 'swish' : swish })
    
    model = load_model(modelFile, custom_objects={'swish' : swish })                     #有動過激勵函數、損失函數
    
    #model= load_model(modelFile, custom_objects={'Hswish' : Hswish, 'swish' : swish})  #mobilenetv3使用
    #model = load_model('./model/0308unet_fish__with_flower_50epochs_400images.hdf5')     #沒有動過激勵函數、損失函數

    test_path = 'D:/unet(revised)/data_new/test/test_jpg/'    #要測試的圖片路徑D:/unet(revised)/data/test/test_jpg/
    label_save_dir = 'D:/unet(revised)/data/predict/predict_label/'    #預測生成的圖(看起來全黑)
    #label_save_dir = 'D:/Admin/Desktop/unet-rgb/test_IoU_label/0320/unet_vgg16_finetune_dataaug/new/'
    visualize_save_dir = 'D:/unet(revised)/data/predict/predict_visualize/'    #將預測生成的圖visualize成看得出來的的圖(彩色)

    test_multi_images(test_path, label_save_dir, visualize_save_dir, write_visualize = True)    #write_visualizeg是要不要把黑圖變彩色的選項
    #test_video() #要測試影片時再用

