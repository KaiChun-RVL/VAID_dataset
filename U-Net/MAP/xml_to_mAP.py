import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = ["1","2","3","4","5","6","7"]
xml_list = []



def convert(size, box):
    dw = 1./size[0]
    dh = 1./size[1]
    x = (box[0] + box[1])/2.0
    y = (box[2] + box[3])/2.0
    w = box[1] - box[0]
    h = box[3] - box[2]
    x = x*dw
    w = w*dw
    y = y*dh
    h = h*dh

    return (x, y, w, h)


def convert_annotation(image_id, num):
    in_file = open('D:/unet(revised)/data/test/test_xml/%s' % (image_id),encoding="utf-8")
    out_file = open('D:/unet(revised)/data/test/test_txt/%s.txt' % (image_id[:-4]), 'w') 
    print(out_file)
    tree = ET.parse(in_file)
    root = tree.getroot()
    size = root.find('size')
    w = int(size.find('width').text)
    h = int(size.find('height').text)

    for obj in root.iter('object'):
        cls = obj.find('name').text
        #print(cls)
        #if cls not in classes:
           #continue
        ########################################
        '''
        cls_id = classes.index(cls)
        print(cls_id)
        if cls_id == 1:
            cls_id = 'sedan'
        elif cls_id == 2:
            cls_id = 'minibus'
        elif cls_id == 3:
            cls_id = 'truck'
        elif cls_id == 4:
            cls_id = 'pickuptruck'
        elif cls_id == 5:
            cls_id = 'bus'
        elif cls_id == 6:
            cls_id = 'cementtruck'
        elif cls_id == 7:
            cls_id = 'trailer'
        '''    
        ########################################
        #cls_id = classes.index(cls)
        #print(cls_id)
        if cls == '1':
            cls = 'sedan'
        elif cls == '2':
            cls = 'minibus'
        elif cls == '3':
            cls = 'truck'
        elif cls == '4':
            cls = 'pickuptruck'
        elif cls == '5':
            cls = 'bus'
        elif cls == '6':
            cls = 'cementtruck'
        elif cls == '7':
            cls = 'trailer'   
        ########################################
        xmlbox = obj.find('bndbox')
        b = (float(xmlbox.find('xmin').text), float(xmlbox.find('ymin').text), float(xmlbox.find('xmax').text), float(xmlbox.find('ymax').text))
        #bb = convert((w, h), b)
        out_file.write(str(cls) + " " + " ".join([str(a) for a in b]) + '\n')  #寫入txt檔的內容

def get_xml_files(path):
    for filename in os.listdir(path):
        if filename.endswith(".xml"):
            xml_list.append(os.path.join(filename))
    return xml_list,filename  #加了filename

	
if __name__  == '__main__':
    path = 'D:/unet(revised)/data/test/test_xml/'  #路徑
    xml_list,filename = get_xml_files(path)   #本來是 get_xml_files(path)
    num = 0
    for image_id in xml_list:
        print(image_id)
        num += 1
        convert_annotation(image_id, num)  

