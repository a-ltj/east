#import crnninterface
import os
import cv2
import time
import math
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from eastdemo import east_detect
from test_shadownet import recognize


def get_images(test_data_dir):
    files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(test_data_dir):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(files)))
    return files

def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    # 创建一个可以在给定图像上绘图的对象
    draw = ImageDraw.Draw(img)
    # 字体的格式
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    # 绘制文本
    draw.text((left, top), text, textColor, font=fontStyle)
    # 转换回OpenCV格式
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)

    
def sort_poly(p):
    min_axis = np.argmin(np.sum(p, axis=1))
    p = p[[min_axis, (min_axis+1)%4, (min_axis+2)%4, (min_axis+3)%4]]
    if abs(p[0, 0] - p[1, 0]) > abs(p[0, 1] - p[1, 1]):
        return p
    else:
        return p[[0, 3, 2, 1]]
        
def cut_roi(output_dir,image, box, im_fn, num):
    im_file_path = os.path.join(output_dir,'{}_img_{}_box.jpg'.format(os.path.basename(im_fn).split('.')[0], num))

    im_cut = image[box[0, 1]:box[3, 1], box[0, 0]:box[1, 0], ::-1]

    cv2.imwrite(im_file_path, im_cut)
    return im_file_path
def main():
    #east args
    output_dir = 'data/output/EAST1/'
    output_dir_img = 'data/output/outimg/'
    test_data_dir = 'data/input/test_img/'
    checkpoint_dir = 'model/checkpoint/east/'
    no_write_images = False
    # #crnn args
    #image_path = 'C:\\Users\\admin\\Desktop\\EAST_RCNN_for_OCR-master\\data\\output\\EAST1\\'
    weights_path = 'model/checkpoint/crnnsyn/'
    char_dict_path = 'crnn/data/char_dict/char_dict.json'
    ord_map_dict_path = 'crnn/data/char_dict/ord_map.json'
    output_path = 'data/outtxt/'
    im_fn_list = get_images(test_data_dir)
    for im_fn in im_fn_list:
        im = cv2.imread(im_fn)[:, :, ::-1]
        boxes = east_detect(output_dir,checkpoint_dir,im_fn)
        if boxes is not None:
            res_file = os.path.join(
                output_dir,
                '{}.txt'.format(
                    os.path.basename(im_fn).split('.')[0]))

            with open(res_file, 'w') as f:
                for i, box in enumerate(boxes):
                    box = sort_poly(box.astype(np.int32))
                    if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        continue
                    f.write('{},{},{},{},{},{},{},{}\n'.format(
                        box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
                        box[3, 1],
                    ))
                    # img1 = cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  # color=(255, 255, 0), thickness=1)
                    cut_roi(output_dir,im[:, :, ::-1], box, im_fn, i)
    #im_fn_list = get_images(image_path)
    textfile = recognize(output_dir,weights_path,char_dict_path,ord_map_dict_path)
    print(textfile)
    # for im_fn in im_fn_list:
        # textfile = recognize(im_fn,weights_path,char_dict_path,ord_map_dict_path,output_path)
        # print(textfile)
            #text.sort()
                    #print(text)
                    # x = box[0, 0]
                    # y = box[0, 1]
                    # cv2.putText(im[:, :, ::-1], text, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,color=(0,255,0), thickness=1)
            
            # with open(res_file, 'w') as f:
                # for i, box in enumerate(boxes):
                    # box = sort_poly(box.astype(np.int32))
                    # if np.linalg.norm(box[0] - box[1]) < 5 or np.linalg.norm(box[3] - box[0]) < 5:
                        # continue
                    # f.write('{},{},{},{},{},{},{},{}\r\n'.format(
                        # box[0, 0], box[0, 1], box[1, 0], box[1, 1], box[2, 0], box[2, 1], box[3, 0],
                        # box[3, 1],
                    # ))
                    # cv2.polylines(im[:, :, ::-1], [box.astype(np.int32).reshape((-1, 1, 2))], True,
                                  # color=(255, 255, 0), thickness=1)
                    # im_file_path = cut_roi(output_dir,im[:, :, ::-1], box, im_fn, i)
                    # text = recognize(im_file_path,weights_path,char_dict_path,ord_map_dict_path,output_path)
                    # print(text)
                    # x = box[0, 0]
                    # y = box[0, 1]
                    # cv2.putText(im[:, :, ::-1], text, (x, y), cv2.FONT_HERSHEY_COMPLEX, fontScale=0.5,color=(0,255,0), thickness=1)
        # res_file = 'data/output/EAST/12_power.txt'
        # result=[]
        # textresult=[]
        # with open(res_file, 'r',encoding='utf_8') as f:
            # for line in f:
                # result.append(list(line.strip('\n').split(',')))
        # with open(textfile, 'r',encoding='utf_8') as f:
           # for line in f: 
                # textresult.append(list(line.strip('\n').split('\t')))
        # def takeone(elem):
            # elem[0] = int(elem[0].split('_',4)[3])
            # return elem[0]
        # textresult.sort(key=takeone)
        # for x, y in zip(result, textresult):
                # a = (int(x[0]), int(x[1]))
                # b = (int(x[4]), int(x[5]))
                # img1 = cv2.rectangle(im[:, :, ::-1], a, b, (0, 255, 0), 1)
                # c = int(x[4])
                # d = int(x[1])
                # txt = y[1]
                # img = cv2ImgAddText(img1, txt, c-10, d - 5, (255,0,0), 10)
        # if not no_write_images:
            # img_path = os.path.join(output_dir_img, os.path.basename(im_fn))
            # cv2.imwrite(img_path, img)
                #cv2.imwrite(img_path, im[:, :, ::-1])

if __name__ == '__main__':
    main()