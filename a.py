#完成框和文本在原图上的标注
import os
import cv2
import time
import math
import numpy as np
from PIL import Image, ImageDraw, ImageFont
res_file = 'data/output/EAST/43_transformer.txt'
textfile = 'data/outtxt/43_power.txt'
output_dir_img = 'data/output/outimg/'
image_path='data/input/test_img/43_power.jpg',
result=[]
textresult=[]
def cv2ImgAddText(img, text, left, top, textColor=(0, 255, 0), textSize=20):
    if (isinstance(img, np.ndarray)):  # 判断是否OpenCV图片类型
        img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)
    fontStyle = ImageFont.truetype(
        "font/simsun.ttc", textSize, encoding="utf-8")
    draw.text((left, top), text, textColor, font=fontStyle)
    return cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    
im = cv2.imread('data/input/test_img/43_transformer.jpg',cv2.IMREAD_COLOR)
with open(res_file, 'r',encoding='utf_8') as f:
    for line in f:
        result.append(list(line.strip('\n').split(',')))
with open(textfile, 'r',encoding='utf_8') as f:
    for line in f: 
        textresult.append(list(line.strip('\n').split(',')))
# def takeone(elem):
    # elem[0] = int(elem[0].split('_',4)[3])
    # return elem[0]
# textresult.sort(key=takeone)
for x, y in zip(result, textresult):
            a = (int(x[0]), int(x[1]))
            b = (int(x[4]), int(x[5]))
            im = cv2.rectangle(im, a, b, (0, 255, 0), 1)
            c = int(x[4])
            d = int(x[1])
            txt = y[1]
            im = cv2ImgAddText(im, txt, c+10, d + 5, (255,0,0), 20)

#img_path = os.path.join(output_dir_img, os.path.basename(im_fn))
img_path = output_dir_img+'43_transformer.jpg'
cv2.imwrite(img_path, im)
            #cv2.imwrite(img_path, im[:, :, ::-1])