#因为文字识别输出的txt是一个crop_img一个txt,需要这个文件来对其进行结果合并，得到一个大图一个文字识别的txt.
import os
#input_path = "C:/Users/admin/Desktop/EAST_RCNN_for_OCR-master/data/outtxt/" #此处填好自己的路径，注意最后的"/"
#获取目标文件夹的路径
filedir =  "C:/Users/admin/Desktop/EAST_RCNN_for_OCR-master/data/outtxt/" 
#获取当前文件夹中的文件名称列表  
filenames=os.listdir(filedir)
#打开当前目录下的result.txt文件，如果没有则创建
f=open('result.txt','w',encoding='utf-8')
#先遍历文件名
for filename in filenames:
    filepath = filedir+filename
    #遍历单个文件，读取行数
    for line in open(filepath,'r',encoding='utf-8'):
        f.writelines(line)
        f.write('\n')
#关闭文件
f.close()