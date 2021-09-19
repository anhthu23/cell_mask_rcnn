# coding:utf-8
import os
from PIL import Image

# bmp to jpg
def bmpToJpg(file_path):
    i = 0
    for fileName in os.listdir(file_path):
        # print(fileName)
        newFileName = str(i) + ".png"
        i = i + 1
        print(newFileName)
        im = Image.open(file_path+"\\"+fileName)
        im.save(file_path+"\\converted\\"+newFileName)

def main():
    file_path = "E:/Thesis/Mask_RCNN-master/resultdatasets"
    bmpToJpg(file_path)

main()