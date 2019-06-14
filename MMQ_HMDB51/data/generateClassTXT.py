import os
# from os import *
# file_dir = 'data/'
'''
根据数据集生成class_txt
'''
class_file = 'hmdb51_int_classes.txt'

class_file = open(class_file, 'w')
print('open file')
dir = 'F:\\Learning_Python\\document\\hmdb51_org\\HMDB51\\'
i = 1
for file_name in os.listdir(dir):
    if os.path.exists(dir +file_name):
        if os.path.isdir(dir + file_name):
            class_file.write(str(i) + " " + file_name + "\n")
            print('start write class_name')
            i += 1
class_file.close()
