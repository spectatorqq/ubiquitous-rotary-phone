# 通过分类文档和split切分文档，划分测试集和训练集
# 只是读取文档 ，进行文档内文字的读写，不处理视频
import os

dir = 'F:\\Learning_Python\\document\\hmdb51_org\\test_train_splits\\testTrainMulti_7030_splits\\'

train_file = open('train.txt', 'w')
test_file = open('test.txt', 'w')
classes = open('hmdb51_int_classes.txt', 'r')

list_clsss = classes.readlines()
#  获取分类文档的每一行
i = 1
for cla in list_clsss:
    # 切分类别和 int 标签
    lab, clas = cla.split(' ')
    clas = clas.strip()
    # 获取每个test_split 文档

    for file in os.listdir(dir):
        # # 如果 类别符合 就打开它，进行分训练集合测试集
        if clas in file:
            # 读取每个符合的文档 格式为 视频名 train_or_test
            text = open(dir + file, 'r')
            print('open txt file')
        # 内容保存为列表
            lines = text.readlines()
            for line in lines:
                # 切分 train or test
                t, num, _ = line.split(' ')
                if num == '1':
                    train_file.write(clas + '/' + clas + '/' + t + ' ' + lab + '\n')
                else:
                    test_file.write(clas + '/' + clas + '/' + t + '\n')
            text.close()
            i += 1
            print(i)
    print('finish write')

classes.close()
train_file.close()
test_file.close()
print('finish file')
