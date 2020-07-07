import json
import ipdb
from tools.get_rec_dataset import get_dataset
from eigenface import Eigenface

CLASSES = {
    'curry':0,
    'dujiang':1,
    'huangbo':2,
    'JacksonYee':3,
    'jingtian':4,
    'JohnMayer':5,
    'kobe':6,
    'shiyuanlimei':7,
    'xuezhiqian':8,
    'dili':9
}

def main():
    # 读取所有原始图片路径和对应的label
    with open ('jsons/dataset.json', 'r') as f:
        dataset = json.load(f)

    '''
    截取所有的人脸区域并缩放至(100, 100)并flatten至10000维的一维向量，每一类随机选5张人脸作为训练集，共10类，剩余作为测试集
    train_imgs: 存放训练集的所有人脸图片，shape = (50, 10000)
    class_train_imgs: 该列表存放10个array(array.shape = (5, 10000))，分别为10类人脸图片, len(class_train_imgs) = 10
    train_cls: 该列表存放上述10个array所对应的人脸类别, len(train_cls) = 10
    test_imgs: 存放测试集的所有人脸图片, shape = (168, 10000)
    test_cls: 该列表存放测试集每张人脸图片的类别，len(test_cls) = 168
    '''
    train_imgs, class_train_imgs, train_cls, test_imgs, test_cls = get_dataset(dataset)

    eigenface = Eigenface()

    # 对训练集进行样本主成分分析，获取Eigenface
    eigenface.pca(train_imgs, class_train_imgs)

    # 使用KNN对测试集进行识别
    err = 0
    for i in range(len(test_imgs)):
        predict_cls = eigenface.KNN_predicgt(test_imgs[i], train_cls)
        if predict_cls != test_cls[i]:
            err += 1
    score = 1 - err/len(test_cls)
    ipdb.set_trace()
    print('score = ', score)

if __name__ == '__main__':
    main()