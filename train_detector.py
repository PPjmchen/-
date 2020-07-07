import numpy as np
import json
import cv2
import pickle
from Haar import harr
from svm import SVM
from tools.config import opt
from tqdm import tqdm

# 从一张原始图片中根据预处理得到的正负样本，剪切缩放得到固定size的子窗口
def get_sample_data(img_path, sample, size = opt.crop_resize):
    crop_imgs = []
    labels = []
    img = cv2.imread(img_path)

    # 若原始图片为彩色，将其转换为灰度图
    if img.shape[2] != 1:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    for bbox in sample['p_sample']:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cropImg = img[y:y+h, x:x+w]
        cropImg = cv2.resize(cropImg, size)
        label = 1
        crop_imgs.append(cropImg)
        labels.append(label)

    for bbox in sample['n_sample']:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cropImg = img[y:y+h, x:x+w]
        cropImg = cv2.resize(cropImg, size)
        label = -1
        crop_imgs.append(cropImg)
        labels.append(label)

    # 返回原始图片的一系列子窗口图片以及对应的正负样本标签
    return crop_imgs, labels


def main(**kwargs):
    opt._parse(kwargs)
    # 获取训练样本和其对应的标签，train_sample.json文件由tools/getAllSample.py生成
    with open('jsons/train_sample.json' ,'r') as f:
        train_samples = json.load(f)

    # 定义SVM模型
    svm = SVM()

    print("loading data...")
    haar_features = []
    labels = []
    # 获取训练集中所有的子窗口特征及对应类别
    for img_path, sample in tqdm(train_samples.items()):
        # 获取原始图片的一系列子窗口图片以及对应的正负样本标签
        crop_imgs, label = get_sample_data(img_path, sample)
        for crop_img_gray in crop_imgs:

            # 若子窗口中所有像素值相等，则去除
            if crop_img_gray.max() == crop_img_gray.min():
                continue

            # 对子窗口提取x2类型的Haar特征
            feature = harr(crop_img_gray)
            haar_features.append(feature)
        labels.extend(label)

    # 将所有的子窗口特征和对应类别转换为array，参与SVM的训练
    haar_features = np.array(haar_features)
    labels = np.array(labels)
    svm.train(haar_features, labels)

    # 保存svm的内部参数，用于测试和后续模型整合
    svm_parameter = open("models/svm_params.pkl", 'wb')
    str = pickle.dumps(svm)
    svm_parameter.write(str)
    svm_parameter.close()

if __name__ == '__main__':
    main()