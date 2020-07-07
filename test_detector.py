import numpy as np
import json
import cv2
import ipdb
import pickle
from Haar import harr
from svm import SVM

# 从一张原始图片中根据预处理得到的正负样本，剪切缩放得到固定size的子窗口
def get_sample_data(img_path, sample):
    crop_imgs = []
    labels = []

    # 若原始图片为彩色，将其转换为灰度图
    img = cv2.imread(img_path)
    if img.shape[2] == 1:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    for bbox in sample['p_sample']:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cropImg = img[y:y + h, x:x + w]
        cropImg = cv2.resize(cropImg, (50, 50))
        label = 1
        crop_imgs.append(cropImg)
        labels.append(label)

    for bbox in sample['n_sample']:
        x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
        cropImg = img[y:y + h, x:x + w]
        try:
            cropImg = cv2.resize(cropImg, (50, 50))
        except:
            ipdb.set_trace()
        label = -1
        crop_imgs.append(cropImg)
        labels.append(label)

    # 返回原始图片的一系列子窗口图片以及对应的正负样本标签
    return crop_imgs, labels

def main():
    # 获取测试样本和其对应的标签，test_sample.json文件由tools/getAllSample.py生成
    with open('jsons/test_sample.json' ,'r') as f:
        test_samples = json.load(f)
    haar_features = []
    ground_truth = []

    # 获取测试集中所有的子窗口特征及对应类别
    for img_path, samples in test_samples.items():
        crop_imgs, labels = get_sample_data(img_path, samples)
        for crop_img in crop_imgs:
            crop_img_gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
            feature = harr(crop_img_gray)  # 获取子窗口的haar特征
            haar_features.append(feature)
        ground_truth.extend(labels)

    haar_features = np.array(haar_features)

    svm = SVM()
    # 读取训练过的SVM模型
    with open("models/svm_params.pkl", 'rb') as file:
        svm = pickle.loads(file.read())

    # 根据测试集的Haar特征对所有子窗口进行分类
    result = svm.predict(haar_features)

    # 计算模型精度
    gt = np.array(ground_truth)
    predict = np.array(result)
    err = 0
    for i in range(len(gt)):
        if gt[i] == predict[i]:
            continue
        else:
            err += 1
    score = 1 - err / len(gt)
    print('score = ', score)


if __name__ == '__main__':
    main()