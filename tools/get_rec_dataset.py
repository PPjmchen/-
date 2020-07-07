import cv2
import random
import numpy as np
from tools.config import opt
from PIL import Image


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

def get_dataset(dataset):

    classes = []
    for img_path, label in dataset.items():
        cls = label['class']
        if cls not in classes:
            classes.append(cls)

    cls_dataset = {}
    for cls in classes:
        cls_dataset[cls] = []
    for img_path, label in dataset.items():
        cls_dataset[label['class']].append(img_path)

    train_dataset = {}
    for cls, img in cls_dataset.items():
        train_img = random.sample(img, opt.pca_train_num)
        train_dataset[cls] = train_img

    test_dataset = cls_dataset
    for cls, img in test_dataset.items():
        for i in img:
            if i in train_dataset[cls]:
                img.remove(i)

    face_train = {}
    for cls, imgs in train_dataset.items():
        face_train[cls] = []
        for img_path in imgs:
            bbox = dataset[img_path]['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            img = cv2.imread(img_path)

            cropImg = img[y:y + h, x:x + w]
            cropImg = cv2.resize(cropImg, opt.face_size)
            cropImg = Image.fromarray(cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY))
            cropImg = np.array(cropImg, dtype=np.uint8).flatten()

            face_train[cls].append(cropImg.tolist())
        face_train[cls] = np.array(face_train[cls])

    face_test = {}
    for cls, imgs in test_dataset.items():
        face_test[cls] = []
        for img_path in imgs:
            bbox = dataset[img_path]['bbox']
            x, y, w, h = bbox[0], bbox[1], bbox[2], bbox[3]
            img = cv2.imread(img_path)
            cropImg = img[y:y + h, x:x + w]
            cropImg = cv2.resize(cropImg, opt.face_size)
            cropImg = Image.fromarray(cv2.cvtColor(cropImg, cv2.COLOR_BGR2GRAY))
            cropImg = np.array(cropImg, dtype=np.uint8).flatten()
            face_test[cls].append(cropImg.tolist())
        face_test[cls] = np.array(face_test[cls])

    train_imgs = []     # 存放所有人脸图片，用于获取Eigenface
    train_cls = []
    class_train_imgs = []   # 分类存放所有人脸图片，用于将训练集每一类人脸映射到Eigenface上
    for cls, faces in face_train.items():
        class_sub_set = []
        for face in faces:
            train_imgs.append(face)
            # train_cls.append(CLASSES[cls])
            class_sub_set.append(face)
        class_train_imgs.append(np.array(class_sub_set, dtype=np.uint8))
        train_cls.append(CLASSES[cls])

    test_imgs = []
    test_cls = []
    for cls, faces in face_test.items():
        for face in faces:
            test_imgs.append(face)
            test_cls.append(CLASSES[cls])

    train_imgs = np.array(train_imgs, dtype=np.uint8)
    test_imgs = np.array(test_imgs)
    return train_imgs, class_train_imgs, train_cls, test_imgs, test_cls