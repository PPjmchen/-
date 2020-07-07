import json
import cv2
import random
from SelectiveSearch import get_windows
from tools.iou import cal_iou
from tools.config import opt
from tqdm import tqdm

def get_samples(sub_dataset):
    # 对训练集中的每张图片采集正负样本
    samples = {}
    for img_path, label in tqdm(sub_dataset.items()):
        img = cv2.imread(img_path)
        bbox = label['bbox']

        if img.shape[2] == 1:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        effectiveWindows = get_windows(img)

        p_samples = []
        n_samples = []
        for w in effectiveWindows:
            if cal_iou(w, bbox) > opt.iou_threshold:
                p_samples.append(w)
            else:
                n_samples.append(w)

        if len(n_samples) >= len(p_samples):
            n_samples = n_samples[:len(p_samples)]

        samples[img_path] = {'p_sample': p_samples, 'n_sample': n_samples}
    return samples

def main(**kwargs):

    with open('jsons/dataset.json', 'r') as f:
        dataset = json.load(f)

    # 选取所有数据的 80% 作为训练集
    train_num = int(opt.train_proportion * len(dataset))    # opt.train_proportion = 0.8
    train_slice = random.sample(list(dataset), train_num)
    train_dataset = {k:dataset[k] for k in train_slice}

    # 剩余 20% 作为测试集
    test_slice = []
    for key in dataset.keys():
        if key not in train_slice:
            test_slice.append(key)
    test_dataset = {k: dataset[k] for k in test_slice}

    # 对训练集中的每张图片采集正负样本
    print("get train samples......")
    train_samples = get_samples(train_dataset)
    json_data = json.dumps(train_samples)
    f = open('jsons/train_sample.json', 'w')
    f.write(json_data)
    f.close()

    # 对测试集中的每张图片采集正负样本
    print("get test samples......")
    test_samples = get_samples(test_dataset)
    json_data = json.dumps(test_samples)
    f = open('jsons/test_sample.json', 'w')
    f.write(json_data)
    f.close()


if __name__ == '__main__':
    main()