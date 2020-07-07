import json
import os
import re

def main():

    data = {}

    data_root = 'dataset'
    peoples = os.listdir(data_root)

    for people in peoples:
        sub_dataset = os.path.join(data_root, people)
        label = os.path.join(sub_dataset, people+'.txt')
        with open(label, 'r') as f:
            for groundTruth in f.readlines():
                groundTruth = re.split(',| ', groundTruth.strip('\n'))
                img_path = os.path.join(data_root, groundTruth[0])
                bbox = groundTruth[-4:]
                
                bbox = list(map(int,bbox))
                # 为便于后续使用opencv缩放、切割等，将bbox调整为[x, y, w, h]的形式
                bbox[2], bbox[3] = bbox[3], bbox[2]
                
                data[img_path] = {}
                data[img_path]['bbox'] = bbox
                data[img_path]['class'] = people
        f.close()
    
    json_data = json.dumps(data)
    f = open('jsons/dataset.json', 'w')
    f.write(json_data)
    f.close()


if __name__ == '__main__':
    main()