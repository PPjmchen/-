# 该函数用于计算两个矩形框之间的IoU(交并比)
def cal_iou(box1, box2):
    # 将[x, y, w, h]调整为[xmin, ymin, xmax, ymax]
    box1 = [box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]]
    box2 = [box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]]

    xmin1, ymin1, xmax1, ymax1 = box1
    xmin2, ymin2, xmax2, ymax2 = box2
    # 计算两个矩形各自的面积
    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)
 
    # 计算相交部分矩形面积
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)
 
    w = max(0, xmax - xmin)
    h = max(0, ymax - ymin)
    area = w * h
    iou = area / (s1 + s2 - area)
    return iou
