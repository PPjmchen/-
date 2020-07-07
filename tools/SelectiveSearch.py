import selectivesearch
import cv2
from config import opt
from iou import cal_iou

def windows_select(img):
    img_lbl, regions = selectivesearch.selective_search(img, scale=opt.scale, sigma=opt.sigma, min_size=opt.min_size)
    return img_lbl, regions

def get_windows(img):
    label, regions = windows_select(img)

    effectiveWindows = []

    for r in regions:

        # 去除超过原图边界的子窗口
        if r['rect'][0]+r['rect'][2] >= opt.w or r['rect'][1]+r['rect'][3] >= opt.h:
            continue
        # 去除重复的子窗口
        if r['rect'] in effectiveWindows:
            continue
        # 去除面积过小或过大的子窗口
        if r['size'] < opt.window_min_size or r['size'] > opt.window_max_size:
            continue      

        effectiveWindows.append(r['rect'])

    return effectiveWindows

# 绘制Selective Search提取结果demo
def main():

    # 读取demo图片
    test_img = 'dataset/JohnMayer/JohnMayer_12.png'
    test_img = cv2.imread(test_img)

    # ground truth bbox
    gt = [231, 101, 128, 169]

    # Select Search
    _ , regions = windows_select(test_img)

    effectiveWindows = []

    for r in regions:
        # 跳过重复的子窗口
        if r['rect'] in effectiveWindows:
            continue
        # 跳过超出边界的子窗口
        if r['size'] < opt.window_min_size or r['size'] > opt.window_max_size:
            continue

        effectiveWindows.append(r['rect'])

    for w in effectiveWindows:

        # iou >= 0.4的认为是正样本（红色框），否则为负样本（黄色框）
        if cal_iou(w, gt) >= 0.4:

            cv2.rectangle(test_img, (w[0], w[1]),
                                    (w[0]+w[2], w[1]+w[3]),
                                    (0, 0, 255), 3)
        else:
            cv2.rectangle(test_img, (w[0], w[1]),
                          (w[0] + w[2], w[1] + w[3]),
                          (0, 255, 255), 3)

    # 绘制ground bbox（蓝色框）
    cv2.rectangle(test_img, (gt[0], gt[1]),
                            (gt[0]+gt[2], gt[1]+gt[3]),
                            (255, 0, 0), 3)
    # 保存demo结果
    cv2.imwrite('test_imgs/test_img.jpg',test_img)

if __name__ == '__main__':
    main()

