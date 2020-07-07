import numpy as np
import ipdb

# 获取积分图
def get_integral(img):
    # 创建空积分图
    integimg = np.zeros(shape=(img.shape[0] + 1, img.shape[1] + 1), dtype=np.int32)
    for i in range(1, img.shape[0]):
        for j in range(1, img.shape[1]):
            integimg[i][j] = img[i][j] + integimg[i - 1][j] + integimg[i][j - 1] - integimg[i - 1][j - 1]
    return integimg


# 获取单一尺度下X2类型的haar特征
def haar_onescale(img, integimg, varNormFactor, haarblock_width, haarblock_height):
    haarimg = np.zeros(shape=(img.shape[0] - haarblock_width + 1, img.shape[1] - haarblock_height + 1), dtype=np.int32)
    haar_feature_onescale = []
    for i in range(haarimg.shape[0]):
        for j in range(haarimg.shape[1]):
            # i,j映射回原图形的坐标
            m = haarblock_width + i
            n = haarblock_height + i

            # 根据积分图计算Haar区域内像素值之和
            haar_all = integimg[m][n] - integimg[m - haarblock_width][n] - integimg[m][n - haarblock_height] + \
                       integimg[m - haarblock_width][n - haarblock_height]

            # 根据积分图计算黑色部分像素值之和
            haar_black = integimg[m][n - int(haarblock_height / 2)] - integimg[m - haarblock_width][
                n - int(haarblock_height / 2)] - integimg[m][n - haarblock_height] + integimg[m - haarblock_width][
                             n - haarblock_height]

            # 对于X2类型的haar特征，特征值 = block内所有像素和 - 两倍黑色区域像素和
            haarimg[i][j] = (1 * haar_all - 2 * haar_black) / varNormFactor
            haar_feature_onescale.append(haarimg[i][j])

    return haar_feature_onescale

# 获取全尺度下的X2类型的Haar特征
def harr(img, haarblock_w=10, haarblock_h=10):
    img = img.astype(np.int32)

    integimg = get_integral(img)

    # 计算两个方向上haarblock的最高缩放比例
    width_max_scale = int(img.shape[0] / haarblock_w)
    height_max_scale = int(img.shape[1] / haarblock_h)
    # 选取较小者作为haarblock的最高缩放比例
    max_scale = min(height_max_scale, width_max_scale)

    sum = img.sum()
    sqsum = (img*img).sum()
    mean = sum / (img.shape[0] * img.shape[1])
    sqmean = sqsum / (img.shape[0] * img.shape[1])
    varNormFactor = np.sqrt(sqmean - np.square(mean))

    feature = []
    haar_num = 0
    for i in range(max_scale):
        haarblock_w = i * haarblock_w + 10
        haarblock_h = i * haarblock_h + 10
        haar_feature_onescale = haar_onescale(img, integimg, varNormFactor, haarblock_w, haarblock_h)
        haar_num += len(haar_feature_onescale)
        feature.extend(haar_feature_onescale)
        haarblock_w = 10
        haarblock_h = 10

    return feature