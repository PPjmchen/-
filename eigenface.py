import numpy as np
import ipdb

class Eigenface():
    def __init__(self):
        self.eigenfaces_matrix = None
        self.variance = None
        self.mean_image = None

    # 输入某一类人脸数据，得到其映射到eigenface空间
    def project_image(self, sub_dataset):
        sub_dataset = sub_dataset - self.mean_image
        return np.dot(sub_dataset, self.eigenfaces_matrix.T)

    def pca(self, train_set, class_train_set):

        # 获取训练集样本数、样本维度
        num, dim = train_set.shape

        # 计算样本均值，即平均脸
        mean = train_set.mean(axis=0)

        # 计算所有样本与样本均值的差
        train_set = train_set - mean    # shape = (50, 10000)
        # 当样本向量维度远大于样本数量时，使用compact trick
        if dim > num:
            # PCA 使用compact trick，得到数量与样本数一致的特征向量
            covariance_matrix = np.dot(train_set, train_set.T)  # 50*50的协方差矩阵

            eigen_value, eigen_vactor = np.linalg.eigh(covariance_matrix)
            tmp = np.dot(train_set.T, eigen_vactor).T   # 利用compact trick计算原协方差矩阵特征向量

            # 由于eigen_value为从小到大排序，因此将特征值与特征向量反转，得到最终的主成分（Eigenface）
            V = tmp[::-1]
            S = np.sqrt(eigen_value[::-1])

            # 将特征向量化为单位特征向量
            for i in range(V.shape[1]):
                V[:, i] /= S
        else:
            # PCA - SVD used
            U, S, V = np.linalg.svd(train_set)
            V = V[:num]  # only makes sense to return the first num_data

        # 将特征向量、特征值、训练集均值存于类变量，用于预测
        self.eigenfaces_matrix = V  # 每个特征向量也称为一个Eigenface
        self.variance = S
        self.mean_image = mean

        # 将每一类人脸子数据集映射到eigenface表示空间上
        self.projected_classes = []
        for class_sample in class_train_set:
            class_weights = self.project_image(class_sample)
            self.projected_classes.append(class_weights.mean(0))
        return None

    # 由于已经将每一类人脸映射到Eigenface空间得到单一的向量表示，因此KNN取K=1, 即最近邻
    def KNN_predicgt(self, X , labels_list):
        predict_class = -1
        # 取无穷大初始化最小距离
        min_distance = np.finfo('float').max
        # 求测试人脸的Eigenface表示
        eigen_test = self.project_image(X)
        eigen_test = np.delete(eigen_test, -1)  # 最后一行为nan值，删去

        # 与训练集中每一类特征脸求L2范数，并以与其最小距离的类别作为预测类别
        for i in range(len(self.projected_classes)):
            L2_distance = np.linalg.norm(eigen_test - np.delete(self.projected_classes[i], -1))
            if L2_distance < min_distance:
                min_distance = L2_distance
                predict_class = labels_list[i]
        return predict_class