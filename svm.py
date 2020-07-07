from tqdm import tqdm
import numpy as np
from tools.config import opt
class SVM(object):

    def __init__(self, kernel='linear', epsilon=0.001, C=1, Max_Interation=opt.Max_Interation):
        self.kernel = kernel
        self.epsilon = epsilon
        self.C = C
        self.Max_Interation = Max_Interation

    def _init_parameters(self, features, labels):
        print('svm initializing......')
        # 参数初始化
        self.X = features
        self.Y = labels

        self.b = 0.0
        self.n = len(features[0])
        self.N = len(features)
        self.alpha = [0.0] * self.N
        self.E = [self._E_(i) for i in tqdm(range(self.N))]

    def _satisfy_KKT(self, i):
        ygx = self.Y[i] * self._g_(i)
        if abs(self.alpha[i]) < self.epsilon:
            return ygx >= 1
        elif abs(self.alpha[i] - self.C) < self.epsilon:
            return ygx <= 1
        else:
            return abs(ygx - 1) < self.epsilon

    def is_stop(self):
        for i in range(self.N):
            satisfy = self._satisfy_KKT(i)
            if not satisfy:
                return False
        return True

    def _select_two_parameters(self):
        index_list = [i for i in range(self.N)]

        i1_list_1 = list(filter(lambda i: self.alpha[i] > 0 and self.alpha[i] < self.C, index_list))
        i1_list_2 = list(set(index_list) - set(i1_list_1))

        i1_list = i1_list_1
        i1_list.extend(i1_list_2)

        for i in i1_list:
            if self._satisfy_KKT(i):
                continue

            E1 = self.E[i]
            max_ = (0, 0)

            for j in index_list:
                if i == j:
                    continue

                E2 = self.E[j]
                if abs(E1 - E2) > max_[0]:
                    max_ = (abs(E1 - E2), j)
            return i, max_[1]

    def _K_(self, x1, x2):
        if self.kernel == 'linear':
            return sum([np.float(x1[k]) * np.float(x2[k]) for k in range(self.n)])
        if self.kernel == 'poly':
            return (sum([x1[k] * x2[k] for k in range(self.n)]) + 1) ** 3
        return 0

    def _g_(self, i):
        result = self.b
        for j in range(self.N):
            result += self.alpha[j] * self.Y[j] * self._K_(self.X[i], self.X[j])
        return result

    def _E_(self, i):
        return self._g_(i) - self.Y[i]

    def clip_LH(self, _alpha, L, H):
        if _alpha > H:
            return H
        elif _alpha < L:
            return L
        else:
            return _alpha

    def thresh_LH(self, i1, i2):
        if self.Y[i1] == self.Y[i2]:
            L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
            H = min(self.C, self.alpha[i2] + self.alpha[i1])
        else:
            L = max(0, self.alpha[i2] - self.alpha[i1])
            H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])
        return L, H

    def train(self, features, labels):

        self._init_parameters(features, labels)
        print("training...")
        for times in tqdm(range(self.Max_Interation)):

            i1, i2 = self._select_two_parameters()

            L = max(0, self.alpha[i2] - self.alpha[i1])
            H = min(self.C, self.C + self.alpha[i2] - self.alpha[i1])

            if self.Y[i1] == self.Y[i2]:
                L = max(0, self.alpha[i2] + self.alpha[i1] - self.C)
                H = min(self.C, self.alpha[i2] + self.alpha[i1])

            E1 = self.E[i1]
            E2 = self.E[i2]
            eta = self._K_(self.X[i1], self.X[i1]) + self._K_(self.X[i2], self.X[i2]) - 2 * self._K_(self.X[i1], self.X[i2])
            # 防止出现除数为零
            if eta == 0:
                eta = 1e-7
            alpha2_new_unc = self.alpha[i2] + self.Y[i2] * (E1 - E2) / eta
            alpha2_new = 0
            if alpha2_new_unc > H:
                alpha2_new = H
            elif alpha2_new_unc < L:
                alpha2_new = L
            else:
                alpha2_new = alpha2_new_unc

            alpha1_new = self.alpha[i1] + self.Y[i1] * \
                         self.Y[i2] * (self.alpha[i2] - alpha2_new)

            b1_new = -E1 - self.Y[i1] * self._K_(self.X[i1], self.X[i1]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self._K_(self.X[i2], self.X[i1]) * (alpha2_new - self.alpha[i2]) + self.b
            b2_new = -E2 - self.Y[i1] * self._K_(self.X[i1], self.X[i2]) * (alpha1_new - self.alpha[i1]) - self.Y[
                i2] * self._K_(self.X[i2], self.X[i2]) * (alpha2_new - self.alpha[i2]) + self.b

            if alpha1_new > 0 and alpha1_new < self.C:
                b_new = b1_new
            elif alpha2_new > 0 and alpha2_new < self.C:
                b_new = b2_new
            else:
                b_new = (b1_new + b2_new) / 2

            self.alpha[i1] = alpha1_new
            self.alpha[i2] = alpha2_new
            self.b = b_new

            self.E[i1] = self._E_(i1)
            self.E[i2] = self._E_(i2)
        return self.alpha, self.b

    def _predict_(self, feature):
        result = self.b

        for i in range(self.N):
            result += self.alpha[i] * self.Y[i] * self._K_(feature, self.X[i])

        if result > 0:
            return 1
        return -1

    def predict(self, features):
        results = []

        for feature in tqdm(features):
            results.append(self._predict_(feature))

        return results
