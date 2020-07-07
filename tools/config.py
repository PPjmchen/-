from pprint import pprint

class Config:

    # original image size
    w = 600
    h = 800

    # train detector params
    Max_Interation = 5000   # SVM最大迭代次数
    crop_resize = (20, 20)  # 检测部分子窗口缩放尺寸
    train_proportion = 0.8  # 检测训练集数据占比
    iou_threshold = 0.4     # 检测正样本阈值
    window_min_size = 5000  # 有效子窗口最小面积
    window_max_size = 24000 # 有效子窗口最大面积
    haar_w = 10             # haar区域初始宽度
    haar_h = 10             # haar区域初始高度

    # train recog params
    pca_train_num = 5       # pca训练集每类个数
    face_size = (100, 100)  # 识别部分人脸区域缩放尺寸

    # Selective Search params
    scale = 100
    sigma = 0.9
    min_size = 10

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items() \
                if not k.startswith('_')}


opt = Config()
