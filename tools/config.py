from pprint import pprint

class Config:

    # original image size
    w = 600
    h = 800

    # svm params
    Max_Interation = 5000

    # train detector params
    crop_resize = (20, 20)
    train_proportion = 0.8
    iou_threshold = 0.4
    window_min_size = 5000
    window_max_size = 24000

    # train recog params
    pca_train_num = 5   # pca训练集每类个数
    face_size = (100, 100)

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
