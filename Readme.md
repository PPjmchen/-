# 人脸检测识别系统

## 所需第三方库

2. tqdm：进度条插件
3. selectivesearch：用于检测部分从原图获取一系列子窗口
4. opencv-python：仅用于图像读取和基本处理（转灰度图、缩放、剪切等）
5. jupyter：完整功能演示

## 程序说明

**所有预设参数（人脸缩放尺寸、SVM训练迭代次数等）都存放于`tools/config.py`**

1. 对原始数据集进行数据整理，保存结果于**`dataset.json`**

```bash
python tools/data2json.py
```

2. 从数据集中选取80%训练集和20%测试集，并生成检测任务所需的正负样本，保存于**`jsons/train_sample.json`**和**`jsons/test_sample.json`**

```python
python tools/getAllSample.py
```

3. 检测部分模型训练

```bash
python train_detector.py
```

4. 检测部分模型测试

```bash
python test_detector.py
```

5. 识别部分模型训练及测试

```bash
python recognition.py
```

6. 完整功能演示（jupyter-notebook）

```python
demonstration.ipynb
```

