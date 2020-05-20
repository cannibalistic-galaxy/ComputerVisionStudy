#  Task1 零基础入门CV赛事——赛题理解

赛题名称：零基础入门CV之街道字符识别

训练集数据包括3W张照片，验证集数据包括1W张照片，每张照片包括颜色图像和对应的编码类别和具体位置；为了保证比赛的公平性，测试集A包括4W张照片，测试集B包括4W张照片。

需要注意的是本赛题需要选手识别图片中所有的字符，为了降低比赛难度，我们提供了训练集、验证集中所有字符的位置框。

对于训练数据每张图片将给出对于的编码标签，和具体的字符框的位置（训练集、验证集都给出字符位置），可用于模型训练：

| Field  | Description |
| ------ | ----------- |
| top    | 左上角坐标X |
| height | 字符高度    |
| left   | 左上角最表Y |
| width  | 字符宽度    |
| label  | 字符编码    |

字符的坐标具体如下所示：

![](E:\Github\GithubProject\ComputerVisionStudy\tianchi_cv_1\figure\1.PNG)

在比赛数据（训练集和验证集）中，同一张图片中可能包括一个或者多个字符，因此在比赛数据的JSON标注中，会有两个字符的边框信息：

![](E:\Github\GithubProject\ComputerVisionStudy\tianchi_cv_1\figure\2.PNG)

测评指标为：

选手提交结果与实际图片的编码进行对比，以编码整体识别准确率为评价指标。任何一个字符错误都为错误，最终评测指标结果越大越好，具体计算公式如下：
 Score=编码识别正确的数量/测试集图片数量

读取JSON文件中的标签的代码为：

```python
import json, os, sys, glob, shutil
import cv2
import numpy as np
import matplotlib.pyplot as plt

import torch
from torch.utils.data.dataset import Dataset
import torchvision.transforms as transforms

train_json = json.load(open('E:/PythonProgram/tianchiCV/input/mchar_train.json'))

# 数据标注处理
def parse_json(d):
   arr = np.array([
       d['top'], d['height'], d['left'],  d['width'], d['label']
   ])
   arr = arr.astype(int)
   return arr

img = cv2.imread('E:/PythonProgram/tianchiCV/input/mchar_train/mchar_train/000000.png')
arr = parse_json(train_json['000000.png'])

plt.figure(figsize=(10, 10))
plt.subplot(1, arr.shape[1]+1, 1)
plt.imshow(img)
plt.xticks([]); plt.yticks([])
plt.show()

for idx in range(arr.shape[1]):
   plt.subplot(1, arr.shape[1]+1, idx+2)
   plt.imshow(img[arr[0, idx]:arr[0, idx]+arr[1, idx],arr[2, idx]:arr[2, idx]+arr[3, idx]])
   plt.title(arr[4, idx])
   plt.xticks([]); plt.yticks([])
   plt.show()

```

得到结果为：

![](E:\Github\GithubProject\ComputerVisionStudy\tianchi_cv_1\figure\3.PNG)

![4](E:\Github\GithubProject\ComputerVisionStudy\tianchi_cv_1\figure\4.PNG)

![5](E:\Github\GithubProject\ComputerVisionStudy\tianchi_cv_1\figure\5.PNG)

解题思路将会在下一个Task一起写出。