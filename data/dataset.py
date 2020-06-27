import os
from PIL import Image
from torch.utils import data
import numpy as np
from torchvision import transforms as T


class DogCat(data.Dataset):

    def __init__(self, root, transforms=None, train=True, test=False):
        '''
        目标：获取所有图片路径，并根据训练、验证、测试划分数据
        '''
        self.test = test
        imgs = [os.path.join(root, img) for img in os.listdir(root)]
        # 训练集和验证集的文件命名不一样
        # test1: data/test1/8973.jpg
        # train: data/train/cat.10004.jpg
        if self.test:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2].split('/')[-1]))
        else:
            imgs = sorted(imgs, key=lambda x: int(x.split('.')[-2]))

        imgs_num = len(imgs)

        # shuffle imgs
        np.random.seed(100)
        imgs = np.random.permutation(imgs)

        # 划分训练、验证集，训练集：验证集 = 7:3
        if self.test:
            self.imgs = imgs
        elif train:
            self.imgs = imgs[:int(0.7 * imgs_num)]
        else:
            self.imgs = imgs[int(0.7 * imgs_num):]

        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        if transforms is None:

            # 数据转换操作，测试验证和训练的数据转换有所区别
            normalize = T.Normalize(mean, std)  # 归一化过程

            # 测试集和验证集不用数据增强
            if self.test or not train:
                self.transforms = T.Compose([
                    T.Resize(224),
                    T.CenterCrop(224),  # 在图片的中间区域进行裁剪
                    T.ToTensor(),
                    normalize
                ])
                # 训练集需要数据增强
            else:
                self.transforms = T.Compose([    # transforms.Compose()函数实现对图片的增强处理
                    T.Resize(256),  # 将输入图片统一成给定大小
                    T.RandomResizedCrop(224),  # 将图片裁剪为任意大小和纵横比
                    T.RandomHorizontalFlip(),  # 随机水平翻转
                    T.ToTensor(),
                    normalize
                ])

    def __getitem__(self, index):
        '''
        返回一张图片的数据
        对于测试集，没有label，返回图片id，如1000.jpg返回1000
        '''
        img_path = self.imgs[index]
        if self.test:
            label = int(self.imgs[index].split('.')[-2].split('/')[-1])
        else:
            label = 1 if 'dog' in img_path.split('/')[-1] else 0
        data = Image.open(img_path)
        data = self.transforms(data)
        return data, label

    def __len__(self):
        '''
        返回数据集中所有图片的个数
        '''
        return len(self.imgs)