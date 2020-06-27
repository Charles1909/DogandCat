from config import opt
import os
import torch
import models
from data.dataset import DogCat
from torch.utils.data import DataLoader
from torchnet import meter  # 用于帮助用户快速统计训练过程中的一些指标
from utils.visualize import Visualizer
from tqdm import tqdm  # 使代码进度可视化
import  matplotlib as plt


@torch.no_grad()
def test(**kwargs):
    opt._parse(kwargs)

    # 模型
    model = getattr(models, opt.model)().eval()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # 数据
    test_data = DogCat(opt.test_data_root, test=True)
    test_dataloader = DataLoader(test_data, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers)
    results = []
    for ii, (data, path) in tqdm(enumerate(test_dataloader)):
        input = data.to(opt.device)
        score = model(input)
        probability = torch.nn.functional.softmax(score, dim=1)[:, 0].detach().tolist()
        # label = score.max(dim = 1)[1].detach().tolist()

        batch_results = [(path_.item(), probability_) for path_, probability_ in zip(path, probability)]

        results += batch_results
    write_csv(results, opt.result_file)

    return results


def write_csv(results, file_name):
    import csv
    with open(file_name, 'w') as f:
        writer = csv.writer(f)
        writer.writerow(['id', 'label'])
        writer.writerows(results)


'''
训练的主要步骤：
    1、定义网络
    2、定义数据
    3、定义损失函数和优化器
    4、计算重要指标
    5、开始训练
        （1）训练网络
        （2）可视化各种指标
        （3）计算在验证集上的指标
'''

def train(**kwargs):
    # 根据命令行参数更新配置
    opt._parse(kwargs)
    vis = Visualizer(opt.env, port=opt.vis_port)

    # step1: 加载模型
    model = getattr(models, opt.model)()  # 使用字符串直接指定使用的模型
    #model = models.SqueezeNet()
    if opt.load_model_path:
        model.load(opt.load_model_path)
    model.to(opt.device)

    # step2: 加载数据
    train_data = DogCat(opt.train_data_root, train=True)  # 训练集
    train_dataloader = DataLoader(train_data, opt.batch_size,
                                  shuffle=True, num_workers=opt.num_workers)

    val_data = DogCat(opt.train_data_root, train=False)  #交叉验证
    val_dataloader = DataLoader(val_data, opt.batch_size,
                                shuffle=False, num_workers=opt.num_workers)

    # step3: 目标函数和优化器
    criterion = torch.nn.CrossEntropyLoss()  # 交叉损失函数
    lr = opt.lr  # learn rate
    optimizer = model.get_optimizer(lr, opt.weight_decay)

    # step4: 统计指标：平滑处理之后的损失，还有混淆矩阵
    loss_meter = meter.AverageValueMeter()
    confusion_matrix = meter.ConfusionMeter(2)  # 统计分类问题中的分类情况
    previous_loss = 1e10

    # 训练
    indices=[]
    losses=[]
    accs=[]
    for epoch in range(opt.max_epoch):

        # 重置损失和混淆矩阵
        loss_meter.reset()
        confusion_matrix.reset()

        for ii, (data, label) in tqdm(enumerate(train_dataloader)):

            # 训练模型参数
            input_batch = data.to(opt.device)
            target_batch = label.to(opt.device)

            optimizer.zero_grad()  # 梯度清零
            score = model(input_batch)
            loss = criterion(score, target_batch)
            loss.backward()  # 反向传播
            optimizer.step()  # 优化

            # 跟新统计指标以及可视化
            loss_meter.add(loss.item())
            losses.append(loss.item())
            indices.append(ii)
            confusion_matrix.add(score.detach(), target_batch.detach())

            if (ii + 1) % opt.print_freq == 0:
                # 可视化
                #vis.plot('loss', loss_meter.value()[0])


                print('epoch:', epoch, '  loss:  ', loss_meter.value()[0])

                # 进入debug模式
                if os.path.exists(opt.debug_file):
                    import ipdb
                    ipdb.set_trace()

        model.save(epoch=epoch)

        # 计算验证集上的指标及可视化
        val_cm, val_accuracy = val(model, val_dataloader)
        accs.append(val_accuracy)
        plt.plot(indices, losses)
        plt.plot(indices, accs)
        plt.show()

        # vis.plot('val_accuracy', val_accuracy)
        # vis.log("epoch:{epoch},lr:{lr},loss:{loss},train_cm:{train_cm},val_cm:{val_cm}".format(
        #     epoch=epoch, loss=loss_meter.value()[0], val_cm=str(val_cm.value()), train_cm=str(confusion_matrix.value()),
        #     # lr=lr))

        # 更新学习率
        if loss_meter.value()[0] > previous_loss:
            lr = lr * opt.lr_decay
            # 第二种降低学习率的方法:不会有moment等信息的丢失
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

        previous_loss = loss_meter.value()[0]

        print('第', str(epoch),'个迭代已结束')
        print('验证集准确率为：', str(val_accuracy))

@torch.no_grad()
def val(model, dataloader):
    """
    计算模型在验证集上的准确率等信息
    """
    # 把模型设为验证模式
    model.eval()

    confusion_matrix = meter.ConfusionMeter(2)
    for ii, (val_input, label) in tqdm(enumerate(dataloader)):
        val_input = val_input.to(opt.device)
        score = model(val_input)
        confusion_matrix.add(score.detach().squeeze(), label.type(torch.LongTensor))

    # 将模型恢复为训练模式
    model.train()

    cm_value = confusion_matrix.value()
    accuracy = 100. * (cm_value[0][0] + cm_value[1][1]) / (cm_value.sum())
    return confusion_matrix, accuracy


def help():
    """
    打印帮助的信息： python file.py help
    """

    print("""
    usage : python file.py <function> [--args=value]
    <function> := train | test | help
    example: 
            python {0} train --env='env0701' --lr=0.01
            python {0} test --dataset='path/to/dataset/root/'
            python {0} help
    avaiable args:""".format(__file__))

    from inspect import getsource
    source = (getsource(opt.__class__))
    print(source)


if __name__ == '__main__':
    # //test
    train()



