import torch
from torch import nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import time
import os
from torch.utils.tensorboard import SummaryWriter


class CustomAlexNet(nn.Module):
    def __init__(self, num_classes=10):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 192, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((3, 3))
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 3 * 3, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


if __name__ == '__main__':
    # 准备数据集
    train_data = datasets.CIFAR10(root="../dataset_chen",
                                  train=True,
                                  transform=transforms.ToTensor(),
                                  download=True)

    test_data = datasets.CIFAR10(root="../dataset_chen",
                                 train=False,
                                 transform=transforms.ToTensor(),
                                 download=True)

    # 数据集长度
    train_data_size = len(train_data)
    test_data_size = len(test_data)
    print(f"训练数据集的长度: {train_data_size}")
    print(f"测试数据集的长度: {test_data_size}")

    # 加载数据集
    train_loader = DataLoader(train_data, batch_size=64, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=64, shuffle=False)

    # 创建网络模型
    chen = CustomAlexNet()
    if torch.cuda.is_available():
        chen = chen.cuda()

    # 创建损失函数
    loss_fn = nn.CrossEntropyLoss()
    if torch.cuda.is_available():
        loss_fn = loss_fn.cuda()

    # 优化器
    learning_rate = 0.01
    optim = torch.optim.SGD(chen.parameters(), lr=learning_rate, momentum=0.9)

    # 设置训练网络的一些参数
    total_train_step = 0  # 记录训练的次数
    total_test_step = 0   # 记录测试的次数
    epoch = 10            # 训练的轮数

    # 添加tensorboard
    writer = SummaryWriter("../logs_train")

    # 添加开始时间
    start_time = time.time()

    for i in range(epoch):
        print(f"-----第{i+1}轮训练开始-----")
        # 训练步骤
        for data in train_loader:
            imgs, targets = data
            if torch.cuda.is_available():
                imgs = imgs.cuda()
                targets = targets.cuda()
            outputs = chen(imgs)
            loss = loss_fn(outputs, targets)

            # 优化器优化模型
            optim.zero_grad()  # 梯度清零
            loss.backward()    # 反向传播
            optim.step()

            total_train_step += 1
            if total_train_step % 500 == 0:
                print(f"第{total_train_step}次训练的loss: {loss.item()}")
                writer.add_scalar("train_loss", loss.item(), total_train_step)

        # 测试步骤（以测试数据上的正确率来评估模型）
        total_test_loss = 0.0
        # 整体正确个数
        total_accuracy = 0
        with torch.no_grad():
            for data in test_loader:
                imgs, targets = data
                if torch.cuda.is_available():
                    imgs = imgs.cuda()
                    targets = targets.cuda()
                outputs = chen(imgs)
                # 损失
                loss = loss_fn(outputs, targets)
                total_test_loss += loss.item()
                # 正确率
                accuracy = (outputs.argmax(1) == targets).sum()
                total_accuracy += accuracy

        print(f"整体测试集上的loss: {total_test_loss}")
        print(f"整体测试集上的正确率: {total_accuracy / test_data_size}")
        writer.add_scalar("test_loss", total_test_loss, total_test_step)
        writer.add_scalar("test_accuracy", total_accuracy, total_test_step)
        total_test_step += 1

        # 保存每一轮训练模型
        model_save_path = os.path.join("E:\实训\训练代码\Day2\model_save\Torchnn\model_save", f"chen_{i}.pth")
        torch.save(chen, model_save_path)
        print("模型已保存")

    writer.close()
