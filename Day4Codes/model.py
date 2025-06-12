# 搭建神经网络
import torch
from torch import nn
import torch.nn.functional as F


# class Chen(nn.Module):
#     def __init__(self):
#         super().__init__()
#         self.model = nn.Sequential(
#             nn.Conv2d(3, 32, 5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 32, 5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Conv2d(32, 64, 5, padding=2),
#             nn.MaxPool2d(kernel_size=2),
#             nn.Flatten(),
#             nn.Linear(1024, 64),
#             nn.Linear(64, 10)
#         )


class Chen(nn.Module):
    def __init__(self):
        super(Chen, self).__init__()
        # 假设你的模型有一个卷积层
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1)
        # 假设你的模型有一个全连接层
        self.fc1 = nn.Linear(16 * 16 * 16, 100)  # 修改为100类

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        # 展平操作
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x



    # def forward(self,x):
    #     x = self.model(x)
    #     return x

if __name__ == '__main__':
    chen = Chen()
    input = torch.ones((64,3,32,32))
    output = chen(input)
    print(output.shape)
