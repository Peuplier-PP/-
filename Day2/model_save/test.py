import torch
import torchvision
from PIL import Image
from model import *
from torchvision import transforms

image_path = "E:\实训\训练代码\Day2\model_save\QQ图片20221202171742.jpg"
image = Image.open(image_path)
print(image)
# png格式是四个通道，除了RGB三个通道外，还有一个透明通道,调用image = image.convert('RGB')
image = image.convert('RGB')

# 改变成tensor格式
trans_reszie = transforms.Resize((32, 32))
trans_totensor = transforms.ToTensor()
transform = transforms.Compose([trans_reszie, trans_totensor])
image = transform(image)
print(image.shape)

# 加载训练模型
model = torch.load("model_save\\chen_9.pth").to("cuda")

# print(model)

image = torch.reshape(image, (1, 3, 32, 32)).to("cuda")
# image = image.cuda()

# 将模型转换为测试模型
model.eval()
with torch.no_grad():
    output = model(image)
# print(output)

print(output.argmax(1))
