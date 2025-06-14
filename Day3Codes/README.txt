关于激活函数

实验性质主要是测试和可视化一个简单的神经网络激活函数的效果，具体来说是使用Sigmoid激活函数。以下是对代码的详细分析：

数据集加载：

使用 torchvision.datasets.CIFAR10 加载CIFAR-10数据集的测试集。
数据集中的图像被转换为 tensor 格式。
数据加载器：

使用 DataLoader 将数据集分为批次，每个批次包含64个图像。
输入数据设置：

创建一个简单的 tensor，形状为 (2, 2)。
将这个 tensor 重塑为 (1, 1, 2, 2)，以匹配模型的输入格式。
模型定义：

定义了一个名为 Chen 的简单神经网络类。
这个类中包含两个激活函数：ReLU和Sigmoid。
在 forward 方法中，使用Sigmoid激活函数对输入进行处理。
TensorBoard可视化：

使用 SummaryWriter 创建一个日志记录器，用于记录和可视化数据。
遍历数据加载器中的每个批次数据。
将输入图像和经过Sigmoid激活函数处理后的图像添加到TensorBoard中，以便进行可视化比较。
模型推理：

使用自定义的输入数据进行模型推理。
打印模型对自定义输入数据的输出结果。
实验性质总结
这个实验的主要目的是：

测试Sigmoid激活函数：通过自定义的输入数据来测试Sigmoid激活函数的效果。
可视化激活函数的影响：使用TensorBoard来可视化输入图像和经过Sigmoid激活函数处理后的图像，以便观察激活函数对图像的影响。
验证模型的正确性：通过简单的实验来验证模型的定义和推理过程是否正确。
具体实验步骤
加载数据集：加载CIFAR-10测试集。
设置输入数据：创建一个简单的 tensor 作为输入数据。
定义模型：定义一个包含Sigmoid激活函数的简单模型。
可视化：使用TensorBoard记录并可视化输入图像和输出图像。
推理：使用自定义输入数据进行模型推理，并打印结果。
改进建议
如果想要更详细地观察Sigmoid和ReLU激活函数的效果，可以考虑以下改进：

添加ReLU激活函数的可视化：

在TensorBoard中分别记录和可视化经过Sigmoid和ReLU激活函数处理后的图像。
使用更多的激活函数：

可以尝试使用其他激活函数（如Tanh、Leaky ReLU等）并进行比较。
调整输入数据：

可以使用更复杂的输入数据，如真实图像的部分区域，来观察激活函数的实际效果。
记录更多指标：

可以记录模型的输出张量的形状、值等，以便更好地理解激活函数的作用。




关于数据集分割

实验性质主要是用于将一个数据集划分成训练集和验证集，并将这些划分后的图片移动到相应的目录中。具体来说，这个实验的目的是：

数据集划分：

将数据集按类别划分，并根据指定的比例（例如70%用于训练，30%用于验证）将图片分配到训练集和验证集中。
目录结构整理：

创建专门的训练集和验证集目录，并将每个类别的图片分别移动到这些目录中，从而形成一个结构化的数据集。
确保可重复性：

通过设置随机种子（random.seed(42)），确保每次运行代码时，训练集和验证集的划分结果是相同的，便于实验的可重复性和比较。
具体步骤解析
导入必要的库：

os 和 shutil 用于文件和目录操作。
train_test_split 用于将数据集划分成训练集和验证集。
random 用于随机划分数据集。
设置随机种子：

random.seed(42) 确保每次运行代码时划分结果相同。
定义数据集路径：

dataset_dir 是原始数据集的路径。
train_dir 和 val_dir 是划分后的训练集和验证集的输出路径。
创建输出目录：

使用 os.makedirs 创建训练集和验证集的目录，如果目录已经存在则不会报错（exist_ok=True）。
遍历类别文件夹：

遍历 dataset_dir 中的每个文件夹，确保这些文件夹不是 train 或 val 目录。
对于每个类别文件夹，获取其中的所有图片文件（支持 .jpg, .jpeg, .png 格式）。
确保图片路径包含类别文件夹：

将图片路径调整为相对路径，包含类别文件夹名称，以便后续操作。
划分训练集和验证集：

使用 train_test_split 函数将图片按指定比例划分成训练集和验证集。
创建类别子文件夹：

在训练集和验证集的输出路径中，为每个类别创建子文件夹。
移动图片到相应目录：

将训练集图片移动到 train_dir 中的相应类别子文件夹。
将验证集图片移动到 val_dir 中的相应类别子文件夹。
删除原始类别文件夹：

使用 shutil.rmtree 删除原始数据集中已经移动到训练集和验证集的类别文件夹。
注意事项
路径问题：

确保路径字符串中的反斜杠一致。可以使用双反斜杠（\\）或原始字符串（在字符串前加 r）。
文件和目录权限：

确保你有权限访问和修改指定的路径和文件。
图片路径调整：

在将图片路径调整为相对路径时，确保路径正确。例如，images = [os.path.join(class_name, img) for img in images] 这一步将图片路径调整为相对于 dataset_dir 的路径。
删除原始类别文件夹：

shutil.rmtree(class_path) 这一步会删除原始数据集中已经移动到训练集和验证集的类别文件夹。请确保这是你想要的操作，否则可能会丢失数据。



关于prepare.py
实验性质主要是用于生成两个文本文件，分别记录训练集和验证集中的图片路径及其对应的类别标签。具体来说，这个实验的目的是：

生成文本文件：

创建 train.txt 文件，记录训练集中每个图片的路径及其类别标签。
创建 val.txt 文件，记录验证集中每个图片的路径及其类别标签。
组织数据集：

将数据集中的图片路径和类别标签整理成文本文件的形式，方便后续的模型训练和验证过程。
便于数据加载：

生成的文本文件可以被用于数据加载器（如在深度学习框架中），使得数据加载更加方便和高效。
具体步骤解析
导入必要的库：

os 用于文件和目录操作。
定义函数 create_txt_file：

该函数接受两个参数：root_dir（数据集的根目录路径）和 txt_filename（要生成的文本文件名）。
打开指定的文本文件并写入图片路径及其类别标签。
遍历类别文件夹：

使用 os.listdir(root_dir) 遍历 root_dir 中的每个文件夹。
使用 enumerate 为每个类别文件夹分配一个标签（label）。
检查是否为目录：

使用 os.path.isdir(category_path) 检查每个文件夹是否为目录。
遍历图片文件：

遍历每个类别文件夹中的所有图片文件。
使用 os.path.join(category_path, img_name) 构建完整的图片路径。
写入文本文件：

将每个图片的路径及其类别标签写入文本文件中，格式为 图片路径 类别标签。
调用函数：

调用 create_txt_file 函数生成 train.txt 文件，记录训练集中的图片路径及其类别标签。
调用 create_txt_file 函数生成 val.txt 文件，记录验证集中的图片路径及其类别标签。











