import torch
import torchvision
import matplotlib.pyplot as plt
import random

# 设置随机种子以确保结果可重现
torch.manual_seed(42)

# 加载MNIST训练数据集
train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True,
                                           transform=torchvision.transforms.ToTensor())

# 创建3x4的子图
fig, axes = plt.subplots(3, 4, figsize=(10, 8))

# 随机选择12个样本
random_indices = random.sample(range(len(train_dataset)), 12)

# 绘制每个子图
for i, ax in enumerate(axes.flat):
    idx = random_indices[i]
    image, label = train_dataset[idx]

    # 将张量转换为numpy数组并去除通道维度
    image = image.squeeze().numpy()

    # 显示图像
    ax.imshow(image, cmap='gray')
    ax.set_title(f'Label: {label}')
    ax.axis('off')

# 调整布局并显示
plt.tight_layout()
plt.show()