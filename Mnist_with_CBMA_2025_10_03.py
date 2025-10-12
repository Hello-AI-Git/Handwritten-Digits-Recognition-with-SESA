import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time


# 定义CBAM注意力模块
class CBAM(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(CBAM, self).__init__()
        # 通道注意力
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction_ratio, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction_ratio, channels, kernel_size=1),
            nn.Sigmoid()
        )

        # 空间注意力
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # 通道注意力
        channel_att = self.channel_attention(x)
        x_channel = x * channel_att

        # 空间注意力
        avg_out = torch.mean(x_channel, dim=1, keepdim=True)
        max_out, _ = torch.max(x_channel, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_att)

        return x * spatial_att


# 定义CNN+CBAM网络
class CNN_CBAM(nn.Module):
    def __init__(self):
        super(CNN_CBAM, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.cbam1 = CBAM(32)

        # 卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.cbam2 = CBAM(64)
        self.pool1 = nn.MaxPool2d(2)

        # 卷积层3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.cbam3 = CBAM(128)

        # 卷积层4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.cbam4 = CBAM(256)
        self.pool2 = nn.MaxPool2d(2)

        # 全连接层
        self.fc = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.cbam1(x)

        x = self.conv2(x)
        x = self.cbam2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.cbam3(x)

        x = self.conv4(x)
        x = self.cbam4(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# 训练函数
def train(model, device, train_loader, optimizer, epoch):
    model.train()
    train_loss = 0
    correct = 0
    processed = 0

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        pred = output.argmax(dim=1, keepdim=True)
        correct += pred.eq(target.view_as(pred)).sum().item()
        processed += len(data)

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{processed}/{len(train_loader.dataset)} '
                  f'({100. * processed / len(train_loader.dataset):.0f}%)]\t'
                  f'Loss: {loss.item():.6f}')

    train_loss /= len(train_loader)
    accuracy = 100. * correct / len(train_loader.dataset)
    print(
        f'\nTrain set: Average loss: {train_loss:.4f}, Accuracy: {correct}/{len(train_loader.dataset)} ({accuracy:.2f}%)\n')
    return train_loss, accuracy


# 测试函数（添加混淆矩阵数据收集）
def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    all_targets = []
    all_predictions = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

            # 收集预测结果和真实标签用于混淆矩阵
            all_predictions.extend(pred.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    print(
        f'Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)\n')

    return test_loss, accuracy, all_targets, all_predictions


# 手动计算混淆矩阵
def manual_confusion_matrix(targets, predictions, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(targets, predictions):
        cm[t][p] += 1
    return cm


# 绘制混淆矩阵（纯Matplotlib实现）
def plot_confusion_matrix(targets, predictions, classes):
    cm = manual_confusion_matrix(targets, predictions)

    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()

    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes)
    plt.yticks(tick_marks, classes)

    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')

    # 在格子中添加数值
    thresh = cm.max() / 2.
    for i in range(len(classes)):
        for j in range(len(classes)):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.savefig('confusion_matrix.png')
    plt.show()
    return cm


# 可视化错误样本
def visualize_errors(model, device, test_loader, classes):
    model.eval()
    errors = []

    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1)

            for i in range(len(target)):
                if pred[i] != target[i]:
                    errors.append({
                        'image': data[i].cpu().squeeze(),
                        'true': target[i].cpu().item(),
                        'pred': pred[i].cpu().item()
                    })
            if len(errors) >= 25:  # 收集25个错误样本
                break

    # 绘制错误样本
    plt.figure(figsize=(12, 10))
    for i in range(min(25, len(errors))):
        plt.subplot(5, 5, i + 1)
        plt.imshow(errors[i]['image'], cmap='gray')
        plt.title(f'T:{errors[i]["true"]}, P:{errors[i]["pred"]}')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('error_examples.png')
    plt.show()


# 主函数
def main():
    # 超参数设置
    batch_size = 64
    epochs = 10
    lr = 0.001

    # 设备配置
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 数据加载
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = datasets.MNIST('./data', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST('./data', train=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 初始化模型
    model = CNN_CBAM().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 训练过程记录
    train_losses, train_accs = [], []
    test_losses, test_accs = [], []

    start_time = time.time()

    # 训练循环
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, epoch)
        test_loss, test_acc, all_targets, all_predictions = test(model, device, test_loader)

        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_losses.append(test_loss)
        test_accs.append(test_acc)

    training_time = time.time() - start_time
    print(f"Total training time: {training_time:.2f} seconds")

    # 保存模型
    torch.save(model.state_dict(), "mnist_cbam_cnn.pth")
    print("Model saved to mnist_cbam_cnn.pth")

    # 绘制训练曲线
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(test_losses, label='Test Loss')
    plt.title('Loss Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy')
    plt.plot(test_accs, label='Test Accuracy')
    plt.title('Accuracy Curve')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.tight_layout()
    plt.savefig('training_results.png')
    plt.show()

    # 绘制混淆矩阵
    classes = [str(i) for i in range(10)]
    cm = plot_confusion_matrix(all_targets, all_predictions, classes)

    # 打印分类报告
    print("Classification Report:")
    print(f"{'Class':<5} {'Precision':<10} {'Recall':<10} {'F1-score':<10}")
    for i in range(10):
        tp = cm[i, i]
        fp = np.sum(cm[:, i]) - tp
        fn = np.sum(cm[i, :]) - tp
        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        print(f"{i:<5} {precision:.4f}     {recall:.4f}     {f1:.4f}")

    # 可视化错误样本
    visualize_errors(model, device, test_loader, classes)


if __name__ == '__main__':
    main()