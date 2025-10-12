import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import time
import cv2


# 定义SE注意力模块
class SE_Attention(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SE_Attention, self).__init__()
        # 全局平均池化
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # 两个全连接层构成瓶颈结构
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction_ratio, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch_size, channels, _, _ = x.size()

        # Squeeze: 全局平均池化获取通道统计信息
        y = self.global_avg_pool(x).view(batch_size, channels)

        # Excitation: 通过全连接层学习通道权重
        y = self.fc(y).view(batch_size, channels, 1, 1)

        # Scale: 将学习到的权重应用到原始特征上
        return x * y.expand_as(x)


# 定义SE-CBAM混合注意力模块（SE通道注意力 + 空间注意力）
class SE_SA_Module(nn.Module):
    def __init__(self, channels, reduction_ratio=16):
        super(SE_SA_Module, self).__init__()
        # 使用SE注意力作为通道注意力
        self.se_attention = SE_Attention(channels, reduction_ratio)

        # 空间注意力（保持不变）
        self.spatial_attention = nn.Sequential(
            nn.Conv2d(2, 1, kernel_size=7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        # SE通道注意力
        x_se = self.se_attention(x)

        # 空间注意力
        avg_out = torch.mean(x_se, dim=1, keepdim=True)
        max_out, _ = torch.max(x_se, dim=1, keepdim=True)
        spatial_att = torch.cat([avg_out, max_out], dim=1)
        spatial_att = self.spatial_attention(spatial_att)

        return x * spatial_att


# 定义CNN+SE-CBAM网络
class CNN_SE_SA_Module(nn.Module):
    def __init__(self):
        super(CNN_SE_SA_Module, self).__init__()
        # 卷积层1
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        self.se_sa_m1 = SE_SA_Module(32)

        # 卷积层2
        self.conv2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.se_sa_m2 = SE_SA_Module(64)
        self.pool1 = nn.MaxPool2d(2)

        # 卷积层3
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.se_sa_m3 = SE_SA_Module(128)

        # 卷积层4
        self.conv4 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
        )
        self.se_sa_m4 = SE_SA_Module(256)
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
        x = self.se_sa_m1(x)

        x = self.conv2(x)
        x = self.se_sa_m2(x)
        x = self.pool1(x)

        x = self.conv3(x)
        x = self.se_sa_m3(x)

        x = self.conv4(x)
        x = self.se_sa_m4(x)
        x = self.pool2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return F.log_softmax(x, dim=1)


# 热力图生成函数
def generate_heatmap(model, device, image, target_class=None):
    """
    生成Grad-CAM热力图
    """
    model.eval()

    # 注册钩子来获取最后一个卷积层的特征图和梯度
    features = []
    gradients = []

    def forward_hook(module, input, output):
        features.append(output)

    def full_backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # 获取最后一个卷积层并注册钩子
    last_conv_layer = model.conv4[0]  # 获取conv4中的Conv2d层
    forward_handle = last_conv_layer.register_forward_hook(forward_hook)
    backward_handle = last_conv_layer.register_full_backward_hook(full_backward_hook)

    try:
        # 准备输入 - 确保requires_grad为True
        image = image.unsqueeze(0).to(device)  # 添加batch维度
        image.requires_grad_(True)

        # 前向传播
        output = model(image)

        # 如果没有指定目标类别，使用预测的类别
        if target_class is None:
            target_class = output.argmax(dim=1).item()

        # 反向传播计算梯度
        model.zero_grad()

        # 创建目标类别的one-hot编码
        one_hot_output = torch.zeros_like(output)
        one_hot_output[0][target_class] = 1

        # 计算梯度
        output.backward(gradient=one_hot_output, retain_graph=True)

        # 获取特征图和梯度
        if len(features) == 0 or len(gradients) == 0:
            print("Warning: No features or gradients captured")
            return np.zeros((28, 28)), target_class

        feature_maps = features[0].cpu().data.numpy()[0]  # [C, H, W]
        grad_val = gradients[0].cpu().data.numpy()[0]  # [C, H, W]

        # 计算权重（全局平均池化梯度）
        weights = np.mean(grad_val, axis=(1, 2))  # [C]

        # 生成热力图
        cam = np.zeros(feature_maps.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * feature_maps[i, :, :]

        # ReLU操作
        cam = np.maximum(cam, 0)

        # 归一化
        if cam.max() > 0:
            cam = cam / cam.max()

        # 上采样到原始图像大小
        cam = cv2.resize(cam, (28, 28))

        return cam, target_class

    except Exception as e:
        print(f"Error generating heatmap: {e}")
        return np.zeros((28, 28)), target_class
    finally:
        # 确保钩子被移除
        forward_handle.remove()
        backward_handle.remove()


# 简化的热力图生成函数
def generate_heatmap_simple(model, device, image, target_class=None):
    """
    简化的热力图生成方法，避免梯度问题
    """
    model.eval()

    # 直接使用最后一个卷积层的输出作为特征重要性
    with torch.no_grad():
        # 前向传播直到最后一个卷积层
        x = image.unsqueeze(0).to(device)

        x = model.conv1(x)
        x = model.se_sa_m1(x)

        x = model.conv2(x)
        x = model.se_sa_m2(x)
        x = model.pool1(x)

        x = model.conv3(x)
        x = model.se_sa_m3(x)

        x = model.conv4(x)
        feature_maps = x  # 最后一个卷积层的输出

        # 继续前向传播获取预测
        x = model.se_sa_m4(x)
        x = model.pool2(x)
        x = x.view(x.size(0), -1)
        output = model.fc(x)

        if target_class is None:
            target_class = output.argmax(dim=1).item()

    # 使用特征图的绝对值平均作为热力图
    feature_maps_np = feature_maps.cpu().numpy()[0]  # [C, H, W]

    # 计算每个空间位置的特征激活强度
    cam = np.mean(np.abs(feature_maps_np), axis=0)

    # 归一化
    if cam.max() > 0:
        cam = cam / cam.max()

    # 上采样到原始图像大小
    cam = cv2.resize(cam, (28, 28))

    return cam, target_class


# 可视化热力图
def visualize_heatmaps(model, device, test_loader, num_samples=10):
    """
    可视化多个样本的热力图
    """
    model.eval()
    samples = []

    # 收集样本
    with torch.no_grad():
        for data, target in test_loader:
            for i in range(len(data)):
                if len(samples) < num_samples:
                    samples.append((data[i], target[i]))
                else:
                    break
            if len(samples) >= num_samples:
                break

    # 生成热力图
    fig, axes = plt.subplots(3, num_samples, figsize=(2 * num_samples, 6))
    if num_samples == 1:
        axes = axes.reshape(3, 1)

    for i, (image, true_label) in enumerate(samples):
        # 原始图像
        img_np = image.squeeze().numpy()

        # 生成热力图 - 使用简化版本避免梯度问题
        try:
            heatmap, pred_class = generate_heatmap_simple(model, device, image)
        except Exception as e:
            print(f"Error generating heatmap for sample {i}: {e}")
            heatmap = np.zeros((28, 28))
            pred_class = -1

        # 显示原始图像
        axes[0, i].imshow(img_np, cmap='gray')
        axes[0, i].set_title(f'True: {true_label}\nPred: {pred_class}')
        axes[0, i].axis('off')

        # 显示热力图
        axes[1, i].imshow(heatmap, cmap='jet')
        axes[1, i].set_title('Heatmap')
        axes[1, i].axis('off')

        # 显示叠加图像
        axes[2, i].imshow(img_np, cmap='gray')
        axes[2, i].imshow(heatmap, cmap='jet', alpha=0.5)
        axes[2, i].set_title('Overlay')
        axes[2, i].axis('off')

    plt.tight_layout()
    plt.savefig('heatmap_visualization.png', dpi=150, bbox_inches='tight')
    plt.show()


# 分析热力图质量
def analyze_heatmap_quality(model, device, test_loader, num_samples=50):
    """
    分析热力图质量，验证模型是否学习到了合理特征
    """
    model.eval()
    good_attention_count = 0
    total_samples = 0

    print("Analyzing heatmap quality...")

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            if total_samples >= num_samples:
                break

            for i in range(len(data)):
                if total_samples >= num_samples:
                    break

                image, true_label = data[i], target[i]

                try:
                    # 生成热力图 - 使用简化版本
                    heatmap, pred_class = generate_heatmap_simple(model, device, image)

                    # 计算热力图在数字区域的比例
                    # 假设数字大致在图像中央区域 (7x7 到 21x21)
                    center_region = heatmap[7:21, 7:21]
                    center_intensity = np.sum(center_region)
                    total_intensity = np.sum(heatmap)

                    # 如果中心区域的热力强度占总强度的60%以上，认为是好的注意力
                    if total_intensity > 0 and center_intensity / total_intensity > 0.6:
                        good_attention_count += 1

                    total_samples += 1

                except Exception as e:
                    print(f"Error processing sample {total_samples}: {e}")
                    continue

    if total_samples == 0:
        print("No samples processed successfully")
        return 0.0

    quality_score = good_attention_count / total_samples
    print(f"Heatmap Quality Analysis:")
    print(f"Total samples analyzed: {total_samples}")
    print(f"Samples with good attention: {good_attention_count}")
    print(f"Quality score: {quality_score:.3f}")

    if quality_score > 0.7:
        print("✓ Model appears to be learning meaningful features")
    elif quality_score > 0.5:
        print("~ Model shows some meaningful feature learning")
    else:
        print("✗ Model may be fitting noise or not learning proper features")

    return quality_score


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


# 测试函数
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


# 绘制混淆矩阵
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
    model = CNN_SE_SA_Module().to(device)
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
    torch.save(model.state_dict(), "mnist_se_cbam_cnn.pth")
    print("Model saved to mnist_se_cbam_cnn.pth")

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

    # 新增：热力图可视化
    print("\nGenerating heatmaps...")
    visualize_heatmaps(model, device, test_loader, num_samples=10)

    # 新增：热力图质量分析
    print("\nPerforming heatmap quality analysis...")
    quality_score = analyze_heatmap_quality(model, device, test_loader, num_samples=100)

    print(f"\nHeatmap analysis completed. Quality score: {quality_score:.3f}")
    print("Heatmaps saved to 'heatmap_visualization.png'")


if __name__ == '__main__':
    main()