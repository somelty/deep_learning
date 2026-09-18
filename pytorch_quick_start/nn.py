import os
import sys
from torch.utils.data import DataLoader, dataloader
from tqdm import tqdm
from data_loader import iris_dataloader
import torch
import torch.nn as nn
import torch.optim as optim

# 初始化神经网络模型
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        super().__init__()
        self.layer1 = nn.Linear(in_dim, hidden_dim1)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        return x

# 定义计算环境
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 划分训练集、验证集、测试集
custom_dataset = iris_dataloader("../pytorch_quick_start/iris.data")
train_size = int(len(custom_dataset) * 0.7)
val_size = int(len(custom_dataset) * 0.2)
test_size = len(custom_dataset) - train_size - val_size
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(custom_dataset, [train_size, val_size, test_size])

# 数据集加载
# Dataloder 将数据集包装为可迭代对象，方便训练循环获取批量数据
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True) # how many samples per batch to load
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=True)
print("Number of training examples: ", len(train_dataset), "\nNumber of validation examples: ", len(val_loader), "\nNumber of test examples: ", len(test_loader))

# 推理函数，计算并返回准确率
# 1. `torch.max(a)`：只传张量，不传 dim
# 作用：把张量里面全部元素摊平，找全局唯一一个最大值，返回该标量张量
# 2. `torch.max(a, dim=k)`：指定 dim 维度 → 返回 `(values, indices)`
# 沿着指定的维度，逐个切片求最大值，返回 `(values, indices)` 元组
# x = x.unsqueeze(0) # 在dim=0增加一维，变成 shape [1,4]
def infer(model, dataset, device):
    model.eval() # 模型定为推理阶段
    acc_num = 0
    sample_n = 0
    with torch.no_grad(): # 不更新参数，使用 with 管理
        for data in dataset:
            features, label = data
            outputs = model(features.to(device)) # DataLoader 会自动加 batch 维，即使 batch_size=1，输出也是 shape [1, 3]，即 (batch, 类别数)
            predict_y = torch.max(outputs, dim=1)[1] # 沿类别维取最大值的下标，shape [batch]
            sample_n += label.shape[0] # 累加真实样本数
            # squeeze：把标签变成 [batch]，与 predict_y 一一对应；否则 batch_size>1 时会广播成 [batch, batch]，准确率被算大 batch 倍
            acc_num += torch.eq(predict_y, label.to(device).squeeze(-1)).sum().item() # 张量做比较运算必须在同一个设备。`.to(device)`：把标签张量移动到目标设备
        acc = acc_num / sample_n # 注意分母要用样本数：len(DataLoader) 是批次数，只有 batch_size=1 时才等于样本数
        return acc

def main(lr=0.005, epochs=20):
    model = NN(4, 12, 6, 3).to(device)
    loss_f = nn.CrossEntropyLoss()
    pg = [p for p in model.parameters() if p.requires_grad] # 可以进行更新的参数
    optimizer = optim.Adam(pg, lr=lr)

    # 权重文件存储路径
    save_path = os.path.join(os.getcwd(), "results/weights")
    if os.path.exists(save_path) is False:
        os.makedirs(save_path)

    # 开始训练
    for epoch in range(epochs):
        model.train()
        # 用 Python 数字累加（.item() 返回的就是 Python 数字），避免 train_acc 变成张量而无法用 {:.3f} 格式化
        acc_num = 0
        sample_n = 0

        train_bar = tqdm(train_loader, ncols=100)
        for datas in train_bar:
            data, label = datas
            label = label.squeeze(-1)
            sample_n += data.shape[0]

            optimizer.zero_grad()
            outputs = model(data.to(device))
            pred_class = torch.max(outputs, dim=1)[1]
            acc_num += torch.eq(pred_class, label.to(device)).sum().item()

            loss = loss_f(outputs, label.to(device))
            loss.backward()
            optimizer.step()

            train_acc = acc_num / sample_n
            train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch+1, epochs, loss)

        val_acc = infer(model, val_loader, device)
        print("train epoch[{}/{}] loss:{:.3f} train_acc{:.3f} val_acc{:.3f}".format(epoch+1, epochs, loss, train_acc, val_acc))
        torch.save(model.state_dict(), os.path.join(save_path, "nn.pth"))

        # 数据集迭代后，对初始化指标清零
        train_acc = 0
        val_acc = 0

    print("Finished training")

    test_acc = infer(model, test_loader, device)
    print("test_acc", test_acc)

if __name__ == '__main__':
    main()


