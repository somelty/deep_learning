# pytorch_quick_start 数据流转详解

> 这份文档只回答一个问题：**这份 Iris 数据，从磁盘上的一个文本文件开始，到最终输出一个预测结果，中间每一步被代码变成了什么形状、什么类型的东西。**
>
> 下面出现的所有形状和数值，都是在本项目上实际跑出来的，不是示意值。

---

## 0. 总览：一条数据的一生

```
磁盘文件 iris.data （纯文本）
  │
  │  pandas.read_csv(...)                          ← data_loader.py
  ▼
DataFrame (150, 5)        4 列数值(float64) + 1 列类别字符串
  │
  │  df[4].map(d)           "Iris-setosa" → 0
  ▼
DataFrame (150, 5)        类别列变成 0/1/2 (int64)
  │
  │  df.iloc[:, :4]  /  df.iloc[:, 4:]             ← 拆成两张表（关键的一步）
  ▼
data DataFrame (150, 4)                            label DataFrame (150, 1)
  │  (data - mean) / std   每列标准化                 │
  ▼                                                  ▼
data DataFrame (150, 4)                            label DataFrame (150, 1)
  │  torch.from_numpy(np.array(...,float32))         │  torch.from_numpy(np.array(...,int64))
  ▼                                                  ▼
self.data  Tensor [150, 4] float32                 self.label Tensor [150, 1] int64
  │                                                  │
  │  __getitem__(i)         按索引取一条              │
  ▼                                                  ▼
单条样本   Tensor [4] float32                        Tensor [1] int64
  │                                                  │
  │  DataLoader(batch_size=16)  自动 list → stack → 加 batch 维
  ▼                                                  ▼
一个 batch Tensor [16, 4] float32                    Tensor [16, 1] int64
  │                                                  │
  │                                                  │  label.squeeze(-1)
  │                                                  ▼
  │                                                  Tensor [16]
  │  model(x)                                        │
  ▼                                                  │
输出     [16, 4] → [16, 12] → [16, 6] → [16, 3]      │
  │                                                  │
  │  argmax(dim=1)                                   │
  ▼                                                  ▼
预测类别 Tensor [16]        ←──── 对比 ────→       真实标签 Tensor [16]
```

**一句话总结**：`[150,4]` 是"150 条样本、每条 4 个特征"，`[16,4]` 是"16 条样本、每条 4 个特征"。所谓"维度"从头到尾只有两个含义——**横着数有几条样本，竖着数每条有几个数**。

---

## 1. 第 0 步：磁盘上的原始文件

`iris.data` 是纯文本，一行一条花，逗号分隔，**没有表头**：

```
5.1,3.5,1.4,0.2,Iris-setosa
4.9,3.0,1.4,0.2,Iris-setosa
4.7,3.2,1.3,0.2,Iris-setosa
...
```

每行 5 个字段：4 个数值（花萼长、花萼宽、花瓣长、花瓣宽）+ 1 个类别名。共 150 行，三个类别各 50 行。

---

## 2. `data_loader.py`：从表格到张量

这个文件做的事可以拆成 6 步，全在 `iris_dataloader.__init__` 里。

### 第 1 步：读成表格

```python
df = pd.read_csv(self.data_path, names=[0, 1, 2, 3, 4])
```

`names=[0,1,2,3,4]` 是因为原文件没有表头，手动补 5 个列名（列名就直接用数字 0~4）。

| 项目 | 结果 |
|---|---|
| 类型 | `pandas.DataFrame` |
| 形状 | `(150, 5)` = 150 行 5 列 |
| 各列 dtype | 第 0~3 列 `float64`，第 4 列 `str` |

```
     0    1    2    3            4
0  5.1  3.5  1.4  0.2  Iris-setosa
1  4.9  3.0  1.4  0.2  Iris-setosa
2  4.7  3.2  1.3  0.2  Iris-setosa
```

> 注意：此时 dtype 是 **float64**（pandas/numpy 的默认浮点），不是 PyTorch 喜欢的 float32。这一步埋了个伏笔，第 5 步要转。

### 第 2 步：类别名 → 数字

```python
d = {"Iris-setosa": 0, "Iris-versicolor": 1, "Iris-virginica": 2}
df[4] = df[4].map(d)
```

`.map(字典)` 把第 4 列的字符串逐行替换成数字。**形状不变**，还是 `(150, 5)`，但第 4 列的 dtype 从 `str` 变成 `int64`。

```
0  → 0
1  → 1
2  → 2
（各 50 条）
```

为什么要映射：神经网络只会算数，`"Iris-setosa"` 这种字符串它没法处理，类别必须变成整数编号。

### 第 3 步：切成"特征表"和"标签表"（**后面所有维度问题的源头**）

```python
data  = df.iloc[:, :4]   # 所有行，第 0~3 列
label = df.iloc[:, 4:]   # 所有行，第 4 列
```

`iloc[行切片, 列切片]` 是按**位置**取数。这里两个切片的写法有个很容易被忽略的差别：

| 变量 | 写法 | 结果 |
|---|---|---|
| `data` | `[:, :4]` → 取 0,1,2,3 四列 | `DataFrame (150, 4)` |
| `label` | `[:, 4:]` → 取 4 到最后一列 | `DataFrame (150, 1)` ← **二维！** |

`label` 用 `4:`（切片）而不是 `4`（单个数），所以拿到的**不是**一列 150 个数字，而是"150 行 × 1 列"的一张小表。

这就是后面所有 `[150, 1]`、`[16, 1]`、`squeeze(-1)` 的根源：

```python
data.iloc[0]     # [5.1, 3.5, 1.4, 0.2]        4 个数
label.iloc[0]    # [0]                          ← 一个长度为 1 的列表，不是裸的 0
```

> 对比：如果当初写的是 `df.iloc[:, 4]`（没有冒号），`label` 会是一维的 `Series (150,)`，后面就完全不需要 `squeeze` 了。这是 pandas 里极常见的一个坑。

### 第 4 步：标准化（z-score）

```python
data = (data - data.mean()) / data.std()
```

对**每一列**分别做：`(原始值 - 该列均值) / 该列标准差`，让每列变成"均值 0、标准差 1"。

本项目的真实统计量（4 列分别是 花萼长/花萼宽/花瓣长/花瓣宽）：

| 列 | 均值 | 标准差 |
|---|---|---|
| 0 花萼长 | 5.8433 | 0.8281 |
| 1 花萼宽 | 3.0540 | 0.4336 |
| 2 花瓣长 | 3.7587 | 1.7644 |
| 3 花瓣宽 | 1.1987 | 0.7632 |

拿第一行 `[5.1, 3.5, 1.4, 0.2]` 手算验证：

```
(5.1 - 5.8433) / 0.8281 = -0.8977
(3.5 - 3.0540) / 0.4336 =  1.0286
(1.4 - 3.7587) / 1.7644 = -1.3368
(0.2 - 1.1987) / 0.7632 = -1.3086
```

标准化后：`[-0.8977, 1.0286, -1.3368, -1.3086]`，整列再检查是 `mean ≈ 0, std = 1.0`。形状仍是 `(150, 4)`。

为什么要做：4 个特征的量纲差很多（花萼长 4~8，花瓣宽 0.1~2.5），不统一尺度的话，数值大的特征会主导梯度，训练变慢甚至学不动。

### 第 5 步：pandas 表 → PyTorch 张量

```python
self.data  = torch.from_numpy(np.array(data,  dtype=np.float32))
self.label = torch.from_numpy(np.array(label, dtype=np.int64))
```

拆开看是两次转换：`DataFrame → np.array（numpy 数组）→ torch.Tensor`。`dtype=` 决定最终精度：

| 变量 | 最终类型 | 形状 | dtype | 为什么是这个 dtype |
|---|---|---|---|---|
| `self.data` | `torch.Tensor` | `[150, 4]` | `float32` | PyTorch 运算默认 float32，显式转换是为了把 pandas 的 float64 降下来 |
| `self.label` | `torch.Tensor` | `[150, 1]` | `int64` | 分类任务的标签必须是 64 位整数，见下方说明 |

第一条样本进张量后是这样（float32 的真实精度）：

```python
self.data[0]  = [-0.8976739, 1.0286113, -1.3367940, -1.3085928]   # 注意结尾多出来的零头，那是 float32 的精度痕迹
self.label[0] = [0]                                                # 注意外面还包着一层方括号
```

> **为什么标签必须是 int64（long）**：`nn.CrossEntropyLoss` 内部要拿标签去索引"第几类"，PyTorch 规定索引必须是 64 位整数。写成 `int8` 会直接报 `RuntimeError: expected target dtype to be Long or Byte`；写 `int32` 同样不行。这是项目里已经踩过一次的坑。

### 第 6 步：`__len__` 和 `__getitem__`

```python
def __len__(self):
    return self.data_num          # 返回 150

def __getitem__(self, index):
    # 经过上一步数据类型转化，这里data，label已经是张量，张量可以索引
    return self.data[index], self.label[index]
```
继承 PyTorch 的 `Dataset` 要重写三个方法。 其中两个：**一共多少条**、**给我第 i 条**。DataLoader 就是靠这两个方法工作的。

`__getitem__` 返回的是**一条**样本，不是一批：

```python
# 当写 dataset[0] 时会调用刚刚写的 __getitem__(0)
# 返回第一行样本 self.data[0], self.label[0]
x, y = dataset[0]
x.shape # → (4,)     不是 [1,4]，也不是 [150,4]！从 [150,4] 里取下标 0，去掉的是"样本维"
y.shape # → (1,)       label 从 [150,1] 取下标 0，剩下一维长度 1
x.dtype # → torch.float32
y.dtype # → torch.int64
```

---

## 3. `nn.py`：把 150 条切成 训练 / 验证 / 测试集

```python
train_size = int(len(custom_dataset) * 0.7)   # 105
val_size   = int(len(custom_dataset) * 0.2)   # 30
test_size  = len(custom_dataset) - train_size - val_size   # 15
train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(
    custom_dataset, [train_size, val_size, test_size])
```

| 项目 | 结果 |
|---|---|
| 划分 | 105 / 30 / 15（合计 150） |
| 返回类型 | `torch.utils.data.Subset`，拥有 `dataset` 和 `indices` 两个属性 |

**关键理解：这一步没有复制任何数据。** `Subset` 只存了两样东西——"原始数据集"和"我要用哪些下标"。实测：

```python
type(train_dataset)       # → Subset
train_dataset.indices[:8] # → [131, 133, 85, 124, 140, 14, 43, 72]   # 105 个乱序下标
train_dataset[0][0]       # → Tensor [4] 
```

注意：`self.data` 内存里只有一份 `[150, 4]`，subset 的 dataset 属性只是指向了 self.data 引用。

> `random_split` 没有设随机种子，所以每次运行 `indices` 都不一样，验证集/测试集内容每次都会变，准确率会在 0.87~1.0 之间波动，属正常现象。

---

## 4. DataLoader：这里容易晕

```python
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=1,  shuffle=True)
test_loader  = DataLoader(test_dataset,  batch_size=1,  shuffle=True)
```

DataLoader 做三件事：**按 `__getitem__` 逐条取样本 → 用 list 攒够 batch_size 条 → 把这个 list 的样本 stack 成一个张量**。

以 `batch_size=16` 遍历训练集为例：

| batch | x 形状      | y 形状      |
|-------|-----------|-----------|
| 0     | `(16, 4)` | `(16, 1)` |
| 1     | `(16, 4)` | `(16, 1)` |
| …     | …         | …         |
| 5     | `(16, 4)` | `(16, 1)` |
| 6     | `(9, 4)`  | `(9, 1)`  | ← 最后一批不够 16 条，有多少算多少 |

`6 × 16 + 9`，所以是 7 个 batch（6 个满的 + 1 个只有 9 条）

**这一步是维度困惑的最大来源，务必记牢：**

```
__getitem__  返回        Tensor [4]        ← 一维张量：一个样本，4 个特征
DataLoader   攒 16 条    新加第 0 维
DataLoader   交付        Tensor [16, 4]    ← 二维张量：一批（16 条样本），每条 4 个特征
```

`batch_size=1` 时也不例外：

```
__getitem__ 返回       Tensor [4]
DataLoader 交付        Tensor [1, 4]     ← 前面那个 1 是"一批里有 1 条"
```

**DataLoader 永远不会给你一维张量**，它交付的东西永远是"样本数在前"的。本项目早先的一个 bug 就是因为误以为 `batch_size=1` 时输出是一维的。

> 另一个坑：`len(DataLoader)` 返回的是**批次数**，不是样本数。
> `len(train_loader)` = 7（批次数），而 `len(train_dataset)` = 105（样本数）。
> 当 `batch_size=1` 时两者恰好相等

---

## 5. 模型 forward

```python
class NN(nn.Module):
    def __init__(self, in_dim, hidden_dim1, hidden_dim2, out_dim):
        self.layer1 = nn.Linear(in_dim, hidden_dim1)      # Linear(4, 12)
        self.layer2 = nn.Linear(hidden_dim1, hidden_dim2) # Linear(12, 6)
        self.layer3 = nn.Linear(hidden_dim2, out_dim)     # Linear(6, 3)
```

一个 batch（16 条）：

| 经过 | 形状 | 含义 |
|---|---|---|
| 输入 x | `(16, 4)` | 16 条样本，每条 4 个特征 |
| `layer1` | `(16, 12)` | 每条样本被映射成 12 个数 |
| `layer2` | `(16, 6)` | 每条样本被压成 6 个数 |
| `layer3` | `(16, 3)` | 每条样本得到 3 个得分（对应 3 个类别） |

`nn.Linear(a, b)` 只动**最后一维**——把每个样本的 `a` 个数变成 `b` 个数，样本维原样保留。所以 `[16,4] → [16,12]`，而不是 `[16,4] → [12]`。

输入是 `[1,4]` 时同理：`[1,4] → [1,12] → [1,6] → [1,3]`。**这就是为什么模型不关心传了一批还是一条数据** 

输出的 3 个数叫 **logits（未归一化的得分）**，实测一条样本：

```python
out[0] = [-0.056, -0.238, -0.643]     # 三个数之和不是 1（还没过 softmax）
argmax(dim=1) # → 0                     # 第 0 类得分最高，所以预测"setosa"
```

`argmax(dim=1)` 的含义：**沿着 dim=1（跨列/类别维）找最大值的下标**。`[16, 3]` 有 16 行，每行各自选一个最大的，结果就是 `[16]` 个类别编号。

> 为什么不 `dim=0`：那是跨行在 16 条样本之间比大小

---

## 6. 训练循环：一个 batch 的完整旅程

对照 `main()` 里的循环，一条数据被做了这些事：

| 代码 | 形状/类型变化 | 说明 |
|---|---|---|
| `for datas in train_bar:` | `datas` 是 `( [16,4], [16,1] )` 的二元组 | DataLoader 每次交付一批 |
| `data, label = datas` | `data [16,4]`，`label [16,1]` | 拆包 |
| `label = label.squeeze(-1)` | `[16,1] → [16]` | **必须做**，见下方说明 |
| `sample_n += data.shape[0]` | → 16 | 累加样本数，用来算准确率 |
| `optimizer.zero_grad()` | — | 清空上一轮的梯度 |
| `outputs = model(data.to(device))` | `[16,4] → [16,3]` | 前向传播 |
| `pred_class = torch.max(outputs, dim=1)[1]` | `[16,3] → [16]` | 取每条样本的预测类别 |
| `acc_num += torch.eq(pred_class, label.to(device)).sum().item()` | `[16]` vs `[16]` → 一个整数 | 逐位比较，相等的个数 |
| `loss = loss_f(outputs, label.to(device))` | `[16,3]` + `[16]` → 标量 | 交叉熵损失 |
| `loss.backward()` | — | 反向传播，算梯度 |
| `optimizer.step()` | — | 用梯度更新权重 |

**为什么必须 `squeeze(-1)`**：`CrossEntropyLoss` 要求标签是"16 个类别编号"（`[16]`），而不是"16 行 1 列"（`[16,1]`）。直接把 `[16,1]` 传进去会报：

```
RuntimeError: 0D or 1D target tensor expected, multi-target not supported
```

`squeeze(-1)` 的意思是"把最后一维里长度为 1 的那一维去掉"：`[16,1] → [16]`。名字读作 squeeze（挤掉）-1（最后一维）。
同理，若 `shape(1,16)`，**第 0 维是 1，最后一维 dim=1 长度是 16，不能删**。
要把`[1,16] → [16]`，要挤掉**dim=0**：`squeeze(0)`

> `CrossEntropyLoss` 内部已经包含了 `log_softmax + NLLLoss`，所以喂给它的必须是**没做过 softmax 的原始 logits**，不要在模型里手动加 softmax。

> `torch.max()`: 若加参数 `dim=k`，会把k维消灭，变为 k-1 维张量
---

## 7. `infer()`：推理时的数据流转

推理用的是 `batch_size=1` 的 loader，所以每次只进来 1 条：

| 代码 | 形状变化 |
|---|---|
| `features, label = data` | `features [1,4]`，`label [1,1]` |
| `outputs = model(features.to(device))` | `[1,4] → [1,3]` |
| `predict_y = torch.max(outputs, dim=1)[1]` | `[1,3] → [1]` |
| `label.to(device).squeeze(-1)` | `[1,1] → [1]` |
| `torch.eq(predict_y, 标签).sum().item()` | `[1]` vs `[1]` → 0 或 1 |
| `sample_n += label.shape[0]` | 累加真实样本数（这里是 1） |
| `acc = acc_num / sample_n` | 正确数 ÷ 样本数 = 准确率 |

这里三处细节都是踩过坑之后才写对的，值得记住：

1. **`dim=1` 不是 `dim=0`**：即使只有 1 条样本，输出也是 `[1,3]` 而不是 `[3]`。用 `dim=0` 是在"1 条样本"这个方向上找最大值，永远只能得到 `[0,0,0]`。
2. **标签要 `squeeze(-1)`**：`predict_y` 是 `[1]`，`label`是 `[1,1]`。不 squeeze 的话 `torch.eq` 会把它们**广播**成 `[1,1]`（batch_size=1 时侥幸不错），但一旦 batch_size>1 就会广播如 `[16,16]`，`.sum()` 出来的正确数被放大 16 倍。
3. **分母要用样本数**：`len(DataLoader)` 是批次数。`batch_size=1` 时批次数恰好奇妙地等于样本数。

---

## 8. 维度速查表

| 变量 | 形状 | dtype | 含义 |
|---|---|---|---|
| 原始文件一行 | 5 个字段 | 文本 | 4 特征 + 1 类别名 |
| `df` | `(150, 5)` | float64 ×4 + str | 全部原始数据 |
| `data`（切分后） | `(150, 4)` | float64 | 全部特征 |
| `label`（切分后） | `(150, 1)` | int64 | 全部标签，**二维** |
| `self.data` | `[150, 4]` | float32 | 全部特征张量 |
| `self.label` | `[150, 1]` | int64 | 全部标签张量，**二维** |
| `dataset[i]` | `([4], [1])` | float32, int64 | 一条样本 |
| `train_loader` 一批 | `([16,4], [16,1])` | float32, int64 | 16 条样本 |
| `label.squeeze(-1)` | `[16]` | int64 | 扁平化的标签 |
| `model(x)` | `[16, 3]` | float32 | 3 个类别的得分 |
| `argmax(dim=1)` | `[16]` | int64 | 预测类别 |
| `loss` | `()` 标量 | float32 | 交叉熵 |

**形状读法**：`[16, 4]` 从左往右读作"16 条样本，每条 4 个特征"。最左边那一维永远是**样本维**，最右边那一维永远是**特征/类别维**。`dim=0` 指最左边的样本方向，`dim=1` 指最右边的特征方向（维数更高时依次往右编号，倒数第一维也可以写成 `-1`）。

---

## 9. 本项目踩过的 5 个形状坑（都是真实报错）

| # | 现象 | 原因 | 正确做法 |
|---|---|---|---|
| 1 | `expected target dtype to be Long or Byte, but got Char` | 标签用了 int8 | 标签必须 `int64` |
| 2 | `unsupported format string passed to Tensor.__format__` | `train_acc` 是 `[1]` 张量却用 `"{:.3f}"` 格式化 | 只有 0 维（标量）张量能这么格式化；用 `.item()` 或改成 Python 数字累加 |
| 3 | `val_acc` 恒等于 1.000 | `torch.max(outputs, dim=0)` 取错维度 | argmax 取类别维要 `dim=1` |
| 4 | batch_size>1 时准确率 = 8.84 | `[16]` 和 `[16,1]` 广播成 `[16,16]` | 标签先 `squeeze(-1)` 再比较 |
| 5 | 准确率分母用了 `len(DataLoader)` | `len()` 返回批次数而非样本数 | 循环里累加 `sample_n += label.shape[0]` |

另外两条不是报错但同样重要的：

- `CrossEntropyLoss` 的 target 传 `[16,1]` 会报 `0D or 1D target tensor expected`
- 索引取一条样本 = 去掉最前面那一维：`[150,4] → [4]`。

---

## 10. 几个容易混的说法

- **"张量"和"数组"**：可以先把 Tensor 理解成 numpy 数组 + 两个额外能力（能放到 GPU 上、能自动求导）。`torch.from_numpy()` 就是在两者间做转换，形状和数值不变。
- **"维度"**：`tensor.dim()` 是"这个张量有几层方括号"（`[16,4]` 是 2），`tensor.shape` 是"每一层方括号里各有几个数"（16 和 4）。日常说"几维几维"通常指 `shape`。
- **`float64` vs `float32`**：pandas/numpy 默认 float64（64 位），PyTorch 默认 float32。本项目在 `np.array(..., dtype=np.float32)` 这一步显式降精度，所以打印 `self.data[0]` 会看到 `-0.8976739` 这种带零头的值。
- **`logits` vs `概率`**：模型输出的是 logits（原始得分，可以是负数、和不为 1）。`argmax` 只需要比大小，所以推理时**不需要** softmax；`CrossEntropyLoss` 内部自己会做。
- **`device`**：`data.to(device)` 是把数据从内存搬到显卡显存。参与同一次运算的所有张量必须在同一设备上，否则报 `Expected all tensors to be on the same device`。本项目用 `torch.cuda.is_available()` 自动选择，本机实际跑在 `cuda` 上。
