# Deep Learning 深度學習

長庚大學深度學習課程的作業與實驗紀錄。

本儲存庫包含三個深度學習作業，涵蓋：

- CNN 影像分類
- BERT 文字情緒分類
- Vision Transformer（ViT）醫療影像分類

主要使用 Python、PyTorch、Hugging Face Transformers 與 Google Colab 完成。

## 目錄結構

```bash
Deep-Learning/
├── hw2/
│   ├── DL_HW2_B1228005.ipynb
│   ├── B1228005 胡樂麒 深度學習HW2.pdf
│   └── requirements.txt
├── hw3/
│   ├── DL_HW3_B1228005.ipynb
│   ├── DL_HW3_B1228005.pdf
│   └── requirement.txt
├── hw4/
│   ├── main_HW4.ipynb
│   ├── model.py
│   ├── helper_funcs.py
│   ├── DL_HW4_B1228005.pdf
│   ├── requirements.txt
│   └── environment type and package list.txt
└── README.md
```

## 作業總覽

| 作業 | 主題 | 主要模型 | 使用技術 |
|------|------|----------|----------|
| HW2 | CIFAR-10 影像分類 | 自訂 CNN | PyTorch、Conv2D、BatchNorm、Dropout |
| HW3 | 文字情緒分類 | BERT | Transformers、Tokenizer、Trainer |
| HW4 | PneumoniaMNIST 肺炎影像分類 | Vision Transformer | ViT、LoRA、PEFT、PyTorch |

---

# HW2：CIFAR-10 影像分類

## 作業目標

使用 CIFAR-10 資料集建立卷積神經網路，完成十分類影像辨識。

CIFAR-10 包含以下類別：

```text
airplane
automobile
bird
cat
deer
dog
frog
horse
ship
truck
```

## 實作內容

### CIFAR-10 資料處理

CIFAR-10 原始資料是 binary batch 格式，因此先使用 `pickle` 讀取資料，再將每筆資料轉換成圖片。

原始資料共有 3072 個數值，會重新排列成：

```text
3 × 32 × 32
```

最後轉換成：

```text
32 × 32 × 3
```

並按照類別儲存：

```text
cifar10/
├── train/
│   ├── 0/
│   ├── 1/
│   ├── ...
│   └── 9/
└── test/
    ├── 0/
    ├── 1/
    ├── ...
    └── 9/
```

### 自訂 Dataset

Notebook 中建立 `CIFAR10ImageFolder` 類別，用來：

- 掃描圖片路徑
- 讀取圖片
- 建立 label
- 套用影像轉換
- 回傳圖片與分類標籤

訓練資料再使用 `random_split` 分割出 validation set。

### 影像增強

訓練資料使用：

```python
RandomHorizontalFlip()
RandomCrop(32, padding=4)
ToTensor()
Normalize()
```

測試資料則只進行：

```python
ToTensor()
Normalize()
```

這樣可以增加訓練資料的變化，同時避免測試資料受到隨機變換影響。

## CNN 模型架構

HW2 使用自訂的 `SimpleCNN`，沒有直接使用現成模型。

```text
Input Image
    ↓
Conv2d(3 → 32) + BatchNorm + ReLU
    ↓
Max Pooling
    ↓
Conv2d(32 → 64) + BatchNorm + ReLU
    ↓
Max Pooling
    ↓
Conv2d(64 → 128) + BatchNorm + ReLU
    ↓
Max Pooling
    ↓
Conv2d(128 → 128) + BatchNorm + ReLU
    ↓
Max Pooling
    ↓
Flatten
    ↓
Fully Connected Layer
    ↓
Dropout
    ↓
10-class Output
```

## 訓練方法

HW2 使用 PyTorch 手動撰寫 training loop：

```python
model.train()
optimizer.zero_grad()
outputs = model(images)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

每個 epoch 會：

- 計算 training loss
- 計算 training accuracy
- 在 validation set 上測試
- 儲存 loss 與 accuracy
- 繪製訓練曲線

## 訓練設定

```text
Optimizer: Adam
Learning rate: 0.001
Loss: CrossEntropyLoss
Label smoothing: 0.1
Training epochs: 50
Training batch size: 64
Testing batch size: 128
Scheduler: StepLR
```

## 結果

Notebook 中記錄的測試準確率約為：

```text
85.92%
```

---

# HW3：文字情緒分類

## 作業目標

使用 Transformer 模型，根據句子內容判斷文字所代表的情緒。

使用的資料集：

```text
dair-ai/emotion
```

資料集共有六種情緒類別，並分成 training、validation 與 testing data。

## 實作內容

### 載入資料集

使用 Hugging Face Datasets：

```python
load_dataset("dair-ai/emotion")
```

分別載入：

```text
train
validation
test
```

### 文字 Tokenization

文字不能直接輸入 BERT，因此使用：

```python
BertTokenizer
```

將句子轉換成 token ID。

處理流程：

```text
Raw Text
    ↓
BERT Tokenizer
    ↓
Input IDs
    ↓
Attention Mask
    ↓
BERT Model
    ↓
Emotion Classification
```

Tokenizer 會處理：

- `[CLS]`
- `[SEP]`
- `[PAD]`
- `[UNK]`
- `[MASK]`

### Dynamic Padding

使用：

```python
DataCollatorWithPadding
```

進行 dynamic padding。

每個 batch 會依照該 batch 中最長的句子補齊，避免所有句子都被固定補到相同長度，以減少不必要的計算。

## BERT 模型

主要使用：

```text
bert-base-uncased
```

透過：

```python
BertForSequenceClassification.from_pretrained()
```

載入預訓練 BERT，並將最後的分類層設定為六分類。

Notebook 中也保留另一個可嘗試的模型：

```text
distilbert-base-uncased
```

## 訓練方法

HW3 使用 Hugging Face `Trainer`，由 Trainer 負責：

- 訓練模型
- 驗證模型
- 儲存 checkpoint
- 載入最佳模型
- 計算評估結果
- 記錄訓練狀態

## 訓練設定

```text
Epochs: 3
Learning rate: 2e-5
Training batch size: 16
Evaluation batch size: 64
Gradient accumulation steps: 2
Warmup steps: 500
Weight decay: 0.01
Random seed: 42
```

## 評估指標

使用以下指標評估模型：

- Accuracy
- Precision
- Recall
- F1-score

主要使用：

```python
accuracy_score()
precision_recall_fscore_support()
```

## 訓練狀態

目前 Notebook 中的訓練因運算時間過長而中斷：

```text
Training step: 6 / 1500
Reason: KeyboardInterrupt
```

因此目前沒有完整的最終測試結果。

---

# HW4：PneumoniaMNIST 影像分類

## 作業目標

使用 PneumoniaMNIST 胸腔 X 光影像，判斷影像是否具有肺炎特徵。

這是一個二元分類問題：

```text
Pneumonia
Normal
```

## 實作內容

### 載入 PneumoniaMNIST

使用：

```python
from medmnist import PneumoniaMNIST
```

資料包含：

- Training set
- Validation set
- Testing set

原始 training set 再以 80%／20% 分成 training 與 validation。

### 影像前處理

使用 Hugging Face：

```python
AutoImageProcessor
```

取得模型所需的影像尺寸、平均值與標準差。

前處理流程：

```text
Resize
    ↓
ToTensor
    ↓
Normalize
```

PneumoniaMNIST 是灰階單通道影像，而預訓練 ViT 通常需要三通道輸入，因此將影像通道複製三次：

```python
pixel_values = pixel_values.repeat(1, 3, 1, 1)
```

也就是：

```text
1 channel → 3 channels
```

## Vision Transformer 模型

主要預訓練模型：

```text
google/vit-base-patch16-224-in21k
```

Notebook 中也保留另一個模型：

```text
google/vit-base-patch16-224
```

模型類型可以透過以下參數切換：

```python
MODEL_TYPE = "HF"
```

或：

```python
MODEL_TYPE = "custom"
```

### Hugging Face 模式

在 `HF` 模式下，使用：

```python
AutoModelForImageClassification
```

載入預訓練 ViT，並設定：

```python
num_labels = 2
```

### 自訂 ViT 分類器

在 `custom` 模式下：

```python
base_model = ViTModel.from_pretrained(MODEL_NAME)
model = CustomViTClassifier(base_model, num_labels=2)
```

自訂模型位於：

```text
hw4/model.py
```

其功能是將 ViT 的輸出接到自訂的線性分類層：

```python
self.classifier = nn.Linear(
    self.vit.config.hidden_size,
    num_labels
)
```

模型流程：

```text
Input Image
    ↓
Vision Transformer
    ↓
Pooler Output
    ↓
Linear Classification Layer
    ↓
2-class Output
```

## LoRA 參數微調

HW4 使用 PEFT 的 LoRA 進行微調。

LoRA 設定：

```text
LoRA rank: 8
LoRA alpha: 16
LoRA dropout: 0.1
Target modules: query, key, value
Modules to save: classifier
```

LoRA 不需要更新整個 ViT 的所有參數，只訓練少量額外參數，因此可以：

- 降低 GPU 記憶體使用量
- 加快訓練速度
- 減少儲存空間
- 保留原本預訓練模型的能力

## 訓練方法

HW4 使用 PyTorch 手動撰寫 training loop：

```python
model.train()
optimizer.zero_grad()
outputs = model(pixel_values)
loss = loss_fn(outputs, labels)
loss.backward()
optimizer.step()
```

每個 epoch 結束後，使用：

```python
do_test()
```

在 validation set 或 test set 上進行評估。

## 訓練設定

```text
Batch size: 32
Epochs: 3
Learning rate: 1e-4
Loss: CrossEntropyLoss
Optimizer: Adam
```

---

## 環境需求

主要套件：

```text
torch
torchvision
transformers
peft
medmnist
tqdm
datasets
scikit-learn
matplotlib
numpy
pandas
```

