# 導入函式庫
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoModelForImageClassification
from sklearn.metrics import f1_score


class ImageDatasetWrapper(Dataset):
    """
    包裝 PneumoniaMNIST 數據集，使其適用於 ViT 模型
    - 將黑白單通道圖像轉換為 3 通道
    - 使用 transformers 的 image_processor 處理圖像
    """
    def __init__(self, dataset, image_processor):
      # 把傳入的數據集存起來
        self.dataset = dataset
        self.image_processor = image_processor# 把傳入的圖像處理器存起來
    
    def __len__(self):
      #回傳數據集的長度
        return len(self.dataset)
    
    def __getitem__(self, idx):
        # PneumoniaMNIST 返回 (image, label) 的元組
        # image 形狀為 [1, H, W]
        image, label = self.dataset[idx]
        
        # 將單通道圖像轉換為 3 通道 (複製通道)
        if image.shape[0] == 1:
            image = image.repeat(3, 1, 1)  # [1, H, W] -> [3, H, W]
        
        # 使用 image_processor 處理圖像
        # AutoImageProcessor 通常需要 PIL 圖像或 numpy 數組
        # 這裡我們將 tensor 轉換為 numpy 數組
        image_np = image.permute(1, 2, 0).numpy()  # [C, H, W] -> [H, W, C]
        
        # 使用 image_processor 來處理 numpy 數組
        # images=image_np 指定要處理的圖像
        # return_tensors="pt" 回傳 tensor 格式
        inputs = self.image_processor(images=image_np, return_tensors="pt")
        
        pixel_values = inputs["pixel_values"].squeeze(0)  # 移除批次維度

        # 把pixel_values,label 變成一個字典回傳
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label, dtype=torch.long)
        }

# num_labels=2 分類的類別數量
def load_HF_model(model_name, num_labels=2):
   # 使用 AutoModelForImageClassification.from_pretrained 來載入預訓練模型
    model = AutoModelForImageClassification.from_pretrained(
        model_name,
        num_labels=num_labels,  # PneumoniaMNIST is a binary classification
        ignore_mismatched_sizes=True, # https://github.com/huggingface/transformers/issues/13127
        problem_type="single_label_classification"  # 指定問題類型為單標籤分類
    )
    return model

def do_test(
      # 接收參數
        dataloader,
        model,
        model_type,
        loss_fn,
        device,
        num_epochs,
        cur_epoch=0,
        mode="validation",
):   # 模型評估,不啟用dropout和BN
    model.eval()
    # 顯示評估的進度
    pbar = tqdm(dataloader)
    # 進度條的描述
    pbar.set_description(f"{mode} epoch [{cur_epoch+1}/{num_epochs}]")
    # 初始化兩個空的 PyTorch tensor，用於儲存pred和gt
    pred = torch.tensor([], dtype=torch.int64)
    gt = torch.tensor([], dtype=torch.int64)
    # 初始化total_loss
    total_loss = 0

    with torch.no_grad():
        for batch in pbar:
            
            pixel_values = batch[0].to(device)  # 假設像素值是第一個元素
            labels = batch[1].to(device)  # 假設像素值是第二個元素

            pixel_values = pixel_values.repeat(1, 3, 1, 1)#複製三次通道
            labels = labels.squeeze().long()# 移除批次維度
             # 根據模型類型來呼叫outputs
            if model_type == "HF":
              # 將 pixel_values 和 labels 輸入模型
                outputs = model(pixel_values=pixel_values, labels=labels).logits
            elif model_type == "custom":
              
              # 將pixel_values輸入模型並回傳logits
                outputs = model(pixel_values=pixel_values)
            # 計算損失函數
            loss = loss_fn(outputs, labels)
            total_loss += loss.item()
            # 在最後一個維度上找到最大值的索引
            # 索引為模型預測的類別
            preds = torch.argmax(outputs, dim=-1)
            # 儲存結果
            pred = torch.cat((pred, preds.cpu()))
            gt = torch.cat((gt, labels.cpu()))
    # 計算準確度
    accuracy = (pred == gt).float().mean().item()
    # 計算F1 score
    f1 = f1_score(gt.numpy(), pred.numpy(), average='macro')
    # 輸出準確度和F1 score
    print(f"Accuracy: {accuracy:.4f} \nF1 Score: {f1:.4f}")
    # 計算平均損失
    total_loss /= len(dataloader)

    # 回傳平均損失
    return total_loss
