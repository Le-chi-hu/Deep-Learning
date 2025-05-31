import torch.nn as nn
# 使用者可以在 ViT 模型上添加自定義的分類層
class CustomViTClassifier(nn.Module):
   # 呼叫 nn.Module 的初始化函式
    def __init__(self, vit_model, num_labels=2):
        super().__init__()
        self.vit = vit_model# 將 ViT 模型存起來
        self.config = self.vit.config# 取得 ViT 模型的設定資訊，像是隱藏層的大小等等
        self.classifier = nn.Linear(self.vit.config.hidden_size, num_labels)
        # 建立一個線性層，將 ViT 輸出向量映射到 num_labels 個分類上
    def forward(self, pixel_values):
        # 將輸入圖像像素送入 ViT 模型並得到輸出
        outputs = self.vit(pixel_values=pixel_values)
        # 將 ViT 模型的池化輸出outputs.pooler_output輸入到分類層self.classifier中。
        logits = self.classifier(outputs.pooler_output)
        return logits