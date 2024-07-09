import torch
import torch.nn as nn
from jellyfish import jaro_winkler_similarity

# 用于提取数值特征的函数
def numeric_feature_extractor(entity):
    if isinstance(entity, (float, int)):
        return torch.tensor([[float(entity)]], dtype=torch.float32)
    elif isinstance(entity, str):
        return torch.tensor([[len(entity)]], dtype=torch.float32)
    else:
        raise ValueError("Unsupported data type for numeric feature extraction")

# 用于计算字符串相似性的函数
def calculate_string_similarity(str1, str2):
    return torch.tensor([[jaro_winkler_similarity(str1, str2)]], dtype=torch.float32)

# 神经网络模型定义
class EntityMatchingModel(nn.Module):
    def __init__(self):
        super(EntityMatchingModel, self).__init__()
        self.numeric_fc = nn.Linear(1, 10)  # 处理数值特征
        self.string_fc = nn.Linear(1, 10)   # 处理字符串特征
        self.final_fc = nn.Linear(20, 1)    # 最终决策层

    def forward(self, numeric_features, string_similarity):
        numeric_features = torch.relu(self.numeric_fc(numeric_features))
        string_features = torch.relu(self.string_fc(string_similarity))
        combined_features = torch.cat((numeric_features, string_features), dim=1)
        match_score = torch.sigmoid(self.final_fc(combined_features))
        return match_score

# 示例实体属性
entity1 = "hello world"
entity2 = "apple Incorporated"

# 提取数值特征
numeric_feature1 = numeric_feature_extractor(entity1)
numeric_feature2 = numeric_feature_extractor(entity2)

# 计算字符串相似性
string_similarity = calculate_string_similarity(entity1, entity2)

# 初始化模型和优化器
model = EntityMatchingModel()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.BCELoss()

# 训练和评估
labels = torch.tensor([[0.0]])  # 假定标签为匹配
optimizer.zero_grad()
outputs = model(torch.abs(numeric_feature1 - numeric_feature2), string_similarity)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()

print(f"Loss: {loss.item()}")
print(f"Predicted Match Score: {outputs.item()}")
