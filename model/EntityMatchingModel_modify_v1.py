import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset
from torch import optim
from torch.utils.data import DataLoader
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from jellyfish import jaro_winkler_similarity
batch_size = 8
n_epochs = 20
class GASA(nn.Module):
    def __init__(self, hidden_size):
        super(GASA, self).__init__()
        self.hidden_size = hidden_size

        # 属性注意力
        self.attr_query = nn.Linear(hidden_size, hidden_size)
        self.attr_key = nn.Linear(hidden_size, hidden_size)
        self.attr_value = nn.Linear(hidden_size, hidden_size)

        # 序列注意力
        self.conv = nn.Conv1d(hidden_size, hidden_size, kernel_size=3, padding=1)
        self.seq_attention = nn.Linear(hidden_size * 2, hidden_size)

        # 融合
        self.fusion = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, x, attr_masks):
        # 属性注意力
        attr_query = self.attr_query(x)
        attr_key = self.attr_key(x)
        attr_value = self.attr_value(x)

        attr_attention = torch.matmul(attr_query, attr_key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attr_attention = attr_attention.masked_fill(attr_masks.unsqueeze(1) == 0, -1e9)
        attr_attention = F.softmax(attr_attention, dim=-1)

        attr_context = torch.matmul(attr_attention, attr_value)

        # 序列注意力
        seq_feature = self.conv(x.transpose(1, 2)).transpose(1, 2)
        seq_pool = torch.cat([
            F.adaptive_avg_pool1d(seq_feature.transpose(1, 2), 1).squeeze(-1),
            F.adaptive_max_pool1d(seq_feature.transpose(1, 2), 1).squeeze(-1)
        ], dim=-1)
        seq_attention = self.seq_attention(seq_pool).unsqueeze(1)

        # 融合
        fused_attention = self.fusion(torch.cat([
            attr_context,
            seq_attention.expand(-1, x.size(1), -1)
        ], dim=-1))

        output = fused_attention + x # 使用残差连接

        return output

class IntegratedEntityMatchingModel(nn.Module):
    def __init__(self, device='cuda'):
        super(IntegratedEntityMatchingModel, self).__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.device = device
        hidden_size = self.bert.config.hidden_size

        self.gasa = GASA(hidden_size)

        # 传统特征处理
        self.numeric_fc = nn.Linear(1, 10)
        self.string_fc = nn.Linear(1, 10)

        # 最终分类层
        self.fc = nn.Linear(hidden_size + 20, 2)  # 20 是传统特征的维度

    def forward(self, x1, attr_masks, numeric_features, string_similarity):
        x1 = x1.to(self.device)
        attr_masks = attr_masks.to(self.device)

        # BERT编码
        bert_output = self.bert(x1)[0]

        # GASA处理
        gasa_output = self.gasa(bert_output, attr_masks)
        gasa_features = gasa_output[:, 0, :]  # 使用[CLS]标记的输出

        # 传统特征处理
        numeric_features = numeric_features.to(self.device)
        string_similarity = string_similarity.to(self.device)
        numeric_features = torch.relu(self.numeric_fc(numeric_features)).squeeze(1)
        string_features = torch.relu(self.string_fc(string_similarity)).squeeze(1)

        # 确保所有特征的维度一致
        batch_size = gasa_features.size(0)
        numeric_features = numeric_features.view(batch_size, -1)
        string_features = string_features.view(batch_size, -1)

        # 特征融合
        combined_features = torch.cat((gasa_features, numeric_features, string_features), dim=1)

        # 最终分类
        output = self.fc(combined_features)
        return output

# 辅助函数
def numeric_feature_extractor(entity):
    if isinstance(entity, (float, int)):
        return torch.tensor([[float(entity)]], dtype=torch.float32)
    elif isinstance(entity, str):
        return torch.tensor([[len(entity)]], dtype=torch.float32)
    else:
        raise ValueError("Unsupported data type for numeric feature extraction")

def calculate_string_similarity(str1, str2):
    return torch.tensor([[jaro_winkler_similarity(str1, str2)]], dtype=torch.float32)

# 数据处理和训练函数
class IntegratedEntityMatchingDataset(Dataset):
    def __init__(self, data_path, max_len=128):
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        self._load_data(data_path)

    def _load_data(self, data_path):
        try:
            with open(data_path, 'r', encoding='utf-8', errors='ignore') as f:
                lines = f.readlines()
            for line in lines:
                s1, s2, label = line.strip().split('\t')
                self.pairs.append((s1, s2))
                self.labels.append(int(label))
        except Exception as e:
            raise RuntimeError(f"Error loading data from {data_path}: {e}")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        left = self.pairs[idx][0]
        right = self.pairs[idx][1]

        # BERT编码
        encoded = self.tokenizer.encode_plus(
            text=left,
            text_pair=right,
            max_length=self.max_len,
            truncation=False,
            padding='max_length',
            return_tensors='pt',
        )

        # 提取数值特征和字符串相似度
        numeric_feature = numeric_feature_extractor(left) - numeric_feature_extractor(right)
        string_similarity = calculate_string_similarity(left, right)

        return {
            'input_ids': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
            'attr_masks': encoded['attention_mask'].squeeze(0),
            'numeric_feature': numeric_feature.squeeze(-1),  # 确保输出形状一致
            'string_similarity': string_similarity.squeeze(-1),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train(trainset, validset, testset):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IntegratedEntityMatchingModel(device=device).to(device)

    train_loader = DataLoader(trainset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(validset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    optimizer = optim.AdamW(model.parameters(), lr=3e-5)
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_loader) * n_epochs)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):
        model.train()
        for batch in train_loader:
            optimizer.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_feature = batch['numeric_feature'].to(device)
            string_similarity = batch['string_similarity'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, numeric_feature, string_similarity)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            scheduler.step()

        # 验证
        model.eval()
        val_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in valid_loader:
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                numeric_feature = batch['numeric_feature'].to(device)
                string_similarity = batch['string_similarity'].to(device)
                labels = batch['label'].to(device)

                outputs = model(input_ids, attention_mask, numeric_feature, string_similarity)
                val_loss += criterion(outputs, labels).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(labels.view_as(pred)).sum().item()

        print(f'Epoch {epoch+1}, Validation Accuracy: {correct / len(validset):.4f}')

    # 测试
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            numeric_feature = batch['numeric_feature'].to(device)
            string_similarity = batch['string_similarity'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, numeric_feature, string_similarity)
            test_loss += criterion(outputs, labels).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(labels.view_as(pred)).sum().item()

    print(f'Test Accuracy: {correct / len(testset):.4f}')

if __name__ == "__main__":
    trainset = IntegratedEntityMatchingDataset("../Beer/train.txt")
    validset = IntegratedEntityMatchingDataset("../Beer/valid.txt")
    testset = IntegratedEntityMatchingDataset("../Beer/test.txt")

    train(trainset, validset, testset)