import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim
from torch.nn.utils.rnn import pad_sequence
from torch.utils import data
from torch.utils.data import Dataset
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
from jellyfish import jaro_winkler_similarity
from sklearn.metrics import f1_score
import numpy as np
from transformers import get_linear_schedule_with_warmup
from tensorboardX import SummaryWriter

save_model = True
batch_size = 32
n_epochs = 20
logdir = r"Structured/"
task = "Structured/Walmart-Amazon"
run_tag = '%s' % (task)
run_tag = run_tag.replace('/', '_')
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

        output = fused_attention + x  # 使用残差连接
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
            truncation=True,
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
            'numeric_feature': numeric_feature.squeeze(-1),
            'string_similarity': string_similarity.squeeze(-1),
            'label': torch.tensor(self.labels[idx], dtype=torch.long)
        }

def train(trainset, validset, testset):
    # create the DataLoaders 加载DataLoader
    # 使用自定义的collate_fn在DataLoader中
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=collate_batch)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=batch_size * 16,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=collate_batch)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=batch_size * 16,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=collate_batch)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = IntegratedEntityMatchingModel(device=device)
    model = model.to(device)  # 模型的所有参数和缓冲区都移动到了 GPU 上
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    # 训练的总步数
    num_steps = (len(trainset) // batch_size) * n_epochs
    # 初始化学习率调度器
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=num_steps)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = os.path.join(logdir, f"{run_tag}_{timestamp}")
    # logging with tensorboardX
    writer = SummaryWriter(log_dir=log_dir)

    best_dev_f1 = best_test_f1 = 0.0
    best_threshold = 0.5
    for epoch in range(1, n_epochs + 1):
        # train 模型会切换到训练模式:执行前向传播、计算损失、反向传播和优化步骤。
        model.train()
        train_step(train_iter, model, optimizer, scheduler)

        # eval 设置模型为评估模式
        model.eval()  # 禁用梯度计算
        dev_f1, dev_threshold = evaluate(model, valid_iter)
        test_f1, _ = evaluate(model, test_iter, threshold=dev_threshold)

        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            best_threshold = dev_threshold
            if save_model:
                # create the directory if not exist
                directory = os.path.join(logdir, task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(directory, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)

        # Logging
        print(f"epoch {epoch}: dev_f1={dev_f1:.4f}, test_f1={test_f1:.4f}, best_dev_f1={best_dev_f1:.4f}, best_test_f1={best_test_f1:.4f}")

        # logging to tensorboard
        writer.add_scalars('F1_Scores', {'Dev F1': dev_f1, 'Test F1': test_f1}, epoch)

    writer.close()
    print(f"Final best test F1: {best_test_f1:.4f} at threshold {best_threshold:.2f}")

def train_step(train_iter, model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()

        # 从批次中解包所有项
        input_ids, attention_mask, attr_masks, numeric_features, string_similarity, labels = batch

        # 前向传播
        # 确保提供了所有必要的输入，以符合模型预期的参数
        prediction = model(input_ids, attr_masks, numeric_features, string_similarity)

        # 计算损失
        loss = criterion(prediction, labels.to(model.device))

        # 反向传播和优化
        loss.backward()
        optimizer.step()
        scheduler.step()

        # # 可选：每隔几步打印一次损失
        # if i % 10 == 0:
        #     print(f"step: {i}, loss: {loss.item()}")

        del loss  # 确保删除损失以释放内存


# 自定义collate_fn
def collate_batch(batch):
    # 分别处理input_ids和attention_mask
    input_ids = pad_sequence([item['input_ids'] for item in batch], batch_first=True, padding_value=1)  # 使用1作为padding值，因为1在Roberta中通常是padding token
    attention_mask = pad_sequence([item['attention_mask'] for item in batch], batch_first=True, padding_value=0)  # attention_mask用0填充
    attr_masks = pad_sequence([item['attr_masks'] for item in batch], batch_first=True, padding_value=0)  # attr_masks用0填充
    numeric_features = torch.stack([item['numeric_feature'] for item in batch])
    string_similarity = torch.stack([item['string_similarity'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])

    return input_ids, attention_mask, attr_masks, numeric_features, string_similarity, labels


def evaluate(model, iterator, threshold=None):
    all_labels = []
    all_preds = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            # 解包批次
            input_ids, attention_mask, attr_masks, numeric_features, string_similarity, labels = batch

            # 模型推理
            logits = model(input_ids, attr_masks, numeric_features, string_similarity)
            probs = logits.softmax(dim=1)[:, 1]  # 获取类别1的概率
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    if threshold is None:
        best_th = 0.5
        best_f1 = 0.0
        for th in np.arange(0.0, 1.0, 0.01):
            preds = [1 if p > th else 0 for p in all_probs]
            f1 = f1_score(all_labels, preds)
            if f1 > best_f1:
                best_f1 = f1
                best_th = th
        threshold = best_th

    preds = [1 if p > threshold else 0 for p in all_probs]
    f1 = f1_score(all_labels, preds)
    return f1, threshold

if __name__ == "__main__":
    trainset = IntegratedEntityMatchingDataset("../data/Walmart-Amazon/train.txt")
    validset = IntegratedEntityMatchingDataset("../data/Walmart-Amazon/valid.txt")
    testset = IntegratedEntityMatchingDataset("../data/Walmart-Amazon/test.txt")

    train(trainset, validset, testset)
