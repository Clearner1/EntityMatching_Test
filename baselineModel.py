from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from sklearn import metrics

from tensorboardX import SummaryWriter
import torch.optim as optim
from torch.utils import data
from torch.utils.data import Dataset
from transformers import RobertaModel, RobertaTokenizer, get_linear_schedule_with_warmup
import os

batch_size = 64
n_epochs = 40
save_model = True
logdir = r"Structured/"
task = "Structured/Beer"
run_tag = '%s' % (task)
run_tag = run_tag.replace('/', '_')
class EntityMatchingModel(nn.Module):
    def __init__(self, device='cuda'):
        super().__init__()
        self.bert = RobertaModel.from_pretrained('roberta-base')
        self.device = device
        # linear layer
        hidden_size = self.bert.config.hidden_size
        # todo  输入768 输出2 归类为2分类任务
        self.fc = torch.nn.Linear(hidden_size, 2)

    def forward(self, x1):
        x1 = x1.to(self.device) # (batch_size, seq_len)
        enc = self.bert(x1)[0][:, 0, :]
        return self.fc(enc)


class EntityMatchingDataset(Dataset):
    def __init__(self, data_path, max_len=64):
        self.pairs = []
        self.labels = []
        self.max_len = max_len
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
        # todo ?
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

        # left + right 将文本对 (left 和 right) 编码成 token ID 的序列
        x = self.tokenizer.encode(text=left,
                                  text_pair=right,
                                  max_length=self.max_len,
                                  truncation=True)
        return x, self.labels[idx]

    @staticmethod
    def pad(batch):
        x12, y = zip(*batch)
        maxlen = max([len(x) for x in x12])
        x12 = [xi + [0] * (maxlen - len(xi)) for xi in x12]
        return torch.LongTensor(x12), torch.LongTensor(y)


def train(trainset, validset, testset):
    padder = trainset.pad  #为了能在一个批次中同时处理多个样本，通常需要将所有样本的长度统一到相同的长度。
    # create the DataLoaders 加载DataLoader
    train_iter = data.DataLoader(dataset=trainset,
                                 batch_size=batch_size,
                                 shuffle=True,
                                 num_workers=4,
                                 collate_fn=padder)
    valid_iter = data.DataLoader(dataset=validset,
                                 batch_size=batch_size * 16,
                                 shuffle=False,
                                 num_workers=4,
                                 collate_fn=padder)
    test_iter = data.DataLoader(dataset=testset,
                                batch_size=batch_size * 16,
                                shuffle=False,
                                num_workers=4,
                                collate_fn=padder)

    # initialize model, optimizer, and LR scheduler
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = EntityMatchingModel(device=device)
    model = model.to(device)  # 模型的所有参数和缓冲区都移动到了 GPU 上
    optimizer = optim.AdamW(model.parameters(), lr=3e-5)

    #     训练的总步数
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
    for epoch in range(1, n_epochs + 1):
        # train 模型会切换到训练模式:执行前向传播、计算损失、反向传播和优化步骤。
        model.train()
        train_step(train_iter, model, optimizer, scheduler)

        # eval 设置模型为评估模式
        model.eval()  # 禁用梯度计算
        dev_f1, th = evaluate(model, valid_iter)
        test_f1 = evaluate(model, test_iter, threshold=th)
        #  如果大于最佳的验证集数值会保存模型
        if dev_f1 > best_dev_f1:
            best_dev_f1 = dev_f1
            best_test_f1 = test_f1
            if save_model:
                # create the directory if not exist
                directory = os.path.join(logdir, task)
                if not os.path.exists(directory):
                    os.makedirs(directory)

                # save the checkpoints for each component
                ckpt_path = os.path.join(logdir, task, 'model.pt')
                ckpt = {'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'epoch': epoch}
                torch.save(ckpt, ckpt_path)
        # Validation Set -> dev  测试集 和 验证集
        print(f"epoch {epoch}: dev_f1={dev_f1}, f1={test_f1}, best_test_f1={best_test_f1}, best_dev_f1={dev_f1}")

        # logging
        scalars = {'f1': dev_f1,
                   't_f1': test_f1}
        writer.add_scalars(run_tag, scalars, epoch)

    writer.close()


def evaluate(model, iterator, threshold=None):
    all_p = []
    all_y = []
    all_probs = []
    with torch.no_grad():
        for batch in iterator:
            x, y = batch
            logits = model(x)
            probs = logits.softmax(dim=1)[:, 1]
            all_probs += probs.cpu().numpy().tolist()
            all_y += y.cpu().numpy().tolist()

    if threshold is not None:
        pred = [1 if p > threshold else 0 for p in all_probs]
        f1 = metrics.f1_score(all_y, pred)
        return f1
    else:
        best_th = 0.5
        f1 = 0.0  # metrics.f1_score(all_y, all_p)

        for th in np.arange(0.0, 1.0, 0.05):
            pred = [1 if p > th else 0 for p in all_probs]
            new_f1 = metrics.f1_score(all_y, pred)
            if new_f1 > f1:
                f1 = new_f1
                best_th = th

        return f1, best_th


def train_step(train_iter, model, optimizer, scheduler):
    criterion = nn.CrossEntropyLoss()
    for i, batch in enumerate(train_iter):
        optimizer.zero_grad()
        x, y = batch
        prediction = model(x)
        loss = criterion(prediction, y.to(model.device))
        loss.backward()
        optimizer.step()
        scheduler.step()
        # if i % 10 == 0: # monitoring
        print(f"step: {i}, loss: {loss.item()}")
        del loss


if __name__ == "__main__":
    trainset = r"Beer\train.txt"
    validset = r"Beer\valid.txt"
    testset = r"Beer\test.txt"
    train_dataset = EntityMatchingDataset(trainset)
    valid_dataset = EntityMatchingDataset(validset)
    test_dataset = EntityMatchingDataset(testset)

    train(train_dataset, valid_dataset, test_dataset)
