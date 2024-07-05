import torch
from torch import nn
import torch.nn.functional as F
# （Multi-view Enhanced Channel and Sequence）
# （Multi-view Enhanced Feature and Sequence）
# 适用于那些需要同时考虑局部和全局特征的NLP任务
def global_median_pooling(x):
    return torch.median(x, dim=1)[0]

class FeatureAttention(nn.Module):
    def __init__(self, input_dim, internal_neurons):
        super(FeatureAttention, self).__init__()
        self.fc1 = nn.Linear(input_dim, internal_neurons)
        self.fc2 = nn.Linear(internal_neurons, input_dim)

    def forward(self, inputs):
        avg_pool = torch.mean(inputs, dim=1)
        max_pool = torch.max(inputs, dim=1)[0]
        median_pool = global_median_pooling(inputs)

        avg_out = self.fc2(F.relu(self.fc1(avg_pool), inplace=False))
        max_out = self.fc2(F.relu(self.fc1(max_pool), inplace=False))
        median_out = self.fc2(F.relu(self.fc1(median_pool), inplace=False))

        out = torch.sigmoid(avg_out + max_out + median_out)
        return out.unsqueeze(1)

class MEFS_NLP(nn.Module):
    def __init__(self, input_dim, hidden_dim, feature_attention_reduce=4, dropout_rate=0.5):
        super(MEFS_NLP, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        assert input_dim == hidden_dim, "Input and hidden dimensions must be the same"

        self.feature_attention = FeatureAttention(input_dim=input_dim, internal_neurons=input_dim // feature_attention_reduce)
        # self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True, dropout=dropout_rate, num_layers=2)
        # 通常在LSTM之后也可以添加一个Dropout层
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_dim * 2, hidden_dim)

    def forward(self, inputs):
        feature_att_vec = self.feature_attention(inputs)
        inputs = feature_att_vec * inputs
        lstm_out, _ = self.lstm(inputs)
        lstm_out = self.dropout(lstm_out)
        out = self.fc(lstm_out)
        return out

if __name__ == '__main__':
    batch_size = 4
    seq_len = 64
    embedding_dim = 128
    input_tensor = torch.randn(batch_size, seq_len, embedding_dim).cuda()

    MEFS_NLP_block = MEFS_NLP(input_dim=128, hidden_dim=128, feature_attention_reduce=4).cuda()

    output_tensor = MEFS_NLP_block(input_tensor)

    print(f"Input shape: {input_tensor.shape}")
    print(f"Output shape: {output_tensor.shape}")