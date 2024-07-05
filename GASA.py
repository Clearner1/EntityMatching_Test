import torch
import torch.nn as nn
import torch.nn.functional as F

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
        # print(f"Input x shape: {x.shape}")


        attr_attention = torch.matmul(attr_query, attr_key.transpose(-2, -1)) / (self.hidden_size ** 0.5)
        attr_attention = attr_attention.masked_fill(attr_masks.unsqueeze(1) == 0, -1e9)
        attr_attention = F.softmax(attr_attention, dim=-1)
        # print(f"Attr_masks shape: {attr_masks.shape}")

        attr_context = torch.matmul(attr_attention, attr_value)
        # print(f"Attr_context shape: {attr_context.shape}")

        # 序列注意力
        seq_feature = self.conv(x.transpose(1, 2)).transpose(1, 2)
        seq_pool = torch.cat([
            F.adaptive_avg_pool1d(seq_feature.transpose(1, 2), 1).squeeze(-1),
            F.adaptive_max_pool1d(seq_feature.transpose(1, 2), 1).squeeze(-1)
        ], dim=-1)
        seq_attention = self.seq_attention(seq_pool).unsqueeze(1)
        # print(f"Seq_attention shape: {seq_attention.shape}")

        # 融合
        fused_attention = self.fusion(torch.cat([
            attr_context,
            seq_attention.expand(-1, x.size(1), -1)
        ], dim=-1))

        output = fused_attention + x # 使用残差连接

        # print(f"Output shape: {output.shape}")
        return output
