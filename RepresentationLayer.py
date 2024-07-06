import torch
import torch.nn as nn
from transformers import AutoModel,AutoTokenizer


class RepresentationLayer(nn.Module):
    def __init__(self, roberta_model_name='roberta-base'):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

    def numeric_representation(self, value):
        # 如果值是数字,直接返回;否则返回长度
        try:
            return torch.tensor(float(value)).unsqueeze(0)
        except ValueError:
            return torch.tensor(float(len(value))).unsqueeze(0)

    def string_representation(self, value):
        # 简单地返回字符串的字符编码
        return torch.tensor([ord(c) for c in value])

    def semantic_representation(self, value):
        # 使用RoBERTa获取语义表示
        inputs = self.tokenizer(value, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.roberta(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # 使用平均池化

    def forward(self, value):
        num_repr = self.numeric_representation(value)
        str_repr = self.string_representation(value)
        sem_repr = self.semantic_representation(value)

        return {
            'numeric': num_repr,
            'string': str_repr,
            'semantic': sem_repr
        }

if __name__ == "__main__":
    # 使用示例
    repr_layer = RepresentationLayer()
    value = "Example Value 123"
    representations = repr_layer(value)

    print("Numeric representation:", representations['numeric'])
    print("String representation shape:", representations['string'].shape)
    print("Semantic representation shape:", representations['semantic'].shape)