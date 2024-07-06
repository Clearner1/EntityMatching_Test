import torch
import torch.nn as nn
from transformers import AutoModel, AutoTokenizer

class RepresentationLayer(nn.Module):
    def __init__(self, roberta_model_name='roberta-base'):
        super().__init__()
        self.roberta = AutoModel.from_pretrained(roberta_model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)

    def numeric_representation(self, value):
        # 返回一个 2D 张量
        try:
            return torch.tensor([[float(value)]])
        except ValueError:
            return torch.tensor([[float(len(value))]])

    def string_representation(self, value):
        # 返回字符串的字符编码，作为一个 1D 张量
        return torch.tensor([ord(c) for c in value])

    def semantic_representation(self, value):
        # 使用RoBERTa获取语义表示
        inputs = self.tokenizer(value, return_tensors="pt", padding=True, truncation=True, max_length=128)
        with torch.no_grad():
            outputs = self.roberta(**inputs)
        return outputs.last_hidden_state.mean(dim=1)  # 使用平均池化，返回 2D 张量

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
    value = "sadsadasd啊是的123"
    representations = repr_layer(value)

    print("Numeric representation:", representations['numeric'])
    print("String representation shape:", representations['string'].shape)
    print("Semantic representation shape:", representations['semantic'].shape)