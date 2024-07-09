# import torch
# import re
# import torch.nn as nn
# from transformers import AutoModel, AutoTokenizer
#
# class RepresentationLayer(nn.Module):
#     def __init__(self, roberta_model_name='roberta-base'):
#         super().__init__()
#         self.roberta = AutoModel.from_pretrained(roberta_model_name)
#         self.tokenizer = AutoTokenizer.from_pretrained(roberta_model_name)
#
#     def numeric_representation(self, input_ids):
#         input_ids_list = input_ids.squeeze().tolist()
#         numeric_tokens = []
#         for tid in input_ids_list:
#             if tid != 0:
#                 decoded_str = self.tokenizer.decode(tid)
#                 match = re.search(r'-?\d+\.?\d*', decoded_str)
#                 if match:
#                     numeric_tokens.append(float(match.group()))
#
#         if len(numeric_tokens) > 0:
#             return torch.tensor([sum(numeric_tokens) / len(numeric_tokens)])
#         else:
#             return torch.tensor([0.0])
#
#     def string_representation(self, input_ids):
#         # 将 token IDs 转为字符串表示的张量
#         return input_ids.float()
