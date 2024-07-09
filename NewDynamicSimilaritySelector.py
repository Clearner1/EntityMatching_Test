# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
# class ImprovedDynamicSimilaritySelector(nn.Module):
#     def __init__(self, hidden_dim):
#         super().__init__()
#         self.hidden_dim = hidden_dim
#
#         # 语义相似度
#         self.semantic_sim = nn.CosineSimilarity(dim=1)
#
#         # 字符串相似度
#         self.string_sim = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
#
#         # 数值相似度
#         self.numeric_sim = nn.Sequential(
#             nn.Linear(2, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 1),
#             nn.Sigmoid()
#         )
#
#         # 注意力机制
#         self.attention = nn.Sequential(
#             nn.Linear(3, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, 3),
#             nn.Softmax(dim=1)
#         )
#
#     def forward(self, repr1, repr2):
#         # 语义相似度
#         semantic_sim = self.semantic_sim(repr1['semantic'], repr2['semantic']).unsqueeze(1)
#
#         # 字符串相似度
#         string_features = torch.abs(repr1['string'].float() - repr2['string'].float())
#         string_sim = self.string_sim(string_features.mean(dim=0).unsqueeze(0))
#
#         # 数值相似度
#         numeric_features = torch.cat([repr1['numeric'], repr2['numeric']], dim=1)
#         numeric_sim = self.numeric_sim(numeric_features)
#
#         # 组合相似度
#         combined_sims = torch.cat([semantic_sim, string_sim, numeric_sim], dim=1)
#
#         # 注意力加权
#         attn_weights = self.attention(combined_sims)
#         weighted_sim = torch.sum(attn_weights * combined_sims, dim=1)
#
#         return weighted_sim, combined_sims, attn_weights
#
#     def explain(self, repr1, repr2):
#         weighted_sim, combined_sims, attn_weights = self(repr1, repr2)
#         explanation = {
#             "final_similarity": weighted_sim.item(),
#             "semantic_similarity": combined_sims[0, 0].item(),
#             "string_similarity": combined_sims[0, 1].item(),
#             "numeric_similarity": combined_sims[0, 2].item(),
#             "attention_weights": attn_weights.detach().numpy()
#         }
#         return explanation