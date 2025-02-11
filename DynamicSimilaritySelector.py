import torch
import torch.nn as nn
import torch.nn.functional as F

class EnhancedDynamicSimilaritySelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_similarity_measures=4):
        super().__init__()
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        self.selector = nn.Linear(hidden_dim // 2, num_similarity_measures)
        self.similarity_functions = nn.ModuleList([
            nn.Sequential(
                nn.Linear(768 * 2, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)
            ) for _ in range(num_similarity_measures)
        ])

    def extract_features(self, repr1, repr2):
        # 更复杂的特征提取
        numeric_diff = torch.abs(repr1['numeric'] - repr2['numeric'])
        string_len_diff = torch.abs(torch.tensor(len(repr1['string']) - len(repr2['string'])).float()).unsqueeze(0)
        semantic_cos = F.cosine_similarity(repr1['semantic'], repr2['semantic'], dim=-1).unsqueeze(0)
        return torch.cat([numeric_diff, string_len_diff, semantic_cos,
                          repr1['semantic'].flatten(), repr2['semantic'].flatten()], dim=-1)

    def compute_similarities(self, repr1, repr2):
        combined = torch.cat([repr1['semantic'], repr2['semantic']], dim=-1)
        return torch.cat([f(combined) for f in self.similarity_functions], dim=-1)

    def forward(self, repr1, repr2):
        features = self.extract_features(repr1, repr2)
        selector_features = self.feature_extractor(features)
        weights = F.softmax(self.selector(selector_features), dim=-1)

        similarities = self.compute_similarities(repr1, repr2)

        weighted_sim = (weights * similarities).sum()

        return weighted_sim, weights, similarities


# 使用示例
if __name__ == "__main__":
    from RepresentationLayer import RepresentationLayer

    repr_layer = RepresentationLayer()
    selector = EnhancedDynamicSimilaritySelector(input_dim=1538, hidden_dim=256)


    # 元学习适应

    # 测试
    value1 = "Test Value 123"
    value2 = "Test Value 456"

    repr1 = repr_layer(value1)
    repr2 = repr_layer(value2)

    similarity, weights, raw_similarities = selector(repr1, repr2)

    print(f"Overall Similarity: {similarity.item():.4f}")
    print(f"Weights: {weights.detach().numpy()}")
    print(f"Raw Similarities: {raw_similarities.detach().numpy()}")
