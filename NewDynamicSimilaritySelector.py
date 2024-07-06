import torch
import torch.nn as nn
import torch.nn.functional as F

class ImprovedDynamicSimilaritySelector(nn.Module):
    def __init__(self, input_dim, hidden_dim, use_learned_similarity=True):
        super().__init__()
        self.input_dim = input_dim
        self.use_learned_similarity = use_learned_similarity

        # 预定义相似度度量
        self.cosine_sim = nn.CosineSimilarity(dim=1)
        self.euclidean_dist = nn.PairwiseDistance(p=2)

        # 特征提取器
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )

        # 改进的学习型相似度计算
        if use_learned_similarity:
            self.learned_similarity = nn.Sequential(
                nn.Linear(hidden_dim + 2, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(hidden_dim // 2, hidden_dim // 4),
                nn.ReLU(),
                nn.Linear(hidden_dim // 4, 1)
            )

        # 注意力机制
        self.attention = nn.Sequential(
            nn.Linear(3 if use_learned_similarity else 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 3 if use_learned_similarity else 2),
            nn.Softmax(dim=1)
        )

    def compute_similarities(self, repr1, repr2):
        features1 = self.feature_extractor(repr1['semantic'])
        features2 = self.feature_extractor(repr2['semantic'])

        cosine_sim = self.cosine_sim(features1, features2).unsqueeze(1)
        euclidean_sim = torch.exp(-self.euclidean_dist(features1, features2)).unsqueeze(1)

        similarities = [cosine_sim, euclidean_sim]

        if self.use_learned_similarity:
            numeric_diff = torch.abs(repr1['numeric'] - repr2['numeric'])
            string_len_diff = torch.abs(torch.tensor(float(len(repr1['string']) - len(repr2['string'])))).unsqueeze(0)

            features1 = features1.view(1, -1)
            features2 = features2.view(1, -1)
            numeric_diff = numeric_diff.view(1, -1)
            string_len_diff = string_len_diff.view(1, -1)

            combined_features = torch.cat([features1, features2, numeric_diff, string_len_diff], dim=1)

            learned_sim = torch.sigmoid(self.learned_similarity(combined_features))
            similarities.append(learned_sim)

        return torch.cat(similarities, dim=1)

    def forward(self, repr1, repr2):
        similarities = self.compute_similarities(repr1, repr2)
        attention_weights = self.attention(similarities)
        weighted_sim = (attention_weights * similarities).sum()
        return weighted_sim, similarities, attention_weights

    def explain(self, repr1, repr2):
        final_sim, similarities, weights = self(repr1, repr2)
        explanation = {
            "final_similarity": final_sim.item(),
            "cosine_similarity": similarities[0, 0].item(),
            "euclidean_similarity": similarities[0, 1].item(),
            "attention_weights": weights.detach().numpy()
        }
        if self.use_learned_similarity:
            explanation["learned_similarity"] = similarities[0, 2].item()
        return explanation

# Ablation study
def ablation_study(repr1, repr2):
    model_with_learned = ImprovedDynamicSimilaritySelector(input_dim=768, hidden_dim=256, use_learned_similarity=True)
    model_without_learned = ImprovedDynamicSimilaritySelector(input_dim=768, hidden_dim=256, use_learned_similarity=False)

    sim_with, _, _ = model_with_learned(repr1, repr2)
    sim_without, _, _ = model_without_learned(repr1, repr2)

    print(f"Similarity with learned component: {sim_with.item():.4f}")
    print(f"Similarity without learned component: {sim_without.item():.4f}")
    print(f"Contribution of learned similarity: {(sim_with - sim_without).item():.4f}")

# 使用示例
if __name__ == "__main__":
    from RepresentationLayer import RepresentationLayer

    repr_layer = RepresentationLayer()

    value1 = "COL Beer_Name VAL C N Red Imperial Red Ale COL Brew_Factory_Name VAL Redwood Lodge COL Style VAL American Amber / Red Ale COL ABV VAL 8.10 %"
    value2 = "COL Beer_Name VAL Kinetic Infrared Imperial Red Ale COL Brew_Factory_Name VAL Kinetic Brewing Company COL Style VAL American Strong Ale COL ABV VAL 9.30 %"

    repr1 = repr_layer(value1)
    repr2 = repr_layer(value2)

    selector = ImprovedDynamicSimilaritySelector(input_dim=768, hidden_dim=256)

    final_sim, similarities, weights = selector(repr1, repr2)
    explanation = selector.explain(repr1, repr2)

    print("Final Similarity:", final_sim.item())
    print("Explanation:", explanation)

    # 进行ablation study
    ablation_study(repr1, repr2)