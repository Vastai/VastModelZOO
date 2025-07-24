# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-
'''
@Author :          lance
@Email : lance.wang@vastaitech.com
@Time  : 	2025/04/21 19:43:31
'''

import torch
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics.pairwise import (
    paired_cosine_distances,
    paired_euclidean_distances,
    paired_manhattan_distances,
)
from sklearn.metrics import average_precision_score


class STSEvaluator():
    def __init__(
        self,
        embeddings1,
        embeddings2,
        gold_scores,
        min_score,
        max_score,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.embeddings1 = embeddings1
        self.embeddings2 = embeddings2
        self.gold_scores = gold_scores
        self.min_score=min_score
        self.max_score=max_score

    def cosine_similarity(self, A, B):
        return np.dot(A.flatten(), B.flatten()) / (np.linalg.norm(A) * np.linalg.norm(B))
    
    def normalize(self, x):
        return (x - self.min_score) / (self.max_score - self.min_score)

    def __call__(self):

        self.gold_scores = list(map(self.normalize, self.gold_scores))

        cosine_scores = 1 - (paired_cosine_distances(self.embeddings1, self.embeddings2))
        manhattan_distances = -paired_manhattan_distances(self.embeddings1, self.embeddings2)
        euclidean_distances = -paired_euclidean_distances(self.embeddings1, self.embeddings2)

        cosine_pearson, _ = pearsonr(self.gold_scores, cosine_scores)
        cosine_spearman, _ = spearmanr(self.gold_scores, cosine_scores)

        manhatten_pearson, _ = pearsonr(self.gold_scores, manhattan_distances)
        manhatten_spearman, _ = spearmanr(self.gold_scores, manhattan_distances)

        euclidean_pearson, _ = pearsonr(self.gold_scores, euclidean_distances)
        euclidean_spearman, _ = spearmanr(self.gold_scores, euclidean_distances)

        similarity_scores = None

        _similarity_scores = [
            float(self.cosine_similarity(e1, e2))  # type: ignore
            for e1, e2 in zip(self.embeddings1, self.embeddings2)
        ]
        similarity_scores = np.array(_similarity_scores)

        if similarity_scores is not None:
            pearson, _ = pearsonr(self.gold_scores, similarity_scores)
            spearman, _ = spearmanr(self.gold_scores, similarity_scores)
        else:
            # if model does not have a similarity function, we assume the cosine similarity
            pearson = cosine_pearson
            spearman = cosine_spearman

        return {
            # using the models own similarity score
            "pearson": pearson,
            "spearman": spearman,
            # generic similarity scores
            "cosine_pearson": cosine_pearson,
            "cosine_spearman": cosine_spearman,
            "manhattan_pearson": manhatten_pearson,
            "manhattan_spearman": manhatten_spearman,
            "euclidean_pearson": euclidean_pearson,
            "euclidean_spearman": euclidean_spearman,
        }



class RerankingEvaluator():
    def __init__(
        self,
        infer_data,
        mrr_at_k=10,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.infer_data = infer_data
        self.mrr_at_k = mrr_at_k

    def mrr_at_k_score(sef, is_relevant, pred_ranking, k):
        mrr_score = 0
        for rank, index in enumerate(pred_ranking[:k]):
            if is_relevant[index]:
                mrr_score = 1 / (rank + 1)
                break
        return mrr_score

    def ap_score(self, is_relevant, pred_scores):
        ap = average_precision_score(is_relevant, pred_scores)
        return ap
        
    def __call__(self):


        all_mrr_scores = []
        all_ap_scores = []

        for data in self.infer_data:
            pos_num = len(data['query_positive'])
            nega_num = len(data['query_negative'])
            is_relevant = [True] * pos_num + [False] * nega_num
            
            pre_score = []
            for score in data['query_positive_score']:
                pre_score.append(score[0])
            for score in data['query_negative_score']:
                pre_score.append(score[0])

            pred_scores_argsort = torch.argsort(-torch.from_numpy(np.array(pre_score)))
            # mrr = self.mrr_at_k_score(is_relevant, pred_scores_argsort, self.mrr_at_k)
            ap = self.ap_score(is_relevant, pre_score)
            # all_mrr_scores.append(mrr)
            all_ap_scores.append(ap)

        mean_ap = np.mean(all_ap_scores)
        # mean_mrr = np.mean(all_mrr_scores)

        return {"map": mean_ap}
