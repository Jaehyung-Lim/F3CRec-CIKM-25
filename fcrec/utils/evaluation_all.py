import torch
import numpy as np
from sklearn.metrics import ndcg_score, recall_score
import time


def ndcg(score_mat, rel_score, device):
    
    NDCG = []
    
    top_k=50
    
    _, top_k_idx = torch.topk(score_mat, top_k, dim = 1)
    sorted_rel_score, _ = torch.topk(rel_score, top_k, dim=1) # 정렬된 스코어
    
    for k in [20]:
        k_idx = top_k_idx[:, :k]
        sorted_k_rel_score = sorted_rel_score[:, :k]
        
        batch_idx = torch.arange(rel_score.size(0)).unsqueeze(1).expand(-1, k).to(device)
        
        k_rel_score = rel_score[batch_idx, k_idx]
        
        log_positions = torch.log2(torch.arange(2, k + 2, dtype = torch.float32)).to(device)
        
        dcg_k = (k_rel_score / log_positions).sum(dim=1)
        idcg_k = (sorted_k_rel_score / log_positions).sum(dim=1)
        
        tested_user = idcg_k != 0
        
        ndcg_k = dcg_k[tested_user] / idcg_k[tested_user]
        NDCG.append(round(ndcg_k.mean().detach().cpu().item(), 5))

    return NDCG


def recall(score_mat, test_mat, device):
    
    tested_user = test_mat.sum(dim=1) != 0 # test item 없는 user 제외
    y_true = test_mat[tested_user]
    
    score_mat_tested = score_mat[tested_user]
    
    y_pred = torch.zeros_like(score_mat_tested).to(device)
    
    RECALL = []
    
    for k in [20]:
        _, top_k_item_idx = torch.topk(score_mat_tested, k, dim=1)
        # 각 user 별로 top-k item에 속한 index만 1로 변경
        # y_pred[해당 원소] = 1

        batch_idx = torch.arange(top_k_item_idx.size(0)).unsqueeze(1)
        y_pred[batch_idx, top_k_item_idx] = 1.0

        y_true_ = y_true.clone().detach().cpu().numpy()
        y_pred_ = y_pred.clone().detach().cpu().numpy()

        tp = (y_true_ * y_pred_).sum(axis=1) # test item과 top_k에 속한 item중에서 겹치는 개수
        tp_fn = y_true_.sum(axis=1)  # test item의 개수
        
        RECALL_K = (tp/tp_fn).mean().item()
        
        RECALL.append(round(RECALL_K, 5))
    
    return RECALL
    

def evaluate(model, train_mat_tensor, test_mat_tensor, device):
    # model.to(device) # 이것도 풀어야됨
    model.to(device)
    model.eval()

    # prediction
    score_mat = model.get_score_mat().to('cpu')
    
    model.to('cpu')
    
    train_mask = (train_mat_tensor > 0)
    score_mat[train_mask] = -torch.inf
    
    score_mat = score_mat.to(device)
    test_mat_tensor = test_mat_tensor.to(device)
    
    NDCG = ndcg(score_mat, test_mat_tensor, device)
    RECALL = recall(score_mat, test_mat_tensor, device)
    
    NR = NDCG + RECALL
    # head = ["N@5", "N@10", "N@20", "N@50", "R@5", "R@10", "R@20", "R@50"]
    # head = ["N@10", "N@20", "R@10", "R@20"]
    head = ["N@20", "R@20"]
    RESULT = {}
    
    for i, metric in zip(NR, head):
        RESULT[metric] = i

    return RESULT


# def evaluate_batch(user_batch, model, train_mat_tensor, test_mat_tensor, device):
#     # model.to(device) # 이것도 풀어야됨
#     model.eval()

#     # prediction
#     score_mat = model.get_score_mat()[user_batch]
#     train_mat_tensor = train_mat_tensor[user_batch]
#     test_mat_tensor = test_mat_tensor[user_batch]
    
#     train_mask = (train_mat_tensor > 0)
#     score_mat[train_mask] = -torch.inf
    
#     score_mat = score_mat.to(device)
#     test_mat_tensor = test_mat_tensor.to(device)
    
#     NDCG = ndcg(score_mat, test_mat_tensor, device)
#     RECALL = recall(score_mat, test_mat_tensor, device)
    
#     NR = NDCG + RECALL
#     head = ["N@5", "N@10", "N@20", "N@50", "R@5", "R@10", "R@20", "R@50"]
    
#     RESULT = {}
    
#     for i, metric in zip(NR, head):
#         RESULT[metric] = i

#     return RESULT