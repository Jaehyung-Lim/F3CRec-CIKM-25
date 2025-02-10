import torch
import numpy as np
from sklearn.metrics import ndcg_score, recall_score
import time


def ndcg(score_mat, rel_score, top_k):
    _, top_k_idx = torch.topk(score_mat, top_k, dim = 1)
    sorted_rel_score, _ = torch.topk(rel_score, top_k, dim=1)    


    batch_idx = torch.arange(rel_score.size(0)).unsqueeze(1).expand(-1, top_k)
    
    top_k_rel_score = rel_score[batch_idx, top_k_idx] 

    log_positions = torch.log2(torch.arange(2, top_k + 2, dtype = torch.float32))
    
    
    dcg_k = (top_k_rel_score / log_positions).sum(dim=1)
    idcg_k = (sorted_rel_score / log_positions).sum(dim=1)

    tested_user = idcg_k != 0
    
    # idcg = 0인 애들은, 해당 test에서 포함되지 않는 애니까 제외해야됨
    ndcg_k = dcg_k[tested_user] / idcg_k[tested_user]
    NDCG = ndcg_k.mean().detach().cpu()

    return NDCG


def recall(score_mat, test_mat, top_k):
    # score_mat에서 top_k로 뽑힌 애들은 모두 1로 변환
    # test_mat에서 y_true로 넣음
    # 그 전에 test_mat_tested 선택해서 score_mat[test_mat_tested]
    
    tested_user = test_mat.sum(dim=1) != 0 # test item 없는 user 제외
    y_true = test_mat[tested_user]
    
    score_mat_tested = score_mat[tested_user]
    
    y_pred = torch.zeros_like(score_mat_tested)
    
    _, top_k_item_idx = torch.topk(score_mat_tested, top_k, dim=1)
    # 각 user 별로 top-k item에 속한 index만 1로 변경
    # y_pred[해당 원소] = 1

    batch_idx = torch.arange(top_k_item_idx.size(0)).unsqueeze(1)
    y_pred[batch_idx, top_k_item_idx] = 1.0

    y_true = y_true.detach().cpu().numpy()
    y_pred = y_pred.detach().cpu().numpy()

    tp = (y_true * y_pred).sum(axis=1) # test item과 top_k에 속한 item중에서 겹치는 개수
    tp_fn = y_true.sum(axis=1)  # test item의 개수
    
    RECALL = (tp/tp_fn).mean()
    
    return RECALL


def evaluate(model, train_mat_tensor, test_mat_tensor, top_k, device):
    # model.to(device) # 이것도 풀어야됨
    model.eval()

    # prediction
    score_mat = model.get_score_mat()
    train_mask = (train_mat_tensor > 0)
    score_mat[train_mask] = -torch.inf
    
    # 여기서 score_mat 중에서 train_mat_tensor의 값이 1인 애들은 -inf로 변환해야됨

    # true
    # test_mat_tensor = test_mat_tensor.to(device) # 이거 풀어야됨
    # NDCG
    # test_mat_tensor = true_relevance_score
    
    NDCG = ndcg(score_mat, test_mat_tensor, top_k)
    
    # RECALL
    RECALL = recall(score_mat, test_mat_tensor, top_k)

    return NDCG, RECALL