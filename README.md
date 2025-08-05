# Federated Continual Recommendation
Official PyTorch implementation of our CIKM 2025 paper.

## 1. Overview
This repository provides the source code of our paper: Federated Continual Recommendation, accepted in CIKM'25 as a full research paper.

In the paper, we first propose novel research task "Federated Continual Recommendation". We deal with Federated Recommendation and Continual Recommendation 동시에. 그리고 각 Recommendation task가 가지고 있는 고유한 constraint를 고려하여, 이를 해결하고자 함.

![Task definition](task_concept.png)

Existing Continual Recommendation 모델과 Federated Recommendation 모델은 양립 불가능함.
이를 해결하기 위해, we propose two continual learning methods:
1. Client-side continual learning: Adaptive Replay Memory
2. Server-side continual learning: Item-wise Temporal Mean

![Task definition](method_figure.png)


Source codes


We provide fine-tuned FedMF weights for base block ($\mathcal{D}^0$) in ml-100k dataset


- enviroment setting
```
conda env create -f enviroment.yml
```

- fine-tuning base block
```
python main.py --save_model 1 --load_model 0 --backbone fedmf --model fcrec --lr 1.0 --dim 32 --patience 30 --client_cl --server_cl --reg_client_cl 0.1 --eps 0.006 --topN 30 --beta 0.9 --num_round 100 --dataset ml-100k
```

- training incremental block
```
python3 main.py --load_model 1 --backbone fedmf --model fcrec --lr 1.0 --dim 32 --patience 30 --client_cl --server_cl --reg_client_cl 0.1 --eps 0.006 --topN 30 --beta 0.9 --num_round 100 --dataset ml-100k
```
