# Federated Continual Recommendation
Official PyTorch implementation of our CIKM 2025 paper.

Source codes
for F3CRec


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
