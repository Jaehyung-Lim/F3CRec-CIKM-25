# Federated Continual Recommendation
for anonymous submission of KDD 2025.

Source codes
for F3CRec


we provide fine-tuned FedMF weights for base block ($\mathcal{D}^0$) in ml-100k dataset


<for fine-tuning base block>

conda env create -f enviroment.yml

python main.py --load_model 1 --backbone fedmf --model fcrec --lr 1.0 --dim 32 --patience 30 --clienet_cl --server_cl --reg_client_cl 0.1 --eps 0.006 --topN 30 --beta 0.9
