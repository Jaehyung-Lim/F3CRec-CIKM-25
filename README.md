# F3CRec
for anonymous submission of KDD 2025.

Source codes
for F3CRec

models.py
train.py
we provide pre-trained BPR for ml-1m dataset
for fine-tuning LLMs

finetune_hug.py
we provide fine-tuned FedMF, FedNCF, PFedRec weights for base block ($\mathcal{D}^0$) in ml-100k dataset

utils.py
uncertainty_hug.py
How to run
for zero-shot ranking

python uncertainty_hug_zs.py --nper 5 --nhist 20 --ncan 20 --bs 5 --ind Candidate --ind_env [ ] --ind_sym A --title_env "'" "'" --dataset ml-1m --model meta-llama/Meta-Llama-3.1-8B-Instruct
for fine-tuned ranking

python uncertainty_hug.py --nper 5 --nhist 20 --ncan 20 --bs 5 --ind Candidate --ind_env [ ] --ind_sym A --title_env "'" "'" --dataset ml-1m --model meta-llama/Meta-Llama-3.1-8B-Instruct --model_ft model/ml-1m/Meta-Llama-3.1-8B-Instruct_3


<for fine-tuning base block>
