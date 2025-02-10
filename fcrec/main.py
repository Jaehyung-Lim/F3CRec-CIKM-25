import random
import numpy as np

import torch
import torch.backends.cudnn as cudnn

from torch.utils.data import DataLoader
from models.fcrec import *

from utils.util import *
from utils.data_util import *
from utils.data_loader import *

import argparse
from phe import paillier

# ===================================================================== FedBPR =====================================================================
from baseline.fedbpr.fedbpr import *
from baseline.fedbpr.fedbpr_reloop2 import *
from baseline.fedbpr.fedbpr_naive_replay import *
from baseline.fedbpr.fedbpr_full_batch import *
from baseline.fedbpr.fedbpr_topn_kd import *
from baseline.fedbpr.fedbpr_ft import *
from baseline.fedbpr.fedbpr_emb_reg import *
from baseline.fedbpr.bpr import *

# ===================================================================== GPFedRec =====================================================================
from baseline.gpfedrec.gpfedrec_ft import *

# ===================================================================== FedMF =====================================================================
from baseline.fedmf.fedmf import *
from baseline.fedmf.fedmf_ft import *
from baseline.fedmf.fedmf_emb_reg import *
from baseline.fedmf.fedmf_full_batch import *
from baseline.fedmf.fedmf_naive_replay import *
from baseline.fedmf.fedmf_topn_kd import *
from baseline.fedmf.fedmf_reloop2 import *
from baseline.fedmf.fedmf_spp import *
from baseline.fedmf.fedmf_ewc import *
# ===================================================================== FedMLP =====================================================================
from baseline.fedmlp.fedmlp_ft import *
from baseline.fedmlp.fedmlp_emb_reg import *
from baseline.fedmlp.fedmlp_full_batch import *
from baseline.fedmlp.fedmlp_naive_replay import *
from baseline.fedmlp.fedmlp_topn_kd import *
from baseline.fedmlp.fedmlp import *
from baseline.fedmlp.fedmlp_reloop2 import *
from baseline.fedmlp.fedmlp_spp import *
from baseline.fedmlp.fedmlp_ewc import *

# ===================================================================== PFedRec =====================================================================
from baseline.pfedrec.pfedrec_ft import *
from baseline.pfedrec.pfedrec import *
from baseline.pfedrec.pfedrec_reloop2 import *
from baseline.pfedrec.pfedrec_full_batch import *
from baseline.pfedrec.pfedrec_naive_replay import *
from baseline.pfedrec.pfedrec_emb_reg import *
from baseline.pfedrec.pfedrec_topn_kd import *
from baseline.pfedrec.pfedrec_spp import *
from baseline.pfedrec.pfedrec_ewc import *

def baseline(args):
    args.num_task = 4 
    
    device = get_device(args, threshold=500)

    data_block_path = f"/home/jaehyunglim/fcrec2025/fcrec_data/dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"/home/jaehyunglim/fcrec2025/fcrec_data/dataset/{args.dataset}"
    
    total_blocks = load_pickle(data_block_path)
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)

    num_user_item_info = get_num_user_item_(total_blocks)

    args.num_user, args.num_item = num_user_item_info['TASK_0']['num_user'], num_user_item_info['TASK_0']['num_item']
    

    args.layers = [args.dim * 2, args.dim // 2]
       
    # ===================================================================== FedMF =====================================================================    
    if args.baseline == 'fedmf_reloop2':
        engine = FedMF_Reloop2_Engine(args, device)
    elif args.baseline == 'fedmf_naive_replay':
        engine = FedMF_Naive_Replay(args, device)
    elif args.baseline == 'fedmf_full_batch':
        engine = FedMF_Full_Batch(args, device)
    elif args.baseline == 'fedmf_topn_kd':
        engine = FedMF_TopN_KD_Engine(args, device)
    elif args.baseline == 'fedmf_ft':
        engine = FedMF_FT_Engine(args, device)
    elif args.baseline == 'fedmf_emb_reg':
        engine = FedMF_Emb_Reg_Engine(args, device)  
    elif args.baseline == 'fedmf_spp':
        engine = FedMF_SPP_Engine(args, device)
    elif args.baseline == 'fedmf_ewc':
        engine = FedMF_EWC_Engine(args, device)
    
    
    # ===================================================================== FedMLP =====================================================================    
    elif args.baseline == 'fedmlp_ft':
        engine = FedMLP_FT_Engine(args, device)
    elif args.baseline == 'fedmlp_emb_reg':
        engine = FedMLP_Emb_Reg_Engine(args, device)
    elif args.baseline == 'fedmlp_full_batch':
        engine = FedMLP_Full_Batch_Engine(args, device)
    elif args.baseline == 'fedmlp_naive_replay':
        engine = FedMLP_Naive_Replay_Engine(args, device)
    elif args.baseline == 'fedmlp_topn_kd':
        engine = FedMLP_TopN_KD_Engine(args, device)
    elif args.baseline == 'fedmlp_reloop2':
        engine = FedMLP_Reloop2_Engine(args, device)
    elif args.baseline == 'fedmlp_spp':
        engine = FedMLP_SPP_Engine(args, device)
    elif args.baseline == 'fedmlp_ewc':
        engine = FedMLP_EWC_Engine(args, device)
        
    # ===================================================================== PFedRec =====================================================================    
    elif args.baseline == 'pfedrec_ft':
        engine = PFedRec_FT_Engine(args, device)
    elif args.baseline == 'pfedrec_emb_reg':
        engine = PFedRec_Emb_Reg_Engine(args, device)
    elif args.baseline == 'pfedrec_topn_kd':
        engine = PFedRec_TopN_KD_Engine(args, device)
    elif args.baseline == 'pfedrec_full_batch':
        engine = PFedRec_Full_Batch_Engine(args, device)
    elif args.baseline == 'pfedrec_naive_replay':
        engine = PFedRec_Naive_Replay_Engine(args, device)
    elif args.baseline == 'pfedrec_reloop2':
        engine = PFedRec_Reloop2_Engine(args, device)
    elif args.baseline == 'pfedrec_spp':
        engine = PFedRec_SPP_Engine(args, device)
    elif args.baseline == 'pfedrec_ewc':
        engine = PFedRec_EWC_Engine(args, device)
        
    # ===================================================================== PFedRec ===================================================================== 
    elif args.baseline == 'gpfedrec_ft':
        engine = GPFedRec_FT_Engine(args, device)
    
    
    if args.load_model == 1:
        engine = load_baseblock_param(args)
        engine.args = args
    
        
    input_total_data = [total_train_dataset, total_valid_dataset, total_test_dataset]
    
    for task in range(args.num_task):   
        is_base = True if task == 0 else False    
        
        if args.load_model == 1 and is_base:
            # 이미 저장된 모델은 base 학습 했으므로 continue 이번꺼 패스
            continue
        
        if not is_base:
            args.num_user, args.num_item = num_user_item_info[f'TASK_{task}']['num_user'], num_user_item_info[f'TASK_{task}']['num_item']
            engine.add_user_item(args.num_user, args.num_item)
        
              
        engine.run(task, input_total_data, is_base)
        
        if is_base and args.save_model == 1:
            save_baseblock_param(args, engine)
            print(f"Save complete")
            break



def fcrec(args):
    args.num_task = 4 
    
    device = get_device(args, threshold=500)

    data_block_path = f"/home/jaehyunglim/fcrec2025/fcrec_data/dataset/{args.dataset}/total_blocks_timestamp.pickle"
    data_dict_path = f"/home/jaehyunglim/fcrec2025/fcrec_data/dataset/{args.dataset}"
    
    total_blocks = load_pickle(data_block_path)
    total_train_dataset, total_valid_dataset, total_test_dataset, total_item_list = load_data_as_dict(data_dict_path, num_task = args.num_task)

    num_user_item_info = get_num_user_item_(total_blocks)

    args.num_user, args.num_item = num_user_item_info['TASK_0']['num_user'], num_user_item_info['TASK_0']['num_item']
    
    is_shuffle = False

    #args.layers = [args.dim * 2, args.dim, args.dim // 2, args.dim // 4]
    args.layers = [args.dim * 2, args.dim // 2]
    
    # ===================================================================== FedMF =====================================================================    
    if args.baseline == 'fedmf_fcrec': 
        engine = FedMF_Engine(args, device)
    
    # ===================================================================== FedMLP =====================================================================    
    elif args.baseline == 'fedmlp_fcrec':
        engine = FedMLP_Engine(args, device)
        
    # ===================================================================== PFedRec =====================================================================    
    elif args.baseline == 'pfedrec_fcrec':
        engine = PFedRec_Engine(args, device)
    
    
    input_total_data = [total_train_dataset, total_valid_dataset, total_test_dataset]
    
    if args.load_model == 0 and args.save_model == 1:
        # fcrec baseblock 학습해서 넣고 싶을 때
        engine.run(task=0, input_total_data=input_total_data, is_base = True)
        save_baseblock_param(args, engine)
        print("Save complete")
        exit()
    
    elif args.load_model == 1 and args.save_model == 0:
        engine = load_baseblock_param(args) # 이때 모든 파라미터가 싹 다 저장되는지? ex) self.old_user_emb, self.old_item_emb 등등 -> 이거 확인해봐야 할 듯
        engine.args = args
        
        # 이미 task 0은 한거니까 제외
        for task in range(1, args.num_task):
            is_base = False
        
            args.num_user, args.num_item = num_user_item_info[f'TASK_{task}']['num_user'], num_user_item_info[f'TASK_{task}']['num_item']
            engine.add_user_item(args.num_user, args.num_item)
            
            engine.run(task, input_total_data, is_base)
        
    
    elif args.load_model == 0 and args.save_model == 0: # 첨부터 학습할 때
        for task in range(args.num_task):   
            is_base = True if task == 0 else False    
            
            if not is_base:
                args.num_user, args.num_item = num_user_item_info[f'TASK_{task}']['num_user'], num_user_item_info[f'TASK_{task}']['num_item']
                engine.add_user_item(args.num_user, args.num_item)
            
            engine.run(task, input_total_data, is_base)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--show_valid', action='store_true', default=False)
    parser.add_argument('--save_model', type = int, default=0, help='0: not save, 1: save')
    parser.add_argument('--load_model', type = int, default=0, help='0: scratch, 1: load')
    parser.add_argument('--save_result', type = int, default=0, help='0: not save, 1: save')
    
    parser.add_argument('--dataset', '--dd', type = str, default = 'lastfm-2k', choices = ['Yelp', "Gowalla", 'ml-1m', 'ml-100k', 'hetrec2011', 'lastfm-2k','ml-latest-small'])
    parser.add_argument('--optimizer', '--o', type = str, default = 'SGD', choices=["Adam", "SGD"]) # Base는 Adam이 더 잘 나오는 듯
    parser.add_argument('--dim', '--d', type=int, default = 32)
    
    parser.add_argument('--clients_sample_ratio', '--c', type=int, default=1.0, help='# participants in a round')
    parser.add_argument('--batch_size', '--b', type=int, default=512, help='# item in a batch')
    parser.add_argument('--num_round', '--r', type=int, default=150, help='fcf: 150, the other: 100')
    parser.add_argument('--local_epoch', '--e', type=int, default=1) # local epoch -> 1
    parser.add_argument('--dropout', type=float, default=0)
    parser.add_argument('--num_ns', '--nn', type=int, default=4)   
    parser.add_argument('--standard', type=str, default="NDCG", choices=["NDCG", "Recall"])
    parser.add_argument('--lr', type=float, default=0.5) # fedbpr sgd -> lr 0.1... 
    parser.add_argument('--lr_eta', type=float, default=170)
    parser.add_argument('--weight_decay', type=float, default=0)
    # ----------------------------------------Fed parameter-----------------------------------------------------------> 
    parser.add_argument('--dp', type=float, default = 0.0)
    parser.add_argument('--two_step_kd', type=str, default='1_step', choices=['1_step','2_step'])
    parser.add_argument('--kd_user', type=str, default='both', choices=['both', 'user'], help='both: upadte item and user    user: update only user')
    parser.add_argument('--fed_lr', type=str, default='fed', choices=['fed', 'cf'], help="if fed, lr = self.args.lr * self.num_item , if cf lr = self.args.lr")

    # ----------------------------------------model specific parameter-----------------------------------------------------------> 
    parser.add_argument('--nsize', type=float, default=0, help='neighborhood size for GPFedRec') 
    parser.add_argument('--nthreshold', type=float, default=0.5, help='neighborhood threshold for GPFedRec') # 0 ~ 2
    parser.add_argument('--construct_graph_source', type=str, default='item', help='for GPFedRec', choices=['item', 'user'])
    parser.add_argument('--sim_metric', type=str, default='cosine', help='for GPFedRec', choices=['euclidean', 'cosine'])
    parser.add_argument('--lamda', type=float, default=0.5, help='GPFedRec (0.5), ReLoop2')
    
    parser.add_argument('--num_spp', type=int, default=3, help="number of samples from sp proxy", choices = [1,3,5])
    
    parser.add_argument('--K', type=int, default=60, help='for ReLoop2')
    parser.add_argument('--L', type=int, default=10, help='for ReLoop2')
    parser.add_argument('--error', type=float, default=0.1, help='for ReLoop2') # 두 값의 차이
    parser.add_argument('--reg_mlp', action='store_true', default=True) 
    # -------------------------------------------KD related parameter-------------------------------------------------------->     
    parser.add_argument('--client_cl', action='store_true', default=True)
    parser.add_argument('--server_cl', action='store_true', default=True)
    parser.add_argument('--beta',type=float, default=0.2)
    parser.add_argument('--diff',type=str, default='norm_l2', choices = ['l2', 'sqrt','norm_l2'])
    parser.add_argument('--eps', type=float, default=1e-3, help = 'epsilon for ranking discrepancy rate')
    parser.add_argument('--topN', type=int, default=50)
    parser.add_argument('--reg_client_cl', type=float, default=1e-3)
    parser.add_argument('--prob_func', type=str, default='exp', choices=['exp', 'tanh'])

    # ------------------------비교 실험: emb_d 및 topN KD 관련된 파라미터-----------------------------------
    parser.add_argument('--topn_kd', action='store_true', default=False, help = 'emb: emb regularizer')
    parser.add_argument('--reg_d', type=float, default=0, help = 'coefficient of emb or topn_kd regularizer') # 얘로 위에 둘 다 사용
    # -----------------------------------------------------------
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--specific_gpu', type=int, default=1, help='0: for scheduler, 1: assign specific gpu')
    parser.add_argument('--tqdm', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--backbone', type=str, default='fedmf', choices=['fedmf', 'fedmlp', 'pfedrec'])
    parser.add_argument('--model', type=str, default='ft', choices=['ft', 'fcrec', 'emb_reg', 'reloop2', 
                                                                    'naive_replay', 'topn_kd', 'full_batch',
                                                                    'spp','ewc'])
    parser.add_argument('--reg_reduction', type=str, default='sum', choices = ['sum', 'mean'])
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--job_id', type=str, default='000')


    args = parser.parse_args()
    
    # if args.model == 'fcrec':
    #     args.load_model =1
    
    args.baseline = args.backbone + '_' + args.model
        
    if args.save_model == 1 and args.load_model == 1:
        print(f"base 모델 저장과 불러오는거 한번에? -> 오류 종료")
        exit()

    set_seed(args.seed)
    
    start = time.time()
    print(f"seed: {args.seed}")
    print(args)
    if args.model == 'fcrec':
        fcrec(args)
    else:
        baseline(args) 
    
    end = time.time()
    
    print(f"걸린 시간: {(end - start)/60:.2f} min")