import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.distributions.laplace import Laplace

from utils.data_loader import *
from utils.evaluation_all import *
from utils.util import *
from utils.early_stop import *

from tqdm import tqdm

import copy

    
class ServerMF(nn.Module): # Server model for convenient evaluation. This class is not used at real-world scenario.
    def __init__(self, args, user_emb_weight, item_emb_weight):
        super(ServerMF, self).__init__()
        
        self.args = args
        self.dim = args.dim
        self.item_emb = nn.Embedding.from_pretrained(item_emb_weight.clone().detach())
        self.user_emb = nn.Embedding.from_pretrained(user_emb_weight.clone().detach())
        
    def get_score_mat(self):
        user = self.user_emb.weight.clone().detach()
        item = self.item_emb.weight.clone().detach()
        
        return user @ item.T


class ClientMF(nn.Module):
    def __init__(self, args):
        super(ClientMF, self).__init__()
        
        self.args = args
        self.user_emb = nn.Embedding(1, args.dim)
        self.item_emb = nn.Embedding(args.num_item, args.dim)

        self.user_emb.weight.requires_grad = True
        self.item_emb.weight.requires_grad = True
        self.sigmoid = nn.Sigmoid()
        
    def get_score_mat(self):
        user = self.user_emb.weight
        item = self.item_emb.weight
        
        return user @ item.T
    
    def forward(self, item_idx):
        user = self.user_emb.weight
        item = self.item_emb(item_idx)
        
        return user @ item.T
    
    def get_score_mat_only_user_grad_true(self):
        user = self.user_emb.weight
        item = self.item_emb.weight.clone().detach()

        return user @ item.T

 
    
class FedMF_Engine(nn.Module):
    def __init__(self, args, device):
        super(FedMF_Engine, self).__init__()
        
        self.args = args
        self.device = device
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim = args.dim
        
        self.lr = args.lr
        
        # global item embedding
        self.item_emb = nn.Embedding(self.num_item, self.dim)
        self.user_emb = nn.Embedding(self.num_user, self.dim)
        
        nn.init.normal_(self.user_emb.weight, std = 0.01)
        nn.init.normal_(self.item_emb.weight, std = 0.01)
        
        # emb in time t-1 block
        self.old_user_emb = nn.Embedding.from_pretrained(self.user_emb.weight.clone().detach())
        self.old_item_emb = nn.Embedding.from_pretrained(self.item_emb.weight.clone().detach())
        
        # list to check whether users are trained in the previous block
        self.topn_list = None
        
        # just for evaluation
        self.model = ServerMF(args, self.user_emb.weight.clone().detach(), self.item_emb.weight.clone().detach())
        
        # Metric Matrix
        self.A_N20 = torch.zeros((args.num_task, args.num_task))
        self.A_R20 = torch.zeros((args.num_task, args.num_task))
        
        self.train_mat_dict = {}
        self.train_mat_tensor = {}
        self.valid_mat_tensor = {}
        self.test_mat_tensor = {}
        
        self.loss_fn = nn.BCEWithLogitsLoss()
        self.reg_fn = nn.MSELoss(reduction=self.args.reg_reduction)
        
    def add_user_item(self, new_num_user, new_num_item):
        if self.num_user != new_num_user:
            new_user_emb = nn.Embedding(new_num_user - self.num_user, self.dim)
            nn.init.normal_(new_user_emb.weight, std=0.01)
            new_weights = torch.cat((self.user_emb.weight, new_user_emb.weight), dim=0).clone().detach()
            self.user_emb = nn.Embedding.from_pretrained(new_weights)
            self.user_emb.weight.requires_grad = True
            self.num_user = new_num_user
        
        if self.num_item != new_num_item:
            new_item_emb = nn.Embedding(new_num_item - self.num_item, self.dim)
            nn.init.normal_(new_item_emb.weight, std=0.01)
            new_weights = torch.cat((self.item_emb.weight, new_item_emb.weight), dim=0).clone().detach()
            self.item_emb = nn.Embedding.from_pretrained(new_weights)
            self.item_emb.weight.requires_grad = True
            self.num_item = new_num_item
            
        # re-define model
        self.model = ServerMF(self.args, self.user_emb.weight.clone().detach(), self.item_emb.weight.clone().detach())
        
    
    def validate(self, train_mat_tensor, test_mat_tensor):
        return evaluate(self.model, train_mat_tensor, test_mat_tensor, self.device)
    
        
    def test(self, task):
        # 1. a_{i,i}
        masking = self.train_mat_tensor[f"TASK_{task}"] + self.valid_mat_tensor[f"TASK_{task}"]
        RESULT = self.validate(masking, self.test_mat_tensor[f"TASK_{task}"])
        N_20, R_20 = RESULT["N@20"], RESULT["R@20"]
        self.A_N20[task][task] = N_20
        self.A_R20[task][task] = R_20
    
        
        # 2. a_{task, i}
        for i in range(task): # i: 0 ~ task-1
            masking = self.train_mat_tensor[f"TASK_{task}"] + self.valid_mat_tensor[f"TASK_{task}"]
            RESULT = self.validate(masking, self.test_mat_tensor[f"TASK_{i}"])
            N_20, R_20 = RESULT["N@20"], RESULT["R@20"]
            self.A_N20[task][i] = N_20
            self.A_R20[task][i] = R_20
        
        return get_LA_RA(task, self.A_N20, self.A_R20)
        
    
    
    def task_specific_data_processing(self, task, input_total_data):
        
        total_train_dataset, total_valid_dataset, total_test_dataset = input_total_data
        
        for i in range(task + 1):
            self.train_mat_dict[f"TASK_{i}"], self.train_mat_tensor[f"TASK_{i}"] = make_rating_mat(total_train_dataset[f"TASK_{i}"], True, self.num_user, self.num_item)
            self.valid_mat_tensor[f"TASK_{i}"] = make_rating_mat(total_valid_dataset[f"TASK_{i}"], False, self.num_user, self.num_item)
            self.test_mat_tensor[f"TASK_{i}"] = make_rating_mat(total_test_dataset[f"TASK_{i}"], False, self.num_user, self.num_item)
    
    
    def run(self, task, input_total_data, is_base):
        
        self.task_specific_data_processing(task, input_total_data)
        
        block_info = f"Inc #{task} Block!" if task != 0 else "Base Block!"
        
        print('\n ==========' + block_info + '==========')
        
        best_valid = -torch.inf
        best_param = self.model.state_dict()
        best_epoch = -1
        
        early = EarlyStopping(patience = self.args.patience, larger_is_better=True)
    
        
        for round in range(self.args.num_round):
            self.fed_train_a_round(task, is_base, round)
            
            RESULT = self.validate(self.train_mat_tensor[f"TASK_{task}"], self.valid_mat_tensor[f"TASK_{task}"])
            N_20, R_20 = RESULT['N@20'], RESULT['R@20']
            epoch_val = N_20
            
            if self.args.show_valid:
                print(f"Epoch: {round + 1} N@20: {N_20:.5f} R@20: {R_20:.5f}")
                
            
            if epoch_val > best_valid:
                best_valid = epoch_val
                best_param = self.model.state_dict()  
                best_epoch = round + 1

            early(epoch_val)
            
            if early.stop_flag:
                break
                
        # test phase
        self.model.load_state_dict(best_param)
        print(f"Best Epoch: {best_epoch} !!")

        RESULT = self.test(task)
        print(f"TEST N@20: {self.A_N20[task][task]:.5f} R@20: {self.A_R20[task][task]:.5f}")
        
        
        
        # save part
        if self.args.save_result ==1 and (not is_base):
            save_result_as_csv(self.args, self.args.baseline, RESULT, task)
        
        # ===================================================================================================        
        self.model.to(self.device)
        self.model.train()
        self.target_mat = self.model.get_score_mat().clone().detach().to('cpu')
        self.model.to('cpu')
        # ===================================================================================================
        
        # for CL (save topn list & old_emb)
        if self.args.client_cl:
            if is_base:
                __, self.topn_list = torch.topk(self.target_mat, self.args.topN, dim=1)
            
            else: 
                diff_num_user = self.num_user - len(self.topn_list)
                tmp_list = torch.zeros((diff_num_user, self.args.topN), dtype=torch.long)
                self.topn_list = torch.concat((self.topn_list, tmp_list), dim=0)
                
                trained_user_in_this_block = torch.tensor(list(self.train_mat_dict[f"TASK_{task}"].keys()))
                ___, block_topn_list = torch.topk(self.target_mat[trained_user_in_this_block], self.args.topN, dim=1)
                self.topn_list[trained_user_in_this_block] = block_topn_list
        
        self.old_user_emb = nn.Embedding.from_pretrained(self.model.user_emb.weight.clone().detach())
        self.old_item_emb = nn.Embedding.from_pretrained(self.model.item_emb.weight.clone().detach())
        
    def fed_train_a_round(self, task, is_base, round_id):

        train_mat_dict = self.train_mat_dict[f"TASK_{task}"]
        
        num_participants = int( len(list(train_mat_dict.keys())) * self.args.clients_sample_ratio)
        participants = random.sample(list(train_mat_dict.keys()), num_participants)
        round_user_params = {}
        
        # train
        for user in participants:
            # self.trained_this_block.append(user)
            
            client_model = ClientMF(self.args)
            user_param_dict = copy.deepcopy(self.model.state_dict())
            user_param_dict['user_emb.weight'] = self.model.user_emb.weight[user].reshape(1, -1)
            client_model.load_state_dict(user_param_dict)
            
            client_model.to(self.device)
            
            if self.args.optimizer == 'SGD':
                optimizer = torch.optim.SGD([{'params': client_model.user_emb.weight, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay},
                                            {'params': client_model.item_emb.weight, 'lr': self.args.lr * self.num_item, 'weight_decay': self.args.weight_decay}])
            elif self.args.optimizer == 'Adam':
                optimizer = torch.optim.Adam([{'params': client_model.user_emb.weight, 'lr': self.args.lr, 'weight_decay': self.args.weight_decay},
                                            {'params': client_model.item_emb.weight, 'lr': self.args.lr * self.num_item, 'weight_decay': self.args.weight_decay}])
                
            user_dataloader = DataLoader(dataset = Pos_Neg_Sampler(self.args, train_mat_dict[user]), batch_size=self.args.batch_size, shuffle = True)
            client_model.train()
            
            for epoch in range(self.args.local_epoch):
                for batch in user_dataloader:
                    client_model= self.fed_train_single_user_unified(client_model, batch, optimizer, user, is_base, round_id)
            
            self.user_emb.weight[user].data.copy_(client_model.user_emb.weight.clone().detach().to('cpu').reshape(-1))
            
            # LDP & Aggregate
            round_user_params[user] = client_model.item_emb.weight.clone().detach().to('cpu')
            if self.args.dp > 0:
                round_user_params[user] += Laplace(0, self.args.dp).expand(round_user_params[user].shape).sample()
            
               
        self.aggregate_clients_params(round_user_params, is_base, round_id)
        self.model.item_emb = nn.Embedding.from_pretrained(self.item_emb.weight.clone().detach())
        self.model.user_emb = nn.Embedding.from_pretrained(self.user_emb.weight.clone().detach())
        

    def fed_train_single_user_unified(self, client_model, batch_data, optimizer, user, is_base, round):
        optimizer.zero_grad()
        
        items, target = batch_data[0].to(self.device), batch_data[1].float().to(self.device)
        prediction = client_model(items).reshape(-1)
        loss = self.loss_fn(prediction, target)
        
        old_user_size = self.old_user_emb.weight.data.shape[0]
        old_item_size = self.old_item_emb.weight.data.shape[0]

        # ================ Client-side Continual ================
        if (not is_base) and (self.args.client_cl) and (user < old_user_size):
            target_mat = self.target_mat[user].to(self.device)
            target_mat.requires_grad = False
            
            pred_mat = client_model.get_score_mat().reshape(-1)
            last_topn_list = self.topn_list[user] 
            
            # Adaptive Replay Memory
            kd_loss = get_rank_discrepancy_kd_loss(self.args, target_mat, pred_mat, last_topn_list, self.device)
            
            loss += kd_loss * self.args.reg_client_cl

        loss.backward()
        optimizer.step()

        return client_model
        
        
    def aggregate_clients_params(self, round_user_params, is_base, round):
        # aggregate item embedding
        
        for idx, user in enumerate(round_user_params.keys()):
            
            user_params = round_user_params[user] # item_emb.weight
            
            if idx == 0:
                server_model_param = user_params
            else:
                server_model_param += user_params
        
        server_model_param = server_model_param / len(round_user_params)
        
        if self.args.server_cl and (not is_base):
            old = self.old_item_emb.weight.data
            num_old_item = len(old)
            new_item_emb = server_model_param[num_old_item:].clone()
            
            weight = self.args.beta * diff(old, server_model_param[:num_old_item]) 
            server_model_param = (1-weight) * server_model_param[:num_old_item] + weight * old
            server_model_param = torch.concat((server_model_param, new_item_emb), dim=0) 
        
        self.item_emb = nn.Embedding.from_pretrained(server_model_param)