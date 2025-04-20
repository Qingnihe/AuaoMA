import torch
import numpy as np
from torch import nn
from torch import optim
from interfusion.planarNF import PlanarNormalizingFlow
from interfusion.RecurrentDistribution import RecurrentDistribution
from interfusion.data import SlidingWindowDataset, SlidingWindowDataLoader
from torch.distributions import Normal
from interfusion.vae import VAE
from data_config import *

def freeze_a_layer(layer: nn.Module):
    for param in layer.parameters():
        param.requires_grad = False
        
class InterFusion:
    def __init__(self, x_dims: int, z1_dims: int = 3, z2_dims: int = 3, max_epochs: int = 20, 
                 pre_max_epochs: int = 20, batch_size: int = 100, window_size: int = 60, n_samples: int = 10,
                 learning_rate: float = 1e-3, output_padding_list: list = [0,0,0], model_dim:int = 100):
        self.x_dims = x_dims
        self.max_epochs = max_epochs
        self.pre_max_epochs = pre_max_epochs
        self.batch_size = batch_size
        self.valid_epoch_freq = global_valid_epoch_freq
        self.z1_dims = z1_dims
        self.z2_dims = z2_dims
        self.window_size = window_size
        self.step = 0
        self.device = global_device
        self.n_samples = n_samples
        self.model_dim = model_dim
        self.vae = VAE(x_dims, z1_dims, z2_dims, window_size, n_samples, model_dim, model_dim, output_padding_list).to(device=self.device)
        
        self.optimizer = optim.Adam(params=self.vae.parameters(), lr=learning_rate)
        self.schedule = optim.lr_scheduler.CosineAnnealingLR(optimizer=self.optimizer, T_max=10)
        self.pretrain_loss =  {'train': [], 'valid': []}
        self.train_loss =  {'train': [], 'valid': [], 'valid_recon': [], 'valid_kl': []}
        self.best_pretrain_valid_loss = 99999999
        self.best_train_valid_loss = 99999999
        self.best_train_valid_recon = 99999999

        self.pretrain_ELBO_loss = global_pretrain_ELBO_loss
        self.train_ELBO_loss = global_train_ELBO_loss
    
    def freeze_layers(self, name):
        if name == 'rnn':
            freeze_a_layer(self.vae.q_net.a_rnn.rnn)
        elif name == 'cnn':
            freeze_a_layer(self.vae.h_for_qz)
            freeze_a_layer(self.vae.h_for_px)
        elif name == 'rnn_cnn':
            freeze_a_layer(self.vae.q_net.a_rnn.rnn)
            freeze_a_layer(self.vae.h_for_qz)
            freeze_a_layer(self.vae.h_for_px)
    
    def valid_log_pretrain(self, valid_data, log_file, save_path, epoch):
        if_break = False
        for w in valid_data:
            w = w.to(self.device)
            q_zx, p_xz, z, x_mean, x_std = None,None,None,None,None
            z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = None,None,None,None,None,None,None,None,None,None,None
            z2_sampled, p_z, q_zx, p_xz, x_mean, x_std = self.vae(w, if_pretrain=True)
            valid_loss = self.pretrain_ELBO_loss(self, w, z2_sampled, q_zx, p_z, p_xz)
            self.pretrain_loss['valid'][-1] += valid_loss.item()
        if self.pretrain_loss['valid'][-1] <= self.best_pretrain_valid_loss and not if_break:
            self.best_pretrain_valid_loss = self.pretrain_loss['valid'][-1]
            self.save(save_path)
            print("***", end=' ')
            print("***", end=' ', file=log_file)    
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.pretrain_loss['valid'][-1]} train_loss:{self.pretrain_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}")
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.pretrain_loss['valid'][-1]} train_loss:{self.pretrain_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}", file=log_file)
        
        return if_break

    def valid_log_train(self, valid_data, log_file, save_path, epoch):
        if_break = False
        for w in valid_data:
            w = w.to(self.device)
            q_zx, p_xz, z, x_mean, x_std = None,None,None,None,None
            z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = None,None,None,None,None,None,None,None,None,None,None
            z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = self.vae(w)
            valid_loss, valid_recon, valid_kl = self.train_ELBO_loss(self, x=w, z1_sampled=z1_sampled, z1_sampled_flowed=z1_sampled_flowed, z1_flow_log_det=z1_flow_log_det, z2_sampled=z2_sampled, 
                p_xz_dist=p_xz_dist, q_z1_dist=q_z1_dist, q_z2_dist=q_z2_dist, pz1_dist=pz1_dist, pz2_dist=pz2_dist)
            self.train_loss['valid'][-1] += valid_loss.item()
            self.train_loss['valid_recon'][-1] += valid_recon.item()
            self.train_loss['valid_kl'][-1] += valid_kl.item()
        if self.train_loss['valid_recon'][-1] <= self.best_train_valid_recon and not if_break:
            self.best_train_valid_loss = self.train_loss['valid'][-1]
            self.best_train_valid_recon = self.train_loss['valid_recon'][-1]

            self.save(save_path)   
            print("***", end=' ')
            print("***", end=' ', file=log_file)                                           
        
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.train_loss['valid'][-1]} recon:{self.train_loss['valid_recon'][-1]} kl:{self.train_loss['valid_kl'][-1]} train_loss:{self.train_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}")
        print(f"epoch:{epoch}/{self.max_epochs} step:{self.step} valid_loss:{self.train_loss['valid'][-1]} recon:{self.train_loss['valid_recon'][-1]} kl:{self.train_loss['valid_kl'][-1]} train_loss:{self.train_loss['train'][-1]} lr:{list(self.optimizer.param_groups)[0]['lr']}", file=log_file)
        
        return False
    
    def get_data(self, values, valid_portion):
        if_drop_last = True
        train_values = values
        train_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(train_values, self.window_size),
            batch_size=self.batch_size,
            shuffle=True,
            drop_last=if_drop_last
        )
        all_data = list(train_sliding_window)
        if len(all_data) > 1:
            valid_index_list = [i for i in range(0, len(all_data), int(1/valid_portion))]
            train_index_list = [i for i in range(0, len(all_data)) if i not in valid_index_list]
            train_data = np.array(all_data)[train_index_list]
            valid_data = np.array(all_data)[valid_index_list]
        else: 
            train_data = all_data
            valid_data = all_data

        return train_data, valid_data
    

    def prefit(self, values, save_path: pathlib.Path, valid_portion=0.3):
        train_data, valid_data = self.get_data(values=values, valid_portion=valid_portion)
        log_path = save_path.parent / 'pretrain_log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')
        for epoch in range(1, self.pre_max_epochs+1):
            if_break = False
            for i, w in enumerate(train_data):
                w = w.to(self.device)
                self.pretrain_loss['train'].append(0)
                self.pretrain_loss['valid'].append(0)
                self.optimizer.zero_grad()
                z2_sampled, p_z, q_zx, p_xz = None,None,None,None
                z2_sampled, p_z, q_zx, p_xz, x_mean, x_std = self.vae(w, if_pretrain=True)
                loss = self.pretrain_ELBO_loss(self, w, z2_sampled, q_zx, p_z, p_xz)
                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                self.pretrain_loss['train'][-1] = loss.item()
                self.step += 1

            if epoch % self.valid_epoch_freq == 0 and valid_portion > 0:
                if_break = self.valid_log_pretrain(valid_data, log_file, save_path, epoch)
            if epoch % learning_rate_decay_by_epoch == 0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= learning_rate_decay_factor
            self.schedule.step()
            if if_break:
                break

    def fit(self, values, x_test_list, save_path: pathlib.Path, valid_portion=0.3):
        train_data, valid_data = self.get_data(values=values, valid_portion=valid_portion)
        
        log_path = save_path.parent / 'train_log.txt'
        if log_path.exists():
            os.remove(log_path)
        log_file = open(log_path, mode='a')

        self.train_loss['train'].append(0)  
        self.train_loss['valid'].append(0)
        self.train_loss['valid_recon'].append(0)
        self.train_loss['valid_kl'].append(0)
        
        self.valid_log_train(valid_data, log_file, save_path, 0)
        for epoch in range(1, self.max_epochs+1):
            if_break = False
            for i, w in enumerate(train_data):
                w = w.to(self.device)

                self.train_loss['train'].append(0)
                self.train_loss['valid'].append(0)
                self.train_loss['valid_recon'].append(0)
                self.train_loss['valid_kl'].append(0)
                self.optimizer.zero_grad()
                z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = None,None,None,None,None,None,None,None,None,None,None
                z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = self.vae(w)

                loss, _, _ = self.train_ELBO_loss(self, x=w, z1_sampled=z1_sampled, z1_sampled_flowed=z1_sampled_flowed, z1_flow_log_det=z1_flow_log_det, z2_sampled=z2_sampled, 
                    p_xz_dist=p_xz_dist, q_z1_dist=q_z1_dist, q_z2_dist=q_z2_dist, pz1_dist=pz1_dist, pz2_dist=pz2_dist)

                self.optimizer.zero_grad()
                loss.backward(retain_graph=True)
                nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=10, norm_type=2)
                self.optimizer.step()
                self.train_loss['train'][-1] = loss.item()
                self.step += 1
            if epoch % self.valid_epoch_freq == 0 and valid_portion > 0:
                if_break = self.valid_log_train(valid_data, log_file, save_path, epoch)
                if if_break:
                    break
            if epoch % learning_rate_decay_by_epoch == 0:
                for p in self.optimizer.param_groups:
                    p['lr'] *= learning_rate_decay_factor
            if dataset_type != 'ctf':
                self.schedule.step()


    def predict(self, values, save_path: pathlib.Path, if_pretrain=False):
        self.vae.eval()
        score_list = []
        recon_mean_list = []
        recon_std_list = []
        z_list = []
        test_sliding_window = SlidingWindowDataLoader(
            SlidingWindowDataset(values, self.window_size),
            batch_size=self.batch_size,
        )
        for i, w in enumerate(test_sliding_window):
            w = w.to(self.device)
            if if_pretrain:
                z2_sampled, p_z, q_zx, p_xz_dist, x_mean, x_std = self.vae(w, if_pretrain=True)
                z_list.append(torch.mean(z2_sampled[:, :, :, -1], dim=0).cpu().detach().numpy())
            else:
                z1_sampled, z1_sampled_flowed, z1_flow_log_det, z2_sampled, p_xz_dist, q_z1_dist, q_z2_dist, pz1_dist, pz2_dist, x_mean, x_std = self.vae(w)
                z_list.append(torch.mean(z1_sampled_flowed[:, :, -1, :], dim=0).cpu().detach().numpy())

            recon_mean_list.append(torch.mean(x_mean[:, :, -1, :], dim=0).cpu().detach().numpy())
            recon_std_list.append(torch.mean(x_std[:, :, -1, :], dim=0).cpu().detach().numpy())
            score_list.append(torch.mean(p_xz_dist.log_prob(w)[:, :, -1, :], dim=0).cpu().detach().numpy())
        
        score = np.concatenate(score_list, axis=0)
        recon_mean = np.concatenate(recon_mean_list, axis=0)
        recon_std = np.concatenate(recon_std_list, axis=0)
        z = np.concatenate(z_list, axis=0)
        return score, recon_mean, recon_std, z

    def save(self, save_path: pathlib.Path):
        torch.save(self.vae.state_dict(), save_path)

    def restore(self, save_path: pathlib.Path):
        self.vae.load_state_dict(torch.load(save_path))

