"""

by Hao Xue @ 12/03/19

"""

import numpy as np
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torch.nn as nn


class LVAttNet(nn.Module):
    def __init__(self, embedding_dim=128, hidden_dim=128, output_dim=2, obs_len=9, pred_len=8, drop_out=0.5, gpu=False):
        super(LVAttNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.obs_len = obs_len
        self.output_dim = output_dim
        self.gpu = gpu
        self.pred_len = pred_len

        self.loc_embeddings = nn.Linear(output_dim, embedding_dim)
        self.vel_embeddings = nn.Linear(output_dim, embedding_dim)
        self.gru_loc = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=drop_out)
        self.gru_vel = nn.GRU(embedding_dim, hidden_dim, batch_first=True, dropout=drop_out)
        self.loc2out = nn.Linear(hidden_dim, output_dim)
        self.vel2out = nn.Linear(hidden_dim, output_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(drop_out)
        self.softmax = nn.Softmax()
        self.attn = nn.Linear(2*output_dim, 2)

    def init_hidden(self, batch_size=1):
        h = Variable(torch.zeros(1, batch_size, self.hidden_dim))
        if self.gpu:
            return h.cuda()
        else:
            return h

    def tweak(self, last_time_input, last_time_pred, atten_weights):
        alpha_loc = torch.index_select(atten_weights, dim=2, index=self.generate_index([0], use_gpu=self.gpu))
        alpha_vel = torch.index_select(atten_weights, dim=2, index=self.generate_index([1], use_gpu=self.gpu))

        x_0 = torch.index_select(last_time_input, dim=2, index=self.generate_index([0], use_gpu=self.gpu))
        y_0 = torch.index_select(last_time_input, dim=2, index=self.generate_index([1], use_gpu=self.gpu))

        x_1 = torch.index_select(last_time_pred, dim=2, index=self.generate_index([0], use_gpu=self.gpu))
        y_1 = torch.index_select(last_time_pred, dim=2, index=self.generate_index([1], use_gpu=self.gpu))
        u_1 = torch.index_select(last_time_pred, dim=2, index=self.generate_index([2], use_gpu=self.gpu))
        v_1 = torch.index_select(last_time_pred, dim=2, index=self.generate_index([3], use_gpu=self.gpu))

        new_x = torch.bmm(alpha_loc, x_1) + torch.bmm(alpha_vel, x_0 + u_1)
        new_y = torch.bmm(alpha_loc, y_1) + torch.bmm(alpha_vel, y_0 + v_1)
        new_u = new_x - x_0
        new_v = new_y - y_0

        return torch.cat([new_x, new_y, new_u, new_v], dim=2), torch.cat([new_x, new_y], dim=2)

    def atten(self, last_time_pred):
        attn_energies = self.attn(last_time_pred)
        out = F.softmax(attn_energies, dim=2)
        return out

    def forward_one_time(self, obs, hidden_loc, hidden_vel):
        obs_loc = torch.index_select(obs, dim=2, index=self.generate_index([0, 1], use_gpu=self.gpu))
        obs_vel = torch.index_select(obs, dim=2, index=self.generate_index([2, 3], use_gpu=self.gpu))
        for i, input_t_loc in enumerate(obs_loc.chunk(obs_loc.size(1), dim=1)):
            emb_loc = self.relu(self.loc_embeddings(input_t_loc))
            self.gru_loc.flatten_parameters()
            out_loc, hidden_loc = self.gru_loc(emb_loc, hidden_loc)

        for i, input_t_vel in enumerate(obs_vel.chunk(obs_vel.size(1), dim=1)):
            emb_vel = self.relu(self.vel_embeddings(input_t_vel))
            self.gru_vel.flatten_parameters()
            out_vel, hidden_vel = self.gru_vel(emb_vel, hidden_vel)

        last_time_pred = torch.cat([self.loc2out(out_loc), self.vel2out(out_vel)], dim=-1)
        atten_weights = self.atten(last_time_pred)
        last_time_input = torch.cat([input_t_loc, input_t_vel], dim=-1)
        return last_time_input, last_time_pred, hidden_loc, hidden_vel, atten_weights

    def forward(self, obs, hidden):
        hidden_loc_init = hidden
        hidden_vel_init = hidden
        pred = []
        last_time_input, last_time_pred, hidden_loc, hidden_vel, atten_weights = \
            self.forward_one_time(obs, hidden_loc_init, hidden_vel_init)
        a, out_loc = self.tweak(last_time_input, last_time_pred, atten_weights)

        for _ in range(self.pred_len):
            last_time_input, last_time_pred, hidden_loc, hidden_vel, atten_weights = \
                self.forward_one_time(a, hidden_loc, hidden_vel)
            a, out_loc = self.tweak(last_time_input, last_time_pred, atten_weights)
            pred.append(torch.squeeze(out_loc, dim=1))

        return torch.stack(pred, dim=1)

    @staticmethod
    def generate_index(index, use_gpu=True):
        if use_gpu:
            return Variable(torch.LongTensor(index)).cuda()
        else:
            return Variable(torch.LongTensor(index))

