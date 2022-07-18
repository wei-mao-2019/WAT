import torch
import numpy as np
from torch import nn
from torch.nn import functional as F
from models.mlp import MLP
from models.rnn import RNN
from utils.torch import *
from models import transformer_vae

class ActVAE(nn.Module):
    """
    decode with action label
    v3_3 + batch norm
    act in posterior
    """
    def __init__(self, nx, ny, nz, horizon, specs):
        super(ActVAE, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', True)
        # self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)
        self.is_layernorm = is_layernorm = specs.get('is_layernorm', False)
        self.is_bn = is_bn = specs.get('is_bn', True)

        # encode
        self.x_rnn = RNN(nx, nh_rnn, bi_dir=x_birnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        self.e_rnn = RNN(ny, nh_rnn, bi_dir=False, cell_type=rnn_type,is_layernorm=is_layernorm)
        # self.e_rnn.set_mode('step')
        self.e_mlp = MLP(3 * nh_rnn, nh_mlp, is_bn=is_bn)
        self.e_mu = nn.Linear(self.e_mlp.out_dim, nz)
        self.e_logvar = nn.Linear(self.e_mlp.out_dim, nz)

        # prior
        self.p_act_mlp = nn.Linear(n_action, nh_rnn)
        self.p_mlp = MLP(2 * nh_rnn, nh_mlp, is_bn=is_bn)
        self.p_mu = nn.Linear(self.p_mlp.out_dim, nz)
        self.p_logvar = nn.Linear(self.p_mlp.out_dim, nz)

        # decode
        # self.d_act_mlp = nn.Linear(n_action, nh_rnn)
        if self.use_drnn_mlp:
            self.drnn_mlp = MLP(nh_rnn, nh_mlp + [nh_rnn], activation='tanh')
        self.d_rnn = RNN(ny + nz + nh_rnn + nh_rnn, nh_rnn, cell_type=rnn_type,is_layernorm=is_layernorm)
        self.d_mlp = MLP(nh_rnn, nh_mlp)
        self.d_out = nn.Linear(self.d_mlp.out_dim, ny)
        self.d_rnn.set_mode('step')
        #
        # self.stop_sign_mlp = MLP(nh_rnn, nh_mlp)
        # self.stop_sign_out = nn.Sequential(nn.Linear(self.d_mlp.out_dim, ny), nn.Sigmoid())

    def encode_x(self, x):
        if self.x_birnn:
            h_x = self.x_rnn(x).mean(dim=0)
        else:
            h_x = self.x_rnn(x)[-1]
        return h_x

    def encode_y(self, y, fn):
        # if self.e_birnn:
        #     h_y = self.e_rnn(y).mean(dim=0)
        # else:
        h_y = self.e_rnn(y)
        h_y = h_y.transpose(0, 1)
        h_y = h_y[fn == 1]

        return h_y

    def encode(self, x, y, act, fn):
        h_x = self.encode_x(x)
        h_y = self.encode_y(y, fn)
        h_act = self.p_act_mlp(act)

        h = torch.cat((h_x, h_y, h_act), dim=1)
        h = self.e_mlp(h)
        emu = self.e_mu(h)
        elogvar = self.e_logvar(h)

        h = torch.cat((h_x, h_act), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)
        return emu, elogvar, pmu, plogvar, h_x, h_act

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, x, z, h_act, h_x=None):
        if h_x is None:
            h_x = self.encode_x(x)
        if self.use_drnn_mlp:
            h_d = self.drnn_mlp(h_x)
            self.d_rnn.initialize(batch_size=z.shape[0], hx=h_d)
        else:
            self.d_rnn.initialize(batch_size=z.shape[0])
        y = []
        for i in range(self.horizon):
            y_p = x[-1] if i == 0 else y_i
            rnn_in = torch.cat([h_x, h_act, z, y_p], dim=1)
            h = self.d_rnn(rnn_in)
            h = self.d_mlp(h)
            y_i = self.d_out(h) + x[-1]
            y.append(y_i)
        y = torch.stack(y)
        return y

    def forward(self, x, y, act, fn):
        mu, logvar, pmu, plogvar, h_x, h_act = self.encode(x, y, act, fn)
        z = self.reparameterize(mu, logvar) if self.training else mu
        return self.decode(x, z, h_act, h_x), mu, logvar, pmu, plogvar

    def sample_prior(self, x, act):
        h_x = self.encode_x(x)
        h_act = self.p_act_mlp(act)
        h = torch.cat((h_x, h_act), dim=1)
        h = self.p_mlp(h)
        pmu = self.p_mu(h)
        plogvar = self.p_logvar(h)
        z = self.reparameterize(pmu, plogvar)
        return self.decode(x, z, h_act, h_x)

class ActClassifier(nn.Module):
    def __init__(self, nx, ny, nz, horizon, specs):
        super(ActClassifier, self).__init__()
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.horizon = horizon
        self.rnn_type = rnn_type = specs.get('rnn_type', 'lstm')
        self.x_birnn = x_birnn = specs.get('x_birnn', False)
        self.e_birnn = e_birnn = specs.get('e_birnn', True)
        self.use_drnn_mlp = specs.get('use_drnn_mlp', False)
        self.nh_rnn = nh_rnn = specs.get('nh_rnn', 128)
        self.nh_mlp = nh_mlp = specs.get('nh_mlp', [300, 200])
        self.n_action = n_action = specs.get('n_action', 15)

        # encode
        self.x_rnn = torch.nn.GRU(nx,nh_rnn,1)
        self.c_mlp = MLP(nh_rnn, nh_mlp, activation='relu', is_bn=specs.get('is_bn',False),
                         is_dropout=specs.get('is_dropout',False))
        self.c_out = nn.Linear(self.c_mlp.out_dim, self.n_action)

    def forward(self, x, fn,is_feat=False):
        h_x = self.x_rnn(x)[0] #[seq, bs, feat]
        h_x = h_x.transpose(0,1)
        h_x = h_x[fn==1] # [bs, feat]

        h_x = self.c_mlp(h_x)
        h = self.c_out(h_x)
        c = torch.softmax(h,dim=1)
        if is_feat:
            return c,h,h_x
        else:
            return c, h

def get_action_vae_model(cfg, traj_dim, model_version=None, max_len=None):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', None)

    # if model_name == 'v3_5_4':
    return ActVAE(traj_dim, traj_dim, cfg.nz, max_len, specs)


def get_action_classifier(cfg, traj_dim, model_version=None, max_len=None):
    specs = cfg.vae_specs
    model_name = specs.get('model_name', None)
    return ActClassifier(traj_dim, traj_dim, cfg.nz, max_len, specs)
