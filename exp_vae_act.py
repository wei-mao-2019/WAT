import os
import sys
import math
import pickle
import argparse
import time

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_ntu_act_transition import DatasetNTU
from motion_pred.utils.dataset_grab_action_transition import DatasetGrab
from motion_pred.utils.dataset_humanact12_act_transition import DatasetACT12
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred import *
from utils.utils import get_dct_matrix

"""dct smoothness + last frame smoothness, with action transition"""

def loss_function(X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m):
    lambdas = cfg.vae_specs['lambdas']

    if cfg.dataset == 'babel':
        index_used = list(range(30)) + list(range(36,66))
        Y_r = Y_r[:,:,index_used]
        Y = Y[:,:,index_used]
        X = X[:,:,index_used]

    MSE = (Y_r - Y).pow(2).sum(dim=-1).transpose(0, 1)[fn_mask == 1].mean()

    # smoothness
    x = torch.cat([X[-args.N:],Y_r[:args.N]],dim=0).transpose(0,1)
    x_est = torch.matmul(idct_m[None,:,:args.dct_n],torch.matmul(dct_m[None,:args.dct_n],x))
    MSE_v1 = (x_est - x).norm(dim=-1).mean()
    MSE_v2 = (X[-1] - Y_r[0]).norm(dim=-1).mean()
    KLD = 0.5 * torch.sum(plogvar - logvar + (logvar.exp() + (mu - pmu).pow(2)) / plogvar.exp() - 1) / Y.shape[1]

    # regularization
    # div =
    if len(lambdas) == 3:
        loss_r = lambdas[0] * MSE + lambdas[1] * MSE_v1 + lambdas[1] * MSE_v2 + lambdas[2] * KLD
    else:
        loss_r = lambdas[0] * MSE + lambdas[1] * MSE_v1 + lambdas[2] * MSE_v2 + lambdas[3] * KLD
    # loss_r = MSE
    return loss_r, np.array([loss_r.item(), MSE.item(), MSE_v1.item(), MSE_v2.item() , KLD.item()])


def train(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    train_grad = 0
    loss_names = ['TOTAL', 'MSE', 'DCT_smooth', 'Lastframe_smooth', 'KLD']
    generator = dataset.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=args.is_other_act, t_pre_extra=args.t_pre_extra,
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 0.08,
                                           is_transi=args.is_transi)

    dct_m, idct_m = get_dct_matrix(args.N*2, is_torch=True,device=device)

    for traj_np, label, fn, fn_mask in generator:
        # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
        traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
        label = tensor(label, device=device, dtype=dtype)
        fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
        fn_mask = tensor(fn_mask[:, t_his:], device=device, dtype=dtype)
        X = traj[:t_his]
        Y = traj[t_his:]
        Y_r, mu, logvar, pmu, plogvar = model(X, Y, label, fn)
        loss, losses = loss_function(X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        grad_norm = torch.nn.utils.clip_grad_norm_(list(model.parameters()), max_norm=100)
        train_grad += grad_norm
        train_losses += losses
        total_num_sample += 1

    scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch: {} Time: {:.2f} {} lr: {:.5f}'.format(epoch, dt, losses_str, lr))
    tb_logger.add_scalar('train_grad', train_grad / total_num_sample, epoch)
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'train': loss}, epoch)


def test(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'MSE', 'DCT_smooth', 'Lastframe_smooth', 'KLD']
    generator = dataset_test.sampling_generator(num_samples=cfg.num_vae_data_sample, batch_size=cfg.batch_size,
                                           is_other_act=args.is_other_act, t_pre_extra=args.t_pre_extra,
                                           act_trans_k= cfg.vae_specs['act_trans_k'] if 'act_trans_k'
                                                                                        in cfg.vae_specs else 0.08,
                                           max_trans_fn= cfg.vae_specs['max_trans_fn'] if 'max_trans_fn'
                                                                                        in cfg.vae_specs else 0.08)

    dct_m, idct_m = get_dct_matrix(args.N*2, is_torch=True,device=device)
    with torch.no_grad():
        for traj_np, label, fn, fn_mask in generator:
            # traj_np = traj_np[..., 1:, :].reshape(traj_np.shape[0], traj_np.shape[1], -1)
            traj = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
            label = tensor(label, device=device, dtype=dtype)
            fn = tensor(fn[:, t_his:], device=device, dtype=dtype)
            fn_mask = tensor(fn_mask[:, t_his:], device=device, dtype=dtype)
            X = traj[:t_his]
            Y = traj[t_his:]
            Y_r, mu, logvar, pmu, plogvar = model(X, Y, label, fn)
            loss, losses = loss_function(X, Y_r, Y, mu, logvar, pmu, plogvar, fn_mask, dct_m, idct_m)
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
            train_losses += losses
            total_num_sample += 1

    # scheduler.step()
    dt = time.time() - t_s
    train_losses /= total_num_sample
    # lr = optimizer.param_groups[0]['lr']
    losses_str = ' '.join(['{}: {:.4f}'.format(x, y) for x, y in zip(loss_names, train_losses)])
    logger.info('====> Epoch Test: {} Time: {:.2f} {}'.format(epoch, dt, losses_str))
    for name, loss in zip(loss_names, train_losses):
        tb_logger.add_scalars('vae_' + name, {'test': loss}, epoch)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='grab_rnn')
    parser.add_argument('--mode', default='train')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--is_other_act', action='store_true', default=False)
    parser.add_argument('--is_transi', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=0)
    parser.add_argument('--N', type=int, default=10) # number of history and future frames for smoothness
    parser.add_argument('--dct_n', type=int, default=5)
    parser.add_argument('--t_pre_extra', type=int, default=0) # extra future poses for stopping
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if 'smooth_N' in cfg.vae_specs:
        args.N = cfg.vae_specs['smooth_N']
    if 'dct_n' in cfg.vae_specs:
        args.dct_n = cfg.vae_specs['dct_n']
    if 't_pre_extra' in cfg.vae_specs:
        args.t_pre_extra = cfg.vae_specs['t_pre_extra']
    if 'is_other_act' in cfg.vae_specs:
        args.is_other_act = cfg.vae_specs['is_other_act']
    logger.info(cfg)

    """data"""
    if cfg.dataset == 'grab':
        dataset_cls = DatasetGrab
    elif cfg.dataset == 'ntu':
        dataset_cls = DatasetNTU
    elif cfg.dataset == 'humanact12':
        dataset_cls = DatasetACT12
    elif cfg.dataset == 'babel':
        dataset_cls = DatasetBabel
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False)
    dataset_test = dataset_cls('test', t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None,
                          w_transi=cfg.vae_specs['w_transi'] if 'w_transi' in cfg.vae_specs else False)
    logger.info(f'Training data sequences {dataset.data_len:d}.')
    logger.info(f'Testing data sequences {dataset_test.data_len:d}.')
    if cfg.normalize_data:
        dataset.normalize_data()

    """model"""
    model = get_action_vae_model(cfg, dataset.traj_dim, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    optimizer = optim.Adam(model.parameters(), lr=cfg.vae_lr)
    scheduler = get_scheduler(optimizer, policy='lambda', nepoch_fix=cfg.num_vae_epoch_fix, nepoch=cfg.num_vae_epoch)
    logger.info(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])

    if mode == 'train':
        model.to(device)
        model.train()
        for i in range(args.iter, cfg.num_vae_epoch):
            train(i)
            test(i)
            if cfg.save_model_interval > 0 and (i + 1) % cfg.save_model_interval == 0:
                with to_cpu(model):
                    cp_path = cfg.vae_model_path % (i + 1)
                    model_cp = {'model_dict': model.state_dict(), 'meta': {'std': dataset.std, 'mean': dataset.mean}}
                    pickle.dump(model_cp, open(cp_path, 'wb'))
