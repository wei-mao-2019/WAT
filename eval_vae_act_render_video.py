import os
import sys
import math
import pickle
import argparse
import time
from torch import optim
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import csv
from numba import cuda

sys.path.append(os.getcwd())
from utils import *
from motion_pred.utils.config import Config
from motion_pred.utils.dataset_ntu_act_transition import DatasetNTU
from motion_pred.utils.dataset_grab_action_transition import DatasetGrab
from motion_pred.utils.dataset_humanact12_act_transition import DatasetACT12
from motion_pred.utils.dataset_babel_action_transition import DatasetBabel
from models.motion_pred import *
from utils.fid import calculate_frechet_distance
from utils.dtw import batch_dtw_torch, batch_dtw_torch_parallel, accelerated_dtw, batch_dtw_cpu_parallel
from utils import eval_util
from utils import data_utils
from utils.vis_util import render_videos_new

def get_stop_sign(Y_r,args):
    # get stop sign
    if args.stop_fn > 0:
        fn_tmp = Y_r.shape[0]
        tmp1 = np.arange(fn_tmp)[:, None]
        tmp2 = np.arange(args.stop_fn)[None, :]
        idxs = tmp1 + tmp2
        idxs[idxs > fn_tmp - 1] = fn_tmp - 1
        yr_tmp = Y_r[idxs]
        yr_mean = yr_tmp.mean(dim=1, keepdim=True)
        dr = torch.mean(torch.norm(yr_tmp - yr_mean, dim=-1), dim=1)
    else:
        dr = torch.norm(Y_r[:-1] - Y_r[1:], dim=2)
        dr = torch.cat([dr[:1, :], dr], dim=0)
    threshold = args.threshold
    tmp = dr < threshold
    idx = torch.arange(tmp.shape[0], 0, -1, device=device)[:, None]
    tmp2 = tmp * idx
    tmp2[:dataset.min_len - 1] = 0
    tmp2[-1, :] = 1
    fn = tmp2 == tmp2.max(dim=0, keepdim=True)[0]
    fn = fn.float()
    return fn

def val(epoch):
    seq_len = []
    with torch.no_grad():
        for i, act in enumerate(dataset.act_name):
            st = time.time()
            generator = dataset.sampling_generator(num_samples=args.num_samp, batch_size=args.bs,t_pre_extra=args.t_pre_extra,
                                                   act=act)
            for traj_np, label, fn_gt, fn_mask_gt in generator:
                seq_gt = np.where(fn_gt == 1)[1]
                traj_tmp = tensor(traj_np, device=device, dtype=dtype).permute(1, 0, 2).contiguous()
                seq_n, bs, dim = traj_tmp.shape
                traj = traj_tmp[:, :, None, None, :].repeat([1, 1, cfg.vae_specs['n_action'], args.nk, 1]) \
                    .reshape([seq_n, -1, dim])
                label = torch.eye(cfg.vae_specs['n_action'], device=device, dtype=dtype)
                label = label[None, :, None, :].repeat([bs, 1, args.nk, 1]).reshape([-1, cfg.vae_specs['n_action']])

                X = traj[:t_his]
                if cfg.dataset == 'babel':
                    index_used = list(range(30)) + list(range(36, 66))
                    X = X[:, :, index_used]
                Y_r = model.sample_prior(X, label)

                Y_r = torch.cat([X, Y_r], dim=0)

                fn = get_stop_sign(Y_r,args)
                seq_l = torch.where(fn[cfg.t_his:].transpose(0, 1) == 1)[1].cpu().data.numpy()+1
                seq_len.append(seq_l)
                seq_l = seq_l.reshape([-1, args.nk])
                seq_l = torch.where(fn.transpose(0, 1) == 1)[1].cpu().data.numpy() + 1
                seq_l = seq_l.reshape([bs, cfg_classifier.vae_specs['n_action'], args.nk])

                x = traj_tmp.cpu().data.numpy()

                if cfg.dataset == 'babel':
                    traj_tmp = torch.clone(traj)
                    index_used = list(range(30)) + list(range(36, 66))
                    traj_tmp[:, :, index_used] = Y_r
                    Y_r = traj_tmp.clone()

                y = Y_r.reshape([-1,bs, cfg_classifier.vae_specs['n_action'], args.nk,Y_r.shape[-1]]).cpu().data.numpy()
                betas = np.zeros(10)
                for ii in range(args.bs):

                    sequence = {'poses': x[:, ii][:seq_gt[ii]], 'betas': betas}
                    key = f'{act}_{ii}_gt'
                    render_videos_new(sequence, device, cfg.result_dir + f'/{args.mode}', key, w_golbalrot=True, smpl_model=smpl_model)

                    for jj in range(cfg_classifier.vae_specs['n_action']):
                        for kk in range(2):
                            sequence = {'poses': y[:, ii,jj,kk][:seq_l[ii,jj,kk]], 'betas': betas}
                            key = f'{act}_{ii}_{dataset.act_name[jj]}_{kk}'
                            render_videos_new(sequence, device, cfg.result_dir + f'/{args.mode}', key, w_golbalrot=True, smpl_model=smpl_model)

            print(f">>>> action {act} time used {time.time()-st:.3f}")

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='babel_rnn')
    parser.add_argument('--cfg_classifier', default='babel_act_classifier')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=500)
    parser.add_argument('--nk', type=int, default=5)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.01)
    parser.add_argument('--stop_fn', type=int, default=5)
    parser.add_argument('--bs', type=int, default=5)
    parser.add_argument('--num_samp', type=int, default=5)
    parser.add_argument('--data_type', default='float32')
    args = parser.parse_args()

    """setup"""
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.data_type == 'float32':
        dtype = torch.float32
    else:
        dtype = torch.float64
    torch.set_default_dtype(dtype)
    device = torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')
    if torch.cuda.is_available():
        torch.cuda.set_device(args.gpu_index)
        cuda.select_device(args.gpu_index)
    cfg = Config(args.cfg, test=args.test)
    cfg_classifier = Config(args.cfg_classifier, test=args.test)
    # tb_logger = SummaryWriter(cfg.tb_dir) if args.mode == 'train' else None
    logger = create_logger(os.path.join(cfg.log_dir, 'log_eval.txt'))

    """parameter"""
    mode = args.mode
    nz = cfg.nz
    t_his = cfg.t_his
    t_pred = cfg.t_pred
    if 't_pre_extra' in cfg.vae_specs:
        args.t_pre_extra = cfg.vae_specs['t_pre_extra']

    """data"""
    if cfg.dataset == 'grab':
        dataset_cls = DatasetGrab
        smpl_model = 'smplx'
    elif cfg.dataset == 'ntu':
        dataset_cls = DatasetNTU
        smpl_model = 'smpl'
    elif cfg.dataset == 'humanact12':
        dataset_cls = DatasetACT12
        smpl_model = 'smpl'
    elif cfg.dataset == 'babel':
        dataset_cls = DatasetBabel
        smpl_model = 'smplh'

    # for act in cfg.vae_specs['actions']:
    dataset = dataset_cls(args.mode, t_his, t_pred, actions='all', use_vel=cfg.use_vel,
                          acts=cfg.vae_specs['actions'] if 'actions' in cfg.vae_specs else None,
                          max_len=cfg.vae_specs['max_len'] if 'max_len' in cfg.vae_specs else None,
                          min_len=cfg.vae_specs['min_len'] if 'min_len' in cfg.vae_specs else None,
                          is_6d=cfg.vae_specs['is_6d'] if 'is_6d' in cfg.vae_specs else False,
                          data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)

    """model"""
    if cfg.dataset == 'babel':
        dataset.traj_dim = 60
        model = get_action_vae_model(cfg, 60, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    else:
        model = get_action_vae_model(cfg, dataset.traj_dim, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

    if args.iter > 0:
        cp_path = cfg.vae_model_path % args.iter
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model.load_state_dict(model_cp['model_dict'])
    model.to(device)
    model.eval()

    """action classifier model"""
    model_classifier = get_action_classifier(cfg_classifier, dataset.traj_dim, max_len=dataset.max_len)
    print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model_classifier.parameters())) / 1000000.0))
    cp_path = cfg_classifier.vae_model_path % (100 if cfg.dataset == 'babel' else 500)
    print('loading model from checkpoint: %s' % cp_path)
    model_cp = pickle.load(open(cp_path, "rb"))
    model_classifier.load_state_dict(model_cp['model_dict'])
    model_classifier.to(device)
    model_classifier.eval()

    val(args.iter)
