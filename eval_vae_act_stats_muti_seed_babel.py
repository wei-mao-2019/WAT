import os
import sys
import math
import pickle
import argparse
import time

import numpy as np
import torch
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
def get_stop_sign_from_ss(ss):
    tmp = ss >= 0.5
    idx = torch.arange(tmp.shape[1], 0, -1, device=device)[None,:]
    tmp2 = tmp * idx
    tmp2[:,:dataset.min_len-t_his - 1] = 0
    tmp2[:,-1] = 1
    fn = tmp2 == tmp2.max(dim=1, keepdim=True)[0]
    fn = fn.float().transpose(0,1)
    fn = torch.cat([torch.zeros_like(fn[:t_his]),fn],dim=0)
    return fn

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

def get_diversity_DTW(Y_r, fn, args, cfg,seq_l=None):
    bs = args.bs
    traj_dim = Y_r.shape[-1]

    # convert to cpu tensor
    Y_r = Y_r#.cpu()

    # diversity after DTW
    fn_mask_inv = torch.cumsum(fn, dim=0)
    fn_mask_inv[fn == 1] = 0
    # pad
    fn_mask_inv = torch.cat([fn_mask_inv, torch.ones_like(fn_mask_inv[:1])], dim=0)
    fn_mask_inv = fn_mask_inv[:, :, None].repeat([1, 1, Y_r.shape[-1]])
    yr_tmp = Y_r.clone()
    # pad
    yr_tmp = torch.cat([yr_tmp, yr_tmp[-1:]], dim=0)
    yr_tmp[fn_mask_inv == 1] = 1e10
    yr = yr_tmp[cfg.t_his:].reshape([-1, bs, cfg.vae_specs['n_action'], args.nk, traj_dim]). \
        permute(1, 2, 3, 0, 4).reshape([bs * cfg.vae_specs['n_action'], args.nk, -1, traj_dim])
    # seq_len = tmp2.max(dim=0, keepdim=True)[1][0].reshape([-1, bs*cfg.vae_specs['n_action']]).\
    #     permute(1,0).cpu().data.numpy()-cfg.t_his
    if seq_l is None:
        seq_l = torch.where(fn[cfg.t_his:].transpose(0, 1) == 1)[1].cpu().data.numpy().reshape([-1, args.nk]) + 1
    seq1 = []
    seq2 = []
    seq_l_1 = np.array([])
    seq_l_2 = np.array([])
    for ii in range(args.nk):
        for jj in range(ii + 1, args.nk):
            seq_l_1 = np.append(seq_l_1, seq_l[:, ii])
            seq_l_2 = np.append(seq_l_2, seq_l[:, jj])
            # ml_j = seq_l[:,jj].max()+1
            seq1.append(yr[:, ii])
            seq2.append(yr[:, jj])
    seq1 = torch.cat(seq1, dim=0)#.data.numpy()
    seq2 = torch.cat(seq2, dim=0)#.data.numpy()
    cost, sl = batch_dtw_torch_parallel(seq1, seq2, seq_l_1, seq_l_2)
    # cost, sl = batch_dtw_cpu_parallel(seq1, seq2, seq_l_1, seq_l_2)
    return cost, sl

def val(epoch):
    t_s = time.time()
    train_losses = 0
    total_num_sample = 0
    loss_names = ['TOTAL', 'MSE', 'MSE_v', 'KLD']
    feat = []
    accuracy = 0
    seq_len = []
    smooth_dist = 0
    smooth_dist_gt = 0
    smooth_dist_rot = 0
    smooth_dist_gt_rot = 0
    accele_est = 0
    accele_gt = 0
    diversity = 0
    diversity_perframe = np.zeros(dataset.max_len-cfg.t_his)
    diversity_perframe_rot = np.zeros(dataset.max_len-cfg.t_his)
    diversity_dtw = 0
    diversity_dtw_rot = 0

    if cfg.vae_specs['model_name'] == 'trans_vae_v3_3' or cfg.vae_specs['model_name'] == 'v3_6_1':
        with_stop_sign = True
    else:
        with_stop_sign = False

    with torch.no_grad():
        for act in dataset.act_name:
            st = time.time()
            generator = dataset.sampling_generator(num_samples=args.num_samp, batch_size=args.bs,t_pre_extra=args.t_pre_extra,
                                                   act=act)
            # traj_est = []
            # label_est = []
            # fn_est = []

            # traj_gt = []
            # label_gt = []
            # fn_gt = []
            acc_peract = 0
            total_peract = 0
            for traj_np, label, fn_gt, fn_mask_gt in generator:
                # label_gt.append(label)
                # traj_gt.append(traj_np.transpose([1,0,2]))
                # fn_gt.append(fn.transpose([1,0]))

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
                # Y = traj[t_his:]
                if with_stop_sign:
                    Y_r,ss = model.sample_prior(X, label)
                else:
                    Y_r = model.sample_prior(X, label)

                if 'is_6d' in cfg.vae_specs and cfg.vae_specs['is_6d']:
                    from utils.utils import compute_rotation_matrix_from_ortho6d
                    yr_proj = compute_rotation_matrix_from_ortho6d(Y_r.reshape([-1, 6]))
                    yr_proj = yr_proj[:, :, :2].transpose(1, 2).reshape([-1, 6]).reshape(Y_r.shape)
                    Y_r = yr_proj.clone()

                smooth_dist += torch.sum(torch.norm(Y_r[0] - X[-1], dim=1)).item()
                smooth_dist_gt += torch.sum(torch.norm(traj[:-1] - traj[1:], dim=2)).item() / (traj.shape[0] - 1)

                Y_r = torch.cat([X, Y_r], dim=0)
                if with_stop_sign:
                    fn = get_stop_sign_from_ss(ss)
                else:
                    fn = get_stop_sign(Y_r,args)
                seq_l = torch.where(fn[cfg.t_his:].transpose(0, 1) == 1)[1].cpu().data.numpy()+1
                seq_len.append(seq_l)
                seq_l = seq_l.reshape([-1, args.nk])

                """
                get perceptual feature
                """
                lest, h, hx = model_classifier(Y_r, fn.transpose(0, 1), is_feat=True)
                # lgt = torch.where(label==1)[1]
                lest = lest == torch.max(lest, dim=1, keepdim=True)[0]
                accuracy += torch.sum(label * lest).item()
                total_num_sample += label.shape[0]
                acc_peract += torch.sum(label * lest).item()
                total_peract += label.shape[0]
                feat.append(hx.cpu().data.numpy())

                """
                get diversity
                """
                yr = Y_r[cfg.t_his:cfg.vae_specs['max_len']].reshape([-1, bs, cfg.vae_specs['n_action'], args.nk, 60]). \
                    reshape([-1, args.nk, 60])
                mask = torch.tril(torch.ones([args.nk, args.nk], device=device)) == 0
                div_tmp = torch.cdist(yr, yr, p=2)[:, mask].mean(dim=-1)\
                    .reshape([-1, bs, cfg.vae_specs['n_action']]).mean(dim=(1,2)).cpu().data.numpy()
                diversity_perframe += div_tmp * label.shape[0]
                # maxlen
                # maxlen = seq_len[-1].max()
                # st1 = time.time()
                cost,sl = get_diversity_DTW(Y_r, fn, args, cfg,seq_l=seq_l)
                # print(f"{time.time()-st1:.3f}")
                diversity_dtw += (cost/sl).mean() * label.shape[0]
            print(f">>>> action {act} time used {time.time()-st:.3f}")
            logger.info(f">>>> action {act} accuracy {acc_peract/total_peract:.3f}")
    smooth_dist_gt = smooth_dist_gt / total_num_sample
    smooth_dist = smooth_dist / total_num_sample

    smooth_dist_gt_rot = smooth_dist_gt_rot / total_num_sample
    smooth_dist_rot = smooth_dist_rot / total_num_sample

    diversity_perframe = diversity_perframe / total_num_sample
    diversity_dtw = diversity_dtw / total_num_sample

    diversity_perframe_rot = diversity_perframe_rot / total_num_sample
    diversity_dtw_rot = diversity_dtw_rot / total_num_sample

    accuracy = accuracy / total_num_sample
    feat = np.concatenate(feat, axis=0)
    mu2 = feat.mean(axis=0)
    cov2 = np.matmul(feat.transpose(1, 0), feat) / feat.shape[0]
    test_data = np.load(cfg_classifier.result_dir + f'/epo100_seed0_test.npz', allow_pickle=True)['data'].item()
    mu1 = test_data['mu']
    cov1 = test_data['cov']
    fid = calculate_frechet_distance(mu1, cov1, mu2, cov2)
    # logger.info(f' accuracy: {accuracy:.3f}, fid {fid:.3f}')
    test_data = np.load(cfg_classifier.result_dir + f'/epo100_seed0_train.npz', allow_pickle=True)['data'].item()
    mu1 = test_data['mu']
    cov1 = test_data['cov']
    fid2 = calculate_frechet_distance(mu1, cov1, mu2, cov2)
    logger.info(
        f'epo {args.iter} mode {args.mode} threshold {args.threshold} action classifier {args.cfg_classifier} accuracy: {accuracy:.3f}, test fid {fid:.3f}, train fid {fid2:.3f}, smoothness gt {smooth_dist_gt:.3f}, smoothness {smooth_dist:.3f}, div at 100frame {diversity:.3f}')

    postfix = 'stopfn'

    stats = {}
    stats['accuracy'] = accuracy*100
    stats['fid_train'] = fid2
    stats['fid_test'] = fid
    stats['div_dtw'] = diversity_dtw
    stats['div_avg_per_frame'] = diversity_perframe.mean()
    stats['lastframe_dist_est'] = smooth_dist
    stats['frame_dist_gt'] = smooth_dist_gt
    return stats

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--cfg', default='babel_rnn')
    parser.add_argument('--cfg_classifier', default='babel_act_classifier_v2')
    parser.add_argument('--mode', default='test')
    parser.add_argument('--test', action='store_true', default=False)
    parser.add_argument('--iter', type=int, default=480)
    parser.add_argument('--nk', type=int, default=2)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--num_seed', type=int, default=1)
    parser.add_argument('--gpu_index', type=int, default=1)
    parser.add_argument('--threshold', type=float, default=0.010)
    parser.add_argument('--stop_fn', type=int, default=5)
    parser.add_argument('--bs', type=int, default=2)
    parser.add_argument('--num_samp', type=int, default=10)
    args = parser.parse_args()

    """setup"""
    state = np.random.get_state()
    np.random.seed(args.seed)
    rand_seeds = np.random.choice(np.arange(100), replace=False, size=(args.num_seed))
    np.random.set_state(state)
    tmp = []
    for seed in rand_seeds:
        np.random.seed(seed)
        torch.manual_seed(seed)
        dtype = torch.float32
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
            args.t_pre_extra = cfg.vae_specs['t_pre_extra'] = 0

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
                              data_file=cfg.vae_specs['data_file'] if 'data_file' in cfg.vae_specs else None)
        logger.info(f'data sequences {dataset.data_len:d}.')
        if cfg.normalize_data:
            dataset.normalize_data()

        """model"""
        model = get_action_vae_model(cfg, 60, max_len=dataset.max_len - cfg.t_his + cfg.vae_specs['t_pre_extra'])
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model.parameters())) / 1000000.0))

        if args.iter > 0:
            cp_path = cfg.vae_model_path % args.iter
            print('loading model from checkpoint: %s' % cp_path)
            model_cp = pickle.load(open(cp_path, "rb"))
            model.load_state_dict(model_cp['model_dict'])
        model.to(device)
        model.eval()

        """action classifier model"""
        # model_classifier = get_action_classifier(cfg_classifier, dataset.traj_dim, max_len=dataset.max_len)
        model_classifier = get_action_classifier(cfg_classifier, 60, max_len=dataset.max_len)
        print(">>> total params: {:.2f}M".format(sum(p.numel() for p in list(model_classifier.parameters())) / 1000000.0))
        cp_path = cfg_classifier.vae_model_path % 100
        print('loading model from checkpoint: %s' % cp_path)
        model_cp = pickle.load(open(cp_path, "rb"))
        model_classifier.load_state_dict(model_cp['model_dict'])
        model_classifier.to(device)
        model_classifier.eval()

        stats = val(args.iter)
        tmp2 = []
        for key in stats.keys():
            tmp2.append(stats[key])
        tmp.append(tmp2)

        postfix = f'epo{args.iter}_{args.mode}_{args.cfg_classifier}_nsamp{args.num_samp}_stfn{args.stop_fn:d}_th{args.threshold:.3f}_nk{args.nk}'
        whead = False
        if not os.path.exists('%s/stats_multirun_%s_%s.csv' % (cfg.result_dir, postfix, args.seed)):
            whead = True
        with open('%s/stats_multirun_%s_%s.csv' % (cfg.result_dir, postfix, args.seed), 'a') as csv_file:
            writer = csv.DictWriter(csv_file, fieldnames=list(stats.keys()))
            if whead:
                writer.writeheader()
            writer.writerow(stats)
    tmp = np.array(tmp)
    std = np.std(tmp,axis=0)
    mean = np.mean(tmp,axis=0)
    stat_tmp = {}
    for i, key in enumerate(stats.keys()):
        stat_tmp[key] = mean[i]
    whead = False
    if not os.path.exists('%s/stats_multirun_%s_%s.csv' % (cfg.result_dir, postfix, args.seed)):
        whead = True
    with open('%s/stats_multirun_%s_%s.csv' % (cfg.result_dir, postfix, args.seed), 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(stat_tmp.keys()))
        if whead:
            writer.writeheader()
        writer.writerow(stat_tmp)

    stat_tmp = {}
    for i, key in enumerate(stats.keys()):
        stat_tmp[key] = std[i]

    whead = False
    if not os.path.exists('%s/stats_multirun_%s_%s.csv' % (cfg.result_dir, postfix, args.seed)):
        whead = True
    with open('%s/stats_multirun_%s_%s.csv' % (cfg.result_dir, postfix, args.seed), 'a') as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=list(stat_tmp.keys()))
        if whead:
            writer.writeheader()
        writer.writerow(stat_tmp)