import numpy as np
import os
from motion_pred.utils.dataset import Dataset
from motion_pred.utils.skeleton import Skeleton
from utils import paramUtil
import joblib
import torch
import torchgeometry


class DatasetACT12(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,is_6d=False, **kwargs):
        self.use_vel = use_vel
        if 'acts' in kwargs.keys() and kwargs['acts'] is not None:
            self.act_name = np.array(kwargs['acts'])
        else:
            # self.act_label = np.array(paramUtil.ntu_action_labels)
            self.act_name = np.array(list(paramUtil.humanact12_coarse_action_enumerator.values()))

        if 'max_len' in kwargs.keys() and kwargs['max_len'] is not None:
            self.max_len = np.array(kwargs['max_len'])
        else:
            self.max_len = None

        if 'min_len' in kwargs.keys() and kwargs['min_len'] is not None:
            self.min_len = np.array(kwargs['min_len'])
        else:
            self.min_len = None

        self.mode = mode

        if 'data_file' in kwargs.keys() and kwargs['data_file'] is not None:
            self.data_file = kwargs['data_file'].format(self.mode)
        else:
            self.data_file = os.path.join('./data', f'humanact12_{self.min_len}_wact_candi_{self.mode}.npz')

        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.std, self.mean = None, None
        # self.data_len = sum([seq.shape[0] for data_s in self.data.values() for seq in data_s.values()])
        self.traj_dim = 72-6
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.is_6d = is_6d
        if is_6d:
            self.traj_dim = self.traj_dim*2
        # if use_vel:
        #     self.traj_dim += 3
        self.process_data()
        self.data_len = sum([len(seq) for seq in self.data.values()])

    def process_data(self):
        print(f'load data from {self.data_file}')
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['data'].item()
        data_cand = data_o['data_cand'].item()


        # get data stats
        seq_n = []
        for key in data_f.keys():
            for tmp in data_f[key]:
                seq_n.append(tmp.shape[0])

        # get data stats
        seq_n = []
        for key in data_f.keys():
            for tmp in data_f[key]:
                seq_n.append(tmp.shape[0])

        if self.is_6d:
            data_f_6d = {}
            for key in data_f.keys():
                if key not in data_f_6d.keys():
                    data_f_6d[key] = []
                data_tmp = data_f[key]
                for i, seq in enumerate(data_tmp):
                    fn = seq.shape[0]
                    seq = seq.reshape([fn,-1,3]).reshape([-1,3])
                    rot = torchgeometry.angle_axis_to_rotation_matrix(torch.from_numpy(seq))#.data.numpy()
                    rot6d = rot[:,:3,:2].transpose(1,2).reshape([-1,6]).reshape([fn,-1,6]).reshape([fn,-1])
                    data_f_6d[key].append(rot6d.data.numpy())
            data_f = data_f_6d
        self.data = data_f
        self.data_cand = data_cand

    def sample(self, action=None, is_other_act=False, t_pre_extra=0,
               k = 0.08,max_trans_fn = 25):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item() - self.t_his + t_pre_extra

        seq = self.data[action]
        # seq = dict_s[action]
        idx = np.random.randint(0, len(seq))
        # fr_end = fr_start + self.t_total
        seq = seq[idx]
        fn = seq.shape[0]
        if fn // 10 > self.t_his:
            fr_start = np.random.randint(0, fn // 10 - self.t_his)
            seq = seq[fr_start:]
            fn = seq.shape[0]

        seq_his = seq[:self.t_his][None, :, :]
        seq_tmp = seq[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp
        seq_gt[0, fn:] = seq_tmp[-1:]
        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        fn_mask_gt = np.zeros([1, max_seq_len])
        fn_mask_gt[:, :fn + t_pre_extra] = 1
        label_gt = np.zeros(len(self.act_name))
        # tmp = str.lower(action.split(' ')[0])
        tmp = str.lower(action)
        label_gt[np.where(tmp == self.act_name)[0]] = 1
        label_gt = label_gt[None, :]

        # randomly find future sequences of other actions
        if is_other_act:
            seq_last = seq_his[0, -1:]
            seq_others = []
            fn_others = []
            fn_mask_others = []
            label_others = []
            cand_seqs = self.data_cand[f'{action}_{idx}']

            act_names = np.random.choice(self.act_name, len(self.act_name))
            for act in act_names:
                cand = cand_seqs[act]
                if len(cand) <= 0:
                    continue
                cand_idxs = np.random.choice(cand,min(10,len(cand)), replace=False)
                for cand_idx in cand_idxs:
                    cand_tmp = self.data[act][cand_idx]
                    cand_fn = cand_tmp.shape[0]
                    cand_his = cand_tmp[:max(cand_fn // 10, 25)]
                    dd = np.linalg.norm(cand_his - seq_last, axis=1)
                    cand_tmp = cand_tmp[np.where(dd == dd.min())[0][0]:]
                    cand_fn = cand_tmp.shape[0]
                    skip_fn = min(int(dd.min() // k + 1), max_trans_fn)
                    if cand_fn + skip_fn + self.t_his > self.max_len:
                        continue
                    # cand_tmp = np.copy(cand[[-1] * (self.max_len.item()-self.t_his)])[None, :, :]
                    cand_tt = np.zeros([1, max_seq_len, seq.shape[-1]])
                    cand_tt[0, :skip_fn] = cand_tmp[:1]
                    cand_tt[0, skip_fn:cand_fn + skip_fn] = cand_tmp
                    cand_tt[0, cand_fn + skip_fn:] = cand_tmp[-1:]
                    fn_tmp = np.zeros([1, max_seq_len])
                    fn_tmp[:, cand_fn + skip_fn - 1] = 1
                    fn_mask_tmp = np.zeros([1, max_seq_len])
                    fn_mask_tmp[:, skip_fn:cand_fn + skip_fn + t_pre_extra] = 1
                    cand_lab = np.zeros(len(self.act_name))
                    cand_lab[np.where(act == self.act_name)[0]] = 1
                    seq_others.append(cand_tt)
                    fn_others.append(fn_tmp)
                    fn_mask_others.append(fn_mask_tmp)
                    label_others.append(cand_lab[None, :])
                    break
                break

            if len(seq_others) > 0:
                seq_others = np.concatenate(seq_others, axis=0)
                fn_others = np.concatenate(fn_others, axis=0)
                fn_mask_others = np.concatenate(fn_mask_others, axis=0)
                label_others = np.concatenate(label_others, axis=0)

                seq_his = seq_his[[0] * (seq_others.shape[0] + 1)]
                seq_gt = np.concatenate([seq_gt, seq_others], axis=0)
                fn_gt = np.concatenate([fn_gt, fn_others], axis=0)
                fn_mask_gt = np.concatenate([fn_mask_gt, fn_mask_others], axis=0)
                label_gt = np.concatenate([label_gt, label_others], axis=0)

        return seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt

    def sampling_generator(self, num_samples=1000, batch_size=8, act=None, is_other_act=False, t_pre_extra=0,
                           act_trans_k=0.08, max_trans_fn=25, is_transi=False):
        for i in range(num_samples // batch_size):
            samp_his = []
            samp_gt = []
            fn = []
            fn_mask = []
            label = []
            for i in range(batch_size):
                seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt = self.sample(action=act, is_other_act=is_other_act,
                                                                           t_pre_extra=t_pre_extra,
                                                                           k=act_trans_k,max_trans_fn=max_trans_fn)
                samp_his.append(seq_his)
                samp_gt.append(seq_gt)
                fn.append(fn_gt)
                fn_mask.append(fn_mask_gt)
                label.append(label_gt)
            samp_his = np.concatenate(samp_his, axis=0)
            samp_gt = np.concatenate(samp_gt, axis=0)
            fn = np.concatenate(fn, axis=0)
            fn_mask = np.concatenate(fn_mask, axis=0)
            label = np.concatenate(label, axis=0)
            samp = np.concatenate([samp_his, samp_gt], axis=1)
            tmp = np.zeros_like(samp_his[:, :, 0])
            fn = np.concatenate([tmp, fn], axis=1)
            tmp = np.ones_like(samp_his[:, :, 0])
            fn_mask = np.concatenate([tmp, fn_mask], axis=1)
            yield samp[:,:,:self.traj_dim], label, fn, fn_mask

    def iter_generator(self, step=25):
        for data_s in self.data.values():
            for seq in data_s.values():
                seq_len = seq.shape[0]
                for i in range(0, seq_len - self.t_total, step):
                    traj = seq[None, i: i + self.t_total]
                    yield traj / 1000.


if __name__ == '__main__':
    np.random.seed(0)
    actions = {'WalkDog'}
    dataset = DatasetACT12('train')
    generator = dataset.sampling_generator()
    # dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data, action, fn in generator:
        print(data.shape)
