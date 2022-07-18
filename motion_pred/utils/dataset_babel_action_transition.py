import numpy as np
import os
from motion_pred.utils.dataset import Dataset


class DatasetBabel(Dataset):

    def __init__(self, mode, t_his=25, t_pred=100, actions='all', use_vel=False,
                 is_6d=False, w_transi=False, **kwargs):
        self.use_vel = use_vel
        if 'acts' in kwargs.keys() and kwargs['acts'] is not None:
            self.act_name = np.array(kwargs['acts'])
        else:
            self.act_name = np.array(['stand','walk','step','stretch','sit','place something',
                                      'take_pick something up','bend','stand up','jump','throw',
                                      'kick','run','catch','wave','squat','punch','jog','kneel','hop'])
        if 'max_len' in kwargs.keys() and kwargs['max_len'] is not None:
            self.max_len = np.array(kwargs['max_len'])
        else:
            self.max_len = 1000

        if 'min_len' in kwargs.keys() and kwargs['min_len'] is not None:
            self.min_len = np.array(kwargs['min_len'])
        else:
            self.min_len = 100

        self.mode = mode
        
        self.act_no_transi = ['sit', 'stand up']

        if 'data_file' in kwargs.keys() and kwargs['data_file'] is not None:
            self.data_file = kwargs['data_file'].format(self.mode)
        else:
            self.data_file = os.path.join('./data', f'babel_30_300_wact_candi_{self.mode}.npz')

        self.t_his = t_his
        self.t_pred = t_pred
        self.t_total = t_his + t_pred
        self.actions = actions
        self.traj_dim = 156
        self.normalized = False
        # iterator specific
        self.sample_ind = None
        self.w_transi = w_transi
        self.is_6d = is_6d
        if is_6d:
            self.traj_dim = self.traj_dim*2
        self.process_data()
        self.std, self.mean = None, None
        self.data_len = sum([len(seq) for seq in self.data.values()])

    def process_data(self):
        print(f'load data from {self.data_file}')
        data_o = np.load(self.data_file, allow_pickle=True)
        data_f = data_o['data'].item()
        data_cand = data_o['data_cand'].item()

        if len(data_f.keys()) != len(self.act_name):
            # get actions of interests
            data_f_tmp = {}
            for k,v in data_f.items():
                if k not in self.act_name:
                    continue
                data_f_tmp[k] = v
            data_cand_tmp = {}

            for k, v in data_cand.items():
                act_tmp = k.split('_')
                if len(act_tmp) == 3:
                    act_tmp = '_'.join(act_tmp[:-1])
                else:
                    act_tmp = act_tmp[0]
                if act_tmp not in self.act_name:
                    continue
                # if 'take_pick something up' in k:
                #     print(1)
                data_cand_tmp[k] = {}
                for k1,v1 in data_cand[k].items():
                    if k1 not in self.act_name:
                        continue
                    data_cand_tmp[k][k1] = v1
            data_f = data_f_tmp
            data_cand = data_cand_tmp

        if self.w_transi:
            print(f'load transition data from ./data/babel_transi_30_300_{self.mode}.npz')
            data_transi = np.load(f'./data/babel_transi_30_300_{self.mode}.npz',allow_pickle=True)['data'].item()
            self.data_transi = []
            for k,vs in data_transi.items():
                skip = False

                for act in self.act_no_transi:
                    if (act+'-') in k or ('-'+act) in k:
                        skip = True
                        break
                # ks = k.split('-')
                for act in vs[0]['act']:
                    if act not in self.act_name and (not act == 'transition'):
                        skip = True
                        break


                if (k.startswith('stand-') and k.endswith("-walk")) \
                        or (k.startswith('walk-') and k.endswith("-stand")):
                    skip = False

                if skip:
                    continue
                for v in vs:
                    if len(v['frame_split']) == 4:
                        transi_len = v['frame_split'][2]-v['frame_split'][1]
                        if transi_len >= 25:
                            continue
                    self.data_transi.append(v)

            self.num_transi = len(self.data_transi)
            print(f'transition sequence num {self.num_transi:d}.')

        self.data = data_f
        self.data_cand = data_cand

    def sample(self, action=None, is_other_act=False, t_pre_extra=0, k=0.08, max_trans_fn=25,
               is_transi=False, n_others=1):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item() - self.t_his + t_pre_extra
        seq = self.data[action]
        # seq = dict_s[action]
        idx = np.random.randint(0, len(seq))
        # fr_end = fr_start + self.t_total
        seq = seq[idx]['poses']
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
        # tmp = str.lower(action.split(' ')[0])
        label_gt[np.where(action == self.act_name)[0]] = 1
        assert np.sum(label_gt) == 1
        label_gt = label_gt[None, :]

        # randomly find future sequences of other actions
        if is_other_act:
            # k = 0.08
            # max_trans_fn = 25
            seq_last = seq_his[0, -1:]
            seq_others = []
            fn_others = []
            fn_mask_others = []
            label_others = []
            cand_seqs = self.data_cand[f'{action}_{idx}']

            act_names = np.random.choice(self.act_name, len(self.act_name), replace=False)
            count = 0
            for act in act_names:
                cand = cand_seqs[act][:5]
                if len(cand) <= 0:
                    continue
                for _ in range(10):
                    cand_idx = np.random.choice(cand, 1)[0]
                    cand_tmp = self.data[act][cand_idx]['poses']
                    cand_fn = cand_tmp.shape[0]
                    cand_his = cand_tmp[:max(cand_fn // 10, 10)]
                    dd = np.linalg.norm(cand_his - seq_last, axis=1)
                    cand_tmp = cand_tmp[np.where(dd == dd.min())[0][0]:]
                    cand_fn = cand_tmp.shape[0]
                    skip_fn = min(int(dd.min() // k + 1), max_trans_fn)
                    if cand_fn + skip_fn > max_seq_len:
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
                    count += 1
                    break
                if count == n_others:
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

        # randomly find sequences with transition
        if is_transi:
            if np.random.rand(1)[0] < 0.3:
                idx = np.random.randint(0, self.num_transi)
                data_transi = self.data_transi[idx]

                transi_tmp = data_transi['poses']
                sf = data_transi['frame_split'][1] - self.t_his

                seq_his_transi = transi_tmp[None, sf:sf + self.t_his]
                seq_transi_tmp = transi_tmp[None, sf + self.t_his:]
                fn_transi = seq_transi_tmp.shape[1]
                if fn_transi <= max_seq_len:
                    seq_transi_gt = np.zeros([1, max_seq_len, seq_his_transi.shape[-1]])
                    seq_transi_gt[:, :fn_transi] = seq_transi_tmp
                    seq_transi_gt[:, fn_transi:] = seq_transi_tmp[:, -1:]
                    fn_transi_gt = np.zeros([1, max_seq_len])
                    fn_transi_gt[:, fn_transi - 1] = 1
                    fn_transi_mask_gt = np.zeros([1, max_seq_len])
                    fn_transi_mask_gt[:, :fn_transi + t_pre_extra] = 1
                    label_transi_gt = np.zeros(len(self.act_name))
                    label_transi_gt[np.where(data_transi['act'][-1] == self.act_name)[0]] = 1
                    assert np.sum(label_transi_gt) == 1
                    label_transi_gt = label_transi_gt[None, :]

                    seq_his = np.concatenate([seq_his, seq_his_transi], axis=0)
                    seq_gt = np.concatenate([seq_gt, seq_transi_gt], axis=0)
                    fn_gt = np.concatenate([fn_gt, fn_transi_gt], axis=0)
                    fn_mask_gt = np.concatenate([fn_mask_gt, fn_transi_mask_gt], axis=0)
                    label_gt = np.concatenate([label_gt, label_transi_gt], axis=0)

        return seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt

    def sample_all_act(self,action=None, is_other_act=False,t_pre_extra=0, k=0.08, max_trans_fn=25,
               is_transi=False, n_others=1):
        if action is None:
            action = np.random.choice(self.act_name)

        max_seq_len = self.max_len.item()-self.t_his + t_pre_extra
        seq = self.data[action]
        # seq = dict_s[action]
        idx = np.random.randint(0, len(seq))
        # fr_end = fr_start + self.t_total
        seq = seq[idx]['poses']
        fn = seq.shape[0]
        if fn // 10 > self.t_his:
            fr_start = np.random.randint(0, fn // 10 - self.t_his)
            seq = seq[fr_start:]
            fn = seq.shape[0]

        seq_his = seq[:self.t_his][None,:,:]
        seq_tmp = seq[self.t_his:]
        fn = seq_tmp.shape[0]
        seq_gt = np.zeros([1, max_seq_len, seq.shape[-1]])
        seq_gt[0, :fn] = seq_tmp
        seq_gt[0,fn:] = seq_tmp[-1:]
        fn_gt = np.zeros([1, max_seq_len])
        fn_gt[:, fn - 1] = 1
        fn_mask_gt = np.zeros([1, max_seq_len])
        fn_mask_gt[:, :fn+t_pre_extra] = 1
        label_gt = np.zeros(len(self.act_name))
        # tmp = str.lower(action.split(' ')[0])
        # tmp = str.lower(action.split(' ')[0])
        label_gt[np.where(action == self.act_name)[0]] = 1
        assert np.sum(label_gt) == 1
        label_gt = label_gt[None,:]

        # randomly find future sequences of other actions
        if is_other_act:
            # k = 0.08
            # max_trans_fn = 25
            seq_last = seq_his[0,-1:]
            seq_others = []
            fn_others = []
            fn_mask_others = []
            label_others = []
            cand_seqs = self.data_cand[f'{action}_{idx}']

            # act_names = np.random.choice(self.act_name, len(self.act_name),replace=False)
            # count = 0
            for act in self.act_name:
                cand = cand_seqs[act][:5]
                if len(cand)<=0:
                    continue
                for _ in range(5):
                    cand_idx = np.random.choice(cand, 1)[0]
                    cand_tmp = self.data[act][cand_idx]['poses']
                    cand_fn = cand_tmp.shape[0]
                    cand_his = cand_tmp[:max(cand_fn//10,10)]
                    dd = np.linalg.norm(cand_his-seq_last, axis=1)
                    cand_tmp = cand_tmp[np.where(dd==dd.min())[0][0]:]
                    cand_fn = cand_tmp.shape[0]
                    skip_fn = min(int(dd.min()//k + 1), max_trans_fn)
                    if cand_fn + skip_fn > max_seq_len:
                        continue
                    # cand_tmp = np.copy(cand[[-1] * (self.max_len.item()-self.t_his)])[None, :, :]
                    cand_tt = np.zeros([1, max_seq_len, seq.shape[-1]])
                    cand_tt[0, :skip_fn] = cand_tmp[:1]
                    cand_tt[0, skip_fn:cand_fn+skip_fn] = cand_tmp
                    cand_tt[0,cand_fn+skip_fn:] = cand_tmp[-1:]
                    fn_tmp = np.zeros([1, max_seq_len])
                    fn_tmp[:, cand_fn+skip_fn-1] = 1
                    fn_mask_tmp = np.zeros([1, max_seq_len])
                    fn_mask_tmp[:, skip_fn:cand_fn+skip_fn+t_pre_extra] = 1
                    cand_lab = np.zeros(len(self.act_name))
                    cand_lab[np.where(act == self.act_name)[0]] = 1
                    seq_others.append(cand_tt)
                    fn_others.append(fn_tmp)
                    fn_mask_others.append(fn_mask_tmp)
                    label_others.append(cand_lab[None,:])
                    # count += 1
                    break
                # if count == n_others:
                #     break

            if len(seq_others) > 0:
                seq_others = np.concatenate(seq_others,axis=0)
                fn_others = np.concatenate(fn_others,axis=0)
                fn_mask_others = np.concatenate(fn_mask_others,axis=0)
                label_others = np.concatenate(label_others,axis=0)

                seq_his = seq_his[[0]*(seq_others.shape[0]+1)]
                seq_gt = np.concatenate([seq_gt,seq_others], axis=0)
                fn_gt = np.concatenate([fn_gt,fn_others], axis=0)
                fn_mask_gt = np.concatenate([fn_mask_gt,fn_mask_others], axis=0)
                label_gt = np.concatenate([label_gt, label_others], axis=0)

        # randomly find sequences with transition
        if is_transi:
            if np.random.rand(1)[0] < 0.3:
                idx = np.random.randint(0,self.num_transi)
                data_transi = self.data_transi[idx]

                transi_tmp = data_transi['poses']
                sf = data_transi['frame_split'][1]-self.t_his

                seq_his_transi = transi_tmp[None,sf:sf+self.t_his]
                seq_transi_tmp = transi_tmp[None,sf+self.t_his:]
                fn_transi = seq_transi_tmp.shape[1]
                if fn_transi <= max_seq_len:
                    seq_transi_gt = np.zeros([1, max_seq_len, seq_his_transi.shape[-1]])
                    seq_transi_gt[:, :fn_transi] = seq_transi_tmp
                    seq_transi_gt[:, fn_transi:] = seq_transi_tmp[:,-1:]
                    fn_transi_gt = np.zeros([1, max_seq_len])
                    fn_transi_gt[:, fn_transi - 1] = 1
                    fn_transi_mask_gt = np.zeros([1, max_seq_len])
                    fn_transi_mask_gt[:, :fn_transi + t_pre_extra] = 1
                    label_transi_gt = np.zeros(len(self.act_name))
                    label_transi_gt[np.where(data_transi['act'][-1] == self.act_name)[0]] = 1
                    assert np.sum(label_transi_gt) == 1
                    label_transi_gt = label_transi_gt[None, :]

                    seq_his = np.concatenate([seq_his,seq_his_transi],axis=0)
                    seq_gt = np.concatenate([seq_gt, seq_transi_gt], axis=0)
                    fn_gt = np.concatenate([fn_gt, fn_transi_gt], axis=0)
                    fn_mask_gt = np.concatenate([fn_mask_gt, fn_transi_mask_gt], axis=0)
                    label_gt = np.concatenate([label_gt, label_transi_gt], axis=0)

        return seq_his,seq_gt,fn_gt,fn_mask_gt,label_gt

    def sampling_generator(self, num_samples=1000, batch_size=8,act=None,is_other_act=False,t_pre_extra=0,
                           act_trans_k=0.08, max_trans_fn=25, is_transi=False,n_others=1,others_all_act=False):
        for i in range(num_samples // batch_size):
            samp_his = []
            samp_gt = []
            fn = []
            fn_mask = []
            label = []
            for i in range(batch_size):
                if others_all_act:
                    seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt = self.sample_all_act(action=act,is_other_act=is_other_act,
                                                                               t_pre_extra=t_pre_extra,
                                                                               k=act_trans_k,max_trans_fn=max_trans_fn,
                                                                               is_transi=is_transi,n_others=n_others)
                else:
                    seq_his, seq_gt, fn_gt, fn_mask_gt, label_gt = self.sample(action=act,is_other_act=is_other_act,
                                                                               t_pre_extra=t_pre_extra,
                                                                               k=act_trans_k,max_trans_fn=max_trans_fn,
                                                                               is_transi=is_transi,n_others=n_others)
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
            samp = np.concatenate([samp_his,samp_gt],axis=1)
            tmp = np.zeros_like(samp_his[:,:,0])
            fn = np.concatenate([tmp,fn],axis=1)
            tmp = np.ones_like(samp_his[:,:,0])
            fn_mask = np.concatenate([tmp,fn_mask],axis=1)
            yield samp,label, fn, fn_mask

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
    dataset = DatasetGrab('train')
    generator = dataset.sampling_generator()
    # dataset.normalize_data()
    # generator = dataset.iter_generator()
    for data, action, fn in generator:
        print(data.shape)
