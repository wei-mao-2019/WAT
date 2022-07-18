#!/bin/bash
# training script
CFG="grab_rnn"
GPU_IDX=0
python python exp_vae_act.py --gpu_index $GPU_IDX --cfg $CFG --is_other_act --seed 0

CFG="babel_rnn"
GPU_IDX=0
python python exp_vae_act_babel.py --gpu_index $GPU_IDX --cfg $CFG --is_other_act --seed 0

CFG="ntu_rnn"
GPU_IDX=0
python python exp_vae_act.py --gpu_index $GPU_IDX --cfg $CFG --is_other_act --seed 0

CFG="humanact12_rnn"
GPU_IDX=0
python python exp_vae_act.py --gpu_index $GPU_IDX --cfg $CFG --is_other_act --seed 0




# testing script
CFG="grab_rnn"
CFG_CLASS=grab_act_classifier
GPU_IDX=0
TH=0.015
python eval_vae_act_stats_muti_seed.py --iter 500 --nk 10 --bs 5 --num_samp 50 --num_seed 1 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH


CFG="babel_rnn"
CFG_CLASS=babel_act_classifier
GPU_IDX=0
TH=0.025
python eval_vae_act_stats_muti_seed_babel.py --iter 500 --nk 10 --bs 5 --num_samp 50 --num_seed 1 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH

CFG="ntu_rnn"
CFG_CLASS=ntu_act_classifier
GPU_IDX=0
TH=0.025
python eval_vae_act_stats_muti_seed.py --iter 500 --nk 10 --bs 5 --num_samp 50 --num_seed 1 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH


CFG="humanact12_rnn"
CFG_CLASS=humanact12_act_classifier
GPU_IDX=0
TH=0.01
python eval_vae_act_stats_muti_seed.py --iter 500 --nk 10 --bs 5 --num_samp 50 --num_seed 1 --stop_fn 5 --cfg $CFG --cfg_classifier $CFG_CLASS --gpu_index $GPU_IDX --threshold $TH
