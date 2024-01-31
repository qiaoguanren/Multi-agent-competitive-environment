task_name="train-masac"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export CUDA_VISIBLE_DEVICES=0
python expert_policy.py --RL_config MASAC_episode500_epoch10_beta1e-1_seed1234 --task 2
# python val_marl.py --RL_config MASAC_eval --task 2
# python train_mappo.py --RL_config MASAC_episode500_epoch20_beta1e-1_seed1234
# python train_mappo.py --RL_config MASAC_episode500_epoch20_beta1e-1_seed2023