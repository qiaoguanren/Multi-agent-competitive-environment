task_name="train-masac"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export CUDA_VISIBLE_DEVICES=3
python train_mappo.py --RL_config CCE-MASAC_episode500_epoch10_beta1e-2_seed666 --task 2
# python train_mappo.py --RL_config MASAC_episode500_epoch10_beta1e-1_seed1234 --task 2
