task_name="train-sac"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export CUDA_VISIBLE_DEVICES=0
# python train_marl.py --RL_config CCE-MASAC_episode500_epoch10_beta1e-2_seed666 --task 2
# python train_rl.py --RL_config SAC_episode500_epoch10_seed666 --task 2
python train_marl.py --RL_config MASAC_episode500_epoch10_seed666 --task 2