task_name="train-masac"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-local-${task_name}-${launch_time}.out"
export CUDA_VISIBLE_DEVICES=0
# python train_ppo.py --RL_config BC_seed1234 --task 1
# python train_ppo.py --RL_config BC_seed1000 --task 1
# python train_ppo.py --RL_config BC_seed666 --task 1
# python train_ppo.py --RL_config MAPPO_episode500_epoch20_beta1e-1_seed1000 --task 1
# python train_ppo.py --RL_config MAPPO_episode500_epoch20_beta1e-1_seed666 --task 1
python train_mappo.py --RL_config MASAC_episode500_epoch10_beta1e-1_seed1234 --task 2