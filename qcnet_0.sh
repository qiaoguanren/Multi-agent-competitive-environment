task_name="train_qcnet_train"
launch_time=$(date +"%m-%d-%y-%H:%M:%S")
log_dir="log-server-${task_name}-${launch_time}.out"
export CUDA_VISIBLE_DEVICES=0,1,2,3
export CUDA_LAUNCH_BLOCKING=1
export TORCH_USE_CUDA_DSA=1
python train_qcnet_0.py --root /home/guanren/QCNet/datasets/ --train_batch_size 4 --val_batch_size 2 --test_batch_size 2 --devices 2 --dataset argoverse_v2 --num_historical_steps 50 --num_future_steps 60 --num_recurrent_steps 3 --pl2pl_radius 20 --time_span 10 --pl2a_radius 15 --a2a_radius 15 --num_t2m_steps 30 --pl2m_radius 20 --a2m_radius 20
#python test.py --model QCNet --root /home/guanren/QCNet/datasets/ --ckpt_path /home/guanren/QCNet/lightning_logs/version_11/checkpoints/epoch=63-step=399872.ckpt
#python val.py --model QCNet --root /home/guanren/QCNet/datasets/ --ckpt_path /home/guanren/QCNet/lightning_logs/version_11/checkpoints/epoch=63-step=399872.ckpt
#python val.py --model QCNet --root /home/guanren/QCNet/datasets/ --ckpt_path /home/guanren/QCNet/lightning_logs/version_17/checkpoints/epoch=63-step=1599232.ckpt
#python test.py --model QCNet --root /home/guanren/QCNet/datasets/ --ckpt_path /home/guanren/QCNet/lightning_logs/version_17/checkpoints/epoch=63-step=1599232.ckpt