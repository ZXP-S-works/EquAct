tasks=(
    insert_onto_square_peg
    )
data_dir=/media/zxp/large/project_data/SE3_bi_equ_data/peract/raw/test/
num_episodes=25
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
interpolation_length=2
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="left_shoulder,right_shoulder,wrist,front"
fps_subsampling_factor=5
lang_enhanced=0
relative_action=0
seed=0
#checkpoint=/home/zxp/projects/3d_diffuser_actor/train_logs/Actor_18Peract_100Demo_multitask/diffusion_multitask-peg-original/last.pth
quaternion_format=xyzw  # for local training
quaternion_format=wxyz  # for pretrianed weight
export PYTHONPATH="$PWD:$PYTHONPATH"
num_ckpts=${#tasks[@]}
exp=3DDA
#for ckp in 5000 10000 15000 20000 25000 30000 35000 40000 45000 50000 55000 60000; do
for ckp in 0; do
#checkpoint=train_logs/$exp/diffusion_multitask_peg_ori_60k/${ckp}.pth
checkpoint=train_logs/Actor_18Peract_100Demo_multitask/diffusion_multitask-C120-B8-lr1e-4-DI1-2-H3-DT100/diffuser_actor_peract.pth
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --diffusion_timesteps 100 \
    --fps_subsampling_factor $fps_subsampling_factor \
    --lang_enhanced $lang_enhanced \
    --relative_action $relative_action \
    --num_history 3 \
    --test_model 3d_diffuser_actor \
    --cameras $cameras \
    --verbose $verbose \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 1 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "6D" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}${ckp}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..199} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04 \
    --quaternion_format $quaternion_format \
    --interpolation_length $interpolation_length \
    --dense_interpolation 1
done
done
