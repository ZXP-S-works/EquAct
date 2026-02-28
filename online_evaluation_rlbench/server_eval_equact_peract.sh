
tasks=(
    close_jar insert_onto_square_peg light_bulb_in meat_off_grill open_drawer place_shape_in_shape_sorter place_wine_at_rack_location push_buttons put_groceries_in_cupboard put_item_in_drawer put_money_in_safe
    reach_and_drag slide_block_to_color_target stack_blocks stack_cups sweep_to_dustpan_of_size turn_tap place_cups
)
data_dir=./data/peract/raw/test/
num_episodes=25
gripper_loc_bounds_file=tasks/18_peract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
single_task_gripper_loc_bounds=0
embedding_dim=120
cameras="left_shoulder,right_shoulder,wrist,front"
seed=0
export PYTHONPATH="$PWD:$PYTHONPATH"
num_ckpts=${#tasks[@]}
exp=equ_act
folder=0515_train_equact
for num_checkpoint in 100000 200000 300000 400000 500000 600000 700000 800000;
do
checkpoint=train_logs/${folder}/${exp}/${num_checkpoint}.pth
for ((i=0; i<$num_ckpts; i++)); do
    CUDA_LAUNCH_BLOCKING=1 CUDA_VISIBLE_DEVICES=1 python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --num_history 1 \
    --test_model equact \
    --cameras $cameras \
    --verbose $verbose \
    --num_ghost_points_val 3000 \
    --num_sampling_level 3 \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 0 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "quat_from_query" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/${folder}/${exp}/${num_checkpoint}/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..200} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04
done
done
