tasks=(place_wine_at_rack_location)
data_dir=./data/peract/raw/test/
num_episodes=10
gripper_loc_eract_tasks_location_bounds.json
use_instruction=1
max_tries=2
verbose=1
single_task_g=120
cameras="left_ATH="$PWD:$PYTHONPATH"
num_ckpts=${#tasks[@]}
exp=equact
checkpoint=train_lA_VISIBLE_DEVICES=0 python online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --num_history 1 \
    --test_model act3d \
    --cameras $cameras \
    --verbose $verbose \
    --num_ghost_points_val 10000 \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 0 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "quat_from_query" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..60} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04
done

