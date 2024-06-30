tashon online_evaluation_rlbench/evaluate_policy.py \
    --tasks ${tasks[$i]} \
    --checkpoint $checkpoint \
    --num_history 1 \
    --test_model act3d \
    --cameras $cameras \
    --verbose $verbose \
    --num_ghost_points_val 6000 \
    --action_dim 8 \
    --collision_checking 0 \
    --predict_trajectory 0 \
    --embedding_dim $embedding_dim \
    --rotation_parametrization "quat_from_query" \
    --single_task_gripper_loc_bounds $single_task_gripper_loc_bounds \
    --data_dir $data_dir \
    --num_episodes $num_episodes \
    --output_file eval_logs/$exp/seed$seed/${tasks[$i]}${num_checkpoint}.json  \
    --use_instruction $use_instruction \
    --instructions instructions/peract/instructions.pkl \
    --variations {0..200} \
    --max_tries $max_tries \
    --max_steps 25 \
    --seed $seed \
    --gripper_loc_bounds_file $gripper_loc_bounds_file \
    --gripper_loc_bounds_buffer 0.04
done

