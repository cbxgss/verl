export VLLM_ATTENTION_BACKEND=XFORMERS
export PET_NODE_RANK=0

export project_name="verl"

set_variable_with_default() {
    local var_name=$1
    shift
    local guidance=$1
    shift
    local default_values=("$@")
    local num_options=${#default_values[@]}

    # 打印所有选项
    for ((i = 0; i < num_options; i++)); do
        echo "$((i + 1)). ${default_values[$i]}"
    done

    read -p "input $guidance (input 1-$num_options to choose a default value): " user_input

    if [[ $user_input =~ ^[1-$num_options]$ ]]; then
        local selected_index=$((user_input - 1))
        export "$var_name=${default_values[$selected_index]}"
    else
        export "$var_name=$user_input"
    fi

    echo "Selected $guidance: ${var_name} = ${!var_name}"
    echo ""
}

set_variable_with_default CUDA_VISIBLE_DEVICES device 0,1 2,3 4,5 6,7 0,1,2,3 4,5,6,7 0,1,2,3,4,5,6,7
gpu_nums=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

set_variable_with_default debug debug true false

set_variable_with_default model model Qwen/Qwen2.5-0.5B-Instruct Qwen/Qwen3-0.6B Qwen/Qwen3-1.7B
if [[ $model == *"0.5"* ]]; then
    model_name=qwen2.5-0.5
elif [[ $model == *"0.6"* ]]; then
    model_name=qwen3-0.6
elif [[ $model == *"1.7"* ]]; then
    model_name=qwen3-1.7
else
    echo "Invalid model name"
    exit 1
fi

set_variable_with_default n grpo_n_agent 5 16

export experiment_name=${model_name}-$(date +%m.%d-%H:%M:%S)-n_${n}-$(echo "$CUDA_VISIBLE_DEVICES" | tr -d ',')

mkdir -p tmp/logs/$experiment_name
PYTHONUNBUFFERED=1 python3 -m verl.trainer.main_ppo \
    data.train_batch_size=256 \
    data.max_prompt_length=30767 \
    data.max_response_length=2000 \
    actor_rollout_ref.model.path=${model} \
    actor_rollout_ref.model.enable_gradient_checkpointing=true \
    actor_rollout_ref.model.use_remove_padding=true \
    actor_rollout_ref.rollout.name=sglang \
    actor_rollout_ref.rollout.mode=sync \
    actor_rollout_ref.rollout.max_model_len=32768 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.3 \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.max_num_batched_tokens=32768 \
    actor_rollout_ref.rollout.n=${n} \
    actor_rollout_ref.rollout.multi_turn.max_turns=4 \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps_ratio=0.95 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.actor.use_dynamic_bsz=true \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=32768 \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=true \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=true \
    actor_rollout_ref.actor.use_kl_loss=true \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=true \
    +critic.model.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.actor.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.ref.fsdp_config.forward_prefetch=True \
    +actor_rollout_ref.actor.entropy_from_logits_with_chunking=True \
    +actor_rollout_ref.ref.entropy_from_logits_with_chunking=True \
    critic.optim.lr=1e-5 \
    critic.model.path=${model} \
    critic.ppo_micro_batch_size_per_gpu=8 \
    algorithm.kl_ctrl.kl_coef=0.001 \
    ray_init.debug=${debug} \
    trainer.project_name=${project_name} \
    trainer.experiment_name=${experiment_name} \
    trainer.val_before_train=false \
    trainer.default_hdfs_dir=null \
    trainer.n_gpus_per_node=$gpu_nums \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=10 \
    trainer.total_epochs=15 2>&1 | tee tmp/logs/$experiment_name/log.log
