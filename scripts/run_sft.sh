set -x

# SFT
#sizes=("0.5" "1.5" "3" "7")
#sizes=("0.6" "1.7" "4" "8")
#
## Loop through each size
#for size in "${sizes[@]}"; do
#    echo "Running training with model: Qwen/Qwen3-${size}B"
#    
#    # Run the torchrun command with the current model
#    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#        -m verl.trainer.fsdp_sft_trainer \
#        data.train_files=$HOME/data/combined_sft \
#        data.val_files=$HOME/data/combined_sft \
#        data.prompt_key=extra_info \
#        data.response_key=extra_info \
#        data.prompt_dict_keys=['question'] \
#        data.response_dict_keys=['answer'] \
#        data.micro_batch_size_per_gpu=1 \
#        data.truncation=right \
#        model.partial_pretrain=Qwen/Qwen3-${size}B \
#        trainer.project_name=mbpp-test-gen-sft \
#        trainer.experiment_name=sft-qwen3-${size}B \
#        trainer.total_epochs=1 \
#        trainer.logger=['wandb'] \
#        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/sft-qwen3-${size}B \
#        optim.lr=1e-5
#    
#    echo "Completed training with model: Qwen/Qwen3-${size}B"
#    echo "-----------------------------------------"
#done

sizes=("0.5" "1.5" "3" "7")

# Loop through each size
for size in "${sizes[@]}"; do
    # Run the torchrun command with the current model
    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$HOME/data/combined_dagger_${size}b/train.parquet \
        data.val_files=$HOME/data/combined_dagger_${size}b/test.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.prompt_dict_keys=['question'] \
        data.response_dict_keys=['answer'] \
        data.micro_batch_size_per_gpu=1 \
        data.truncation=right \
        model.partial_pretrain=/share/rush/models/sft-${size}B \
        trainer.project_name=mbpp-test-gen-sft \
        trainer.experiment_name=qwen2-${size}b-random-lr1e-6 \
        trainer.total_epochs=1 \
        trainer.logger=['wandb'] \
        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen2-${size}b-random_lr1e-6 \
        optim.lr=1e-6

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$HOME/data/combined_dagger_${size}b_easy/train.parquet \
        data.val_files=$HOME/data/combined_dagger_${size}b_easy/test.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.prompt_dict_keys=['question'] \
        data.response_dict_keys=['answer'] \
        data.micro_batch_size_per_gpu=1 \
        data.truncation=right \
        model.partial_pretrain=/share/rush/models/sft-${size}B \
        trainer.project_name=mbpp-test-gen-sft \
        trainer.experiment_name=qwen2-${size}b-easy-lr1e-6 \
        trainer.total_epochs=1 \
        trainer.logger=['wandb'] \
        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen2-${size}b-easy_lr1e-6 \
        optim.lr=1e-6

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$HOME/data/combined_dagger_${size}b_unique/train.parquet \
        data.val_files=$HOME/data/combined_dagger_${size}b_unique/test.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.prompt_dict_keys=['question'] \
        data.response_dict_keys=['answer'] \
        data.micro_batch_size_per_gpu=1 \
        data.truncation=right \
        model.partial_pretrain=/share/rush/models/sft-${size}B \
        trainer.project_name=mbpp-test-gen-sft \
        trainer.experiment_name=qwen2-${size}b-unique-lr1e-6 \
        trainer.total_epochs=1 \
        trainer.logger=['wandb'] \
        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen2-${size}b-unique_lr1e-6 \
        optim.lr=1e-6

    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
        -m verl.trainer.fsdp_sft_trainer \
        data.train_files=$HOME/data/combined_dagger_${size}b_easy_unique/train.parquet \
        data.val_files=$HOME/data/combined_dagger_${size}b_easy_unique/test.parquet \
        data.prompt_key=extra_info \
        data.response_key=extra_info \
        data.prompt_dict_keys=['question'] \
        data.response_dict_keys=['answer'] \
        data.micro_batch_size_per_gpu=1 \
        data.truncation=right \
        model.partial_pretrain=/share/rush/models/sft-${size}B \
        trainer.project_name=mbpp-test-gen-sft \
        trainer.experiment_name=qwen2-${size}b-easy-unique-lr1e-6 \
        trainer.total_epochs=1 \
        trainer.logger=['wandb'] \
        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen2-${size}b-easy-unique_lr1e-6 \
        optim.lr=1e-6
 
    echo "-----------------------------------------"
done

## Dagger iter0
#sizes=("0.5" "1.5" "3" "7")
#sizes=("0.6" "1.7" "4" "8")
#
#
## Loop through each size
#for size in "${sizes[@]}"; do
#    # Run the torchrun command with the current model
#    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#        -m verl.trainer.fsdp_sft_trainer \
#        data.train_files=$HOME/data/combined_qwen3_${size}b/train.parquet \
#        data.val_files=$HOME/data/combined_qwen3_${size}b/test.parquet \
#        data.prompt_key=extra_info \
#        data.response_key=extra_info \
#        data.prompt_dict_keys=['question'] \
#        data.response_dict_keys=['answer'] \
#        data.micro_batch_size_per_gpu=1 \
#        data.truncation=right \
#        model.partial_pretrain=models/sft-qwen3-${size}B \
#        trainer.project_name=mbpp-test-gen-sft \
#        trainer.experiment_name=qwen3-${size}b-random \
#        trainer.total_epochs=1 \
#        trainer.logger=['wandb'] \
#        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen3-${size}b-random_lr1e-5 \
#        optim.lr=1e-5
#
#    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#        -m verl.trainer.fsdp_sft_trainer \
#        data.train_files=$HOME/data/combined_qwen3_${size}b_easy/train.parquet \
#        data.val_files=$HOME/data/combined_qwen3_${size}b_easy/test.parquet \
#        data.prompt_key=extra_info \
#        data.response_key=extra_info \
#        data.prompt_dict_keys=['question'] \
#        data.response_dict_keys=['answer'] \
#        data.micro_batch_size_per_gpu=1 \
#        data.truncation=right \
#        model.partial_pretrain=models/sft-qwen3-${size}B \
#        trainer.project_name=mbpp-test-gen-sft \
#        trainer.experiment_name=qwen3-${size}b-easy \
#        trainer.total_epochs=1 \
#        trainer.logger=['wandb'] \
#        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen3-${size}b-easy_lr1e-5 \
#        optim.lr=1e-5
#
#    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#        -m verl.trainer.fsdp_sft_trainer \
#        data.train_files=$HOME/data/combined_qwen3_${size}b_unique/train.parquet \
#        data.val_files=$HOME/data/combined_qwen3_${size}b_unique/test.parquet \
#        data.prompt_key=extra_info \
#        data.response_key=extra_info \
#        data.prompt_dict_keys=['question'] \
#        data.response_dict_keys=['answer'] \
#        data.micro_batch_size_per_gpu=1 \
#        data.truncation=right \
#        model.partial_pretrain=models/sft-qwen3-${size}B \
#        trainer.project_name=mbpp-test-gen-sft \
#        trainer.experiment_name=qwen3-${size}b-unique \
#        trainer.total_epochs=1 \
#        trainer.logger=['wandb'] \
#        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen3-${size}b-unique_lr1e-5 \
#        optim.lr=1e-5
#
#    torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#        -m verl.trainer.fsdp_sft_trainer \
#        data.train_files=$HOME/data/combined_qwen3_${size}b_easy_unique/train.parquet \
#        data.val_files=$HOME/data/combined_qwen3_${size}b_easy_unique/test.parquet \
#        data.prompt_key=extra_info \
#        data.response_key=extra_info \
#        data.prompt_dict_keys=['question'] \
#        data.response_dict_keys=['answer'] \
#        data.micro_batch_size_per_gpu=1 \
#        data.truncation=right \
#        model.partial_pretrain=models/sft-qwen3-${size}B \
#        trainer.project_name=mbpp-test-gen-sft \
#        trainer.experiment_name=qwen3-${size}b-easy-unique \
#        trainer.total_epochs=1 \
#        trainer.logger=['wandb'] \
#        trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/qwen3-${size}b-easy-unique_lr1e-5 \
#        optim.lr=1e-5
#    
#    echo "-----------------------------------------"
#done

#torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#    -m verl.trainer.fsdp_sft_trainer \
#    data.train_files=$HOME/data/merged_train.parquet \
#    data.val_files=$HOME/data/merged_train.parquet \
#    data.prompt_key=extra_info \
#    data.response_key=extra_info \
#    data.prompt_dict_keys=['question'] \
#    data.response_dict_keys=['answer'] \
#    data.micro_batch_size_per_gpu=1 \
#    data.truncation=right \
#    model.partial_pretrain=models/mbpp-dagger-easy-qwen-coder-7b-instruct-from-sft \
#    trainer.project_name=mbpp-test-gen-sft \
#    trainer.experiment_name=mbpp-dagger-easy-qwen-coder-7b-instruct-from-sft-iter1 \
#    trainer.total_epochs=10 \
#    trainer.logger=['wandb'] \
#    trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/mbpp-dagger-easy-qwen-coder-7b-instruct-from-sft-iter1 \
#    optim.lr=1e-6

#torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#    -m verl.trainer.fsdp_sft_trainer \
#    data.train_files=$HOME/data/mbpp_dagger_7b/train.parquet \
#    data.val_files=$HOME/data/mbpp_dagger_7b/test.parquet \
#    data.prompt_key=extra_info \
#    data.response_key=extra_info \
#    data.prompt_dict_keys=['question'] \
#    data.response_dict_keys=['answer'] \
#    data.micro_batch_size_per_gpu=1 \
#    data.truncation=right \
#    model.partial_pretrain=models/mbpp-qwen-coder-7b-instruct \
#    trainer.project_name=mbpp-test-gen-sft \
#    trainer.experiment_name=mbpp-dagger-qwen-coder-7b-instruct-from-sft \
#    trainer.total_epochs=10 \
#    trainer.logger=['wandb'] \
#    trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/mbpp-dagger-qwen-coder-7b-instruct-from-sft
#torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#    -m verl.trainer.fsdp_sft_trainer \
#    data.train_files=$HOME/data/mbpp_dagger_7b_bucket/train.parquet \
#    data.val_files=$HOME/data/mbpp_dagger_7b_bucket/test.parquet \
#    data.prompt_key=extra_info \
#    data.response_key=extra_info \
#    data.prompt_dict_keys=['question'] \
#    data.response_dict_keys=['answer'] \
#    data.micro_batch_size_per_gpu=1 \
#    data.truncation=right \
#    model.partial_pretrain=models/mbpp-qwen-coder-7b-instruct \
#    trainer.project_name=mbpp-test-gen-sft \
#    trainer.experiment_name=mbpp-dagger-easy-bucket-qwen-coder-7b-instruct-from-sft \
#    trainer.total_epochs=10 \
#    trainer.logger=['wandb'] \
#    trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/mbpp-dagger-easy-bucket-qwen-coder-7b-instruct-from-sft

#torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#    -m verl.trainer.fsdp_sft_trainer \
#    data.train_files=$HOME/data/mbpp/train.parquet \
#    data.val_files=$HOME/data/mbpp/test.parquet \
#    data.prompt_key=extra_info \
#    data.response_key=extra_info \
#    data.prompt_dict_keys=['question'] \
#    data.response_dict_keys=['answer'] \
#    data.micro_batch_size_per_gpu=1 \
#    data.truncation=right \
#    model.partial_pretrain=models/mbpp-qwen-coder-7b-instruct \
#    trainer.project_name=mbpp-test-gen-sft \
#    trainer.experiment_name=mbpp-qwen-coder-7b-instruct-from-sft \
#    trainer.total_epochs=10 \
#    trainer.logger=['wandb'] \
#    trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/mbpp-qwen-coder-7b-instruct-from-sft

#torchrun --standalone --nnodes=1 --nproc_per_node=2 \
#    -m verl.trainer.fsdp_sft_trainer \
#    data.train_files=$HOME/data/mbpp/train.parquet \
#    data.val_files=$HOME/data/mbpp/test.parquet \
#    data.prompt_key=extra_info \
#    data.response_key=extra_info \
#    data.prompt_dict_keys=['question'] \
#    data.response_dict_keys=['answer'] \
#    data.micro_batch_size_per_gpu=1 \
#    data.truncation=right \
#    model.partial_pretrain=Qwen/Qwen2.5-Coder-7B-Instruct \
#    trainer.project_name=mbpp-test-gen-sft \
#    trainer.experiment_name=mbpp-qwen-coder-7b-instruct-from-sft \
#    trainer.total_epochs=10 \
#    trainer.logger=['wandb'] \
#    trainer.default_local_dir=checkpoints/mbpp-test-gen-sft/mbpp-qwen-coder-7b-instruct-from-sft \
