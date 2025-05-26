temp=1.0
code_size=7

# distilled
for benchmark in livecodebench humaneval mbpp
    do
        for size in 1.5 7 14 32
            do
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_DeepSeek-R1-Distill-Qwen-${size}B_t0.6_n1_think_generated_tests --num_parallel 100
           done
    done

# oai models
for benchmark in livecodebench humaneval mbpp
    do
	for model in o3 o4-mini
	    do
	        python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_${model}_t0_n1_generated_tests --num_parallel 100
	    done
    done

# number of tests ablation
for benchmark in livecodebench mbpp humaneval
    do
	for num in 1 5 10
            do
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen-7b-easy_t0.0_n1_generated_tests --num_parallel 100 --num_tests $num
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen3-4B_t0.6_n1_think_generated_tests --num_parallel 100 --num_tests $num
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_o3_t0_n1_generated_tests --num_parallel 100 --num_tests $num
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_o4-mini_t0_n1_generated_tests --num_parallel 100 --num_tests $num
	    done
    done

# check for different codegen sizes
for benchmark in mbpp humaneval livecodebench
    do
        for size in 0.5 1.5 3 14 32
	    do
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_o3_t0_n1_generated_tests --num_parallel 100
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_o4-mini_t0_n1_generated_tests --num_parallel 100
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen3-4B_t0.6_n1_think_generated_tests --num_parallel 100
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen-7b-easy_t0.0_n1_generated_tests --num_parallel 100
	    done
    done

# few-shot qwen2
for benchmark in livecodebench humaneval mbpp
    do
        for size in 0.5 1.5 3 7 14 32
            do
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${size}B-Instruct_t0.0_n1_generated_tests --num_parallel 100
            done
    done

# qwen2 sft
for benchmark in livecodebench humaneval mbpp
    do
	for size in 0.5 1.5 3 7
	    do
		python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_sft-${size}B_t0.0_n1_generated_tests --num_parallel 100
	    done
    done

# qwen2 sample and repair
for benchmark in livecodebench humaneval mbpp
    do
	for size in 0.5 1.5 3 7
	    do
		for config in random easy unique easy-unique
		    do
			python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen2-${size}b-${config}_lr1e-6_t0.0_n1_generated_tests --num_parallel 100
			python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen-${size}b-${config}_t0.0_n1_generated_tests --num_parallel 100
		    done
	    done
    done

# fewshot qwen 3 / think and non-think
for benchmark in livecodebench humaneval mbpp
    do
        for size in 0.6 1.7 4 8 14 32
            do
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen3-${size}B_t0.6_n1_think_generated_tests --num_parallel 100
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen3-${size}B_t0.7_n1_generated_tests --num_parallel 100
            done
    done

# qwen3 sft
for benchmark in livecodebench humaneval mbpp
    do
        for size in 0.6 1.7 4 8
            do
                python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_sft-qwen3-${size}B_t0.0_n1_generated_tests --num_parallel 100
            done
    done

# qwen3 sample and repair
for benchmark in livecodebench mbpp humaneval
    do
	for size in 0.6 1.7 4 8
            do
		for config in random easy unique easy-unique
		    do
               	        python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen3-${size}b-${config}_t0.0_n1_generated_tests --num_parallel 100
               	        python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${code_size}B-Instruct_t${temp}_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen3-${size}b-${config}_lr1e-5_t0.0_n1_generated_tests --num_parallel 100
		    done
	    done
    done
