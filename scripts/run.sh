# generate code
for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
    do
	for size in 0.5 1.5 14 32
	    do
		python verification_scaling/generate_code.py --model Qwen/Qwen2.5-Coder-${size}B-Instruct --dataset_name ${benchmark} --dataset_split "test" --temperature 0 --num_generations 1 --num_parallel 100
	    done
    done

# SFT
# generate tests for evaluation
#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 3 7
#            do
#		python verification_scaling/generate_tests.py --model /share/rush/models/sft-${size}B --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only_no_few_shot" --temperature 0 --num_generations 1
#	    done
#    done
#
### run generated tests
###for benchmark in livecodebench humaneval mbpp
#for benchmark in humaneval
#    do
#	for size in 3 7
#	    do
#		python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_sft-${size}B_t0.0_n1_generated_tests --num_parallel 100
#	    done
#    done
#
### execute
###for benchmark in livecodebench humaneval mbpp
#for benchmark in humaneval
#    do
#	for size in 3 7
#            do
#		python verification_scaling/execute.py --dataset_name test-gen/${benchmark}_sft-${size}B_t0.0_n1_generated_tests --num_parallel 100 --dataset_split "test"
#	    done
#    done
#
## MINE
# generate tests for evaluation
#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 7
#            do
#	        for config in random easy unique easy-unique
#		    do
#		        python verification_scaling/generate_tests.py --model models/qwen-${size}b-${config} --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only_no_few_shot" --temperature 0 --num_generations 1
#		    done
#	    done
#    done
#
## run generated tests
#for benchmark in livecodebench humaneval mbpp
#    do
#	for size in 7
#	    do
#		for config in random easy unique easy-unique
#		    do
#			python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_qwen-${size}b-${config}_t0.0_n1_generated_tests --num_parallel 100
#		    done
#	    done
#    done
#
## execute
#for benchmark in livecodebench humaneval mbpp
#    do
#	for size in 7
#            do
#	        for config in random easy unique easy-unique
#		    do
#			python verification_scaling/execute.py --dataset_name test-gen/${benchmark}_qwen-${size}b-${config}_t0.0_n1_generated_tests --num_parallel 100 --dataset_split "test"
#		    done
#	    done
#    done

#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 0.6 1.7 4 8 14 32
#            do
#		python verification_scaling/generate_tests.py --model Qwen/Qwen3-${size}B --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only" --temperature 0.7 --num_generations 1 --top_p 0.8 --top_k 20 --min_p 0 --max_tokens 512
#	    done
#    done

#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 0.6 1.7 4 8 14 32
#            do
#		python verification_scaling/generate_tests.py --model Qwen/Qwen3-${size}B --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only" --temperature 0.6 --num_generations 1 --top_p 0.95 --top_k 20 --min_p 0 --max_tokens 16000 --thinking
#	    done
#    done

#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 0.5 1.6 3 7 14 32
#            do
#		python verification_scaling/generate_tests.py --model Qwen/Qwen2.5-Coder-${size}B-Instruct --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only" --temperature 0 --num_generations 1
#	    done
#    done

#for benchmark in livecodebench humaneval mbpp
#    do
#       for size in 0.5 1.5 3 7 14 32
#           do
#               python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-${size}B-Instruct_t0.0_n1_generated_tests --num_parallel 100
#           done
#    done

#for benchmark in livecodebench humaneval mbpp
#    do
#       for size in 0.6 1.7 4 8 14 32
#           do
#               python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen3-${size}B_t0.6_n1_think_generated_tests --num_parallel 100
#               python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_Qwen3-${size}B_t0.7_n1_generated_tests --num_parallel 100
#           done
#    done

# generate training data
#for size in 0.6 1.7 4 8
#    do
#        python verification_scaling/generate_tests.py --model models/sft-qwen3-${size}B --dataset_name test-gen/combined --dataset_split "validation" --test_prompt_format "instruction_only_no_few_shot" --temperature 1 --num_generations 8 --max_tokens 512
#        python verification_scaling/generate_tests.py --model models/sft-qwen3-${size}B --dataset_name test-gen/combined --dataset_split "train" --test_prompt_format "instruction_only_no_few_shot" --temperature 1 --num_generations 8 --max_tokens 512
#	python verification_scaling/execute.py --dataset_name test-gen/combined_sft-qwen3-${size}B_t1.0_n8_generated_tests --num_parallel 100 --dataset_split validation
#	python verification_scaling/execute.py --dataset_name test-gen/combined_sft-qwen3-${size}B_t1.0_n8_generated_tests --num_parallel 100 --dataset_split train
#    done
#
#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 0.6 1.7 4 8
#            do
#		python verification_scaling/generate_tests.py --model models/sft-qwen3-${size}B --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only_no_few_shot" --temperature 0 --num_generations 1 --max_tokens 512
#	    done
#    done
#for benchmark in livecodebench humaneval mbpp
#    do
#       for size in 0.6 1.7 4 8
#           do
#               python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_sft-qwen3-${size}B_t0.0_n1_generated_tests --num_parallel 100
#           done
#    done
#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 0.6 1.7 4 8
#            do
#		for config in random easy unique easy-unique
#		    do
#		        python verification_scaling/generate_tests.py --model models/qwen3-${size}b-${config}_lr1e-5 --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only_no_few_shot" --temperature 0 --num_generations 1 --max_tokens 512
#			new_benchmark=$(echo "$benchmark" | awk -F'/' '{print $2}' | awk -F'_' '{print $NF}')
#               	python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${new_benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${new_benchmark}_qwen3-${size}b-${config}_lr1e-5_t0.0_n1_generated_tests --num_parallel 100
#		    done
#	    done
#    done
#for benchmark in test-gen/livecodebench google-research-datasets/mbpp openai/openai_humaneval
#    do
#	for size in 1.5 7 14 32
#            do
#		python verification_scaling/generate_tests.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-${size}B --dataset_name ${benchmark} --dataset_split "test" --test_prompt_format "instruction_only" --temperature 0.6 --num_generations 1 --top_p 0.95 --top_k 20 --min_p 0 --max_tokens 16000 --thinking
#	    done
#    done
#for benchmark in livecodebench humaneval mbpp
#    do
#       for size in 1.5 7 14 32
#           do
#               python verification_scaling/run_generated_tests.py --generated_code_dataset_name test-gen/${benchmark}_Qwen2.5-Coder-3B-Instruct_t0.1_n8_generated_code --generated_tests_dataset_name test-gen/${benchmark}_DeepSeek-R1-Distill-Qwen-${size}B_t0.6_n1_think_generated_tests --num_parallel 100
#           done
#    done
