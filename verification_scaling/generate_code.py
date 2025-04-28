import argparse
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from verification_scaling.utils import (
    code_reward,
    prepare_mbpp_prompt
)


code_generation_format = '''
Solve the following coding problem using the programming language python:

{problem}

Now solve the problem and return the code.
'''

def generate_code(problems, model, temperature, num_generations):
    """Generate code solutions for a dataset using vLLM.
    
    Args:
        problems: List of problems
        model: vLLM model name or path
        temperature: Sampling temperature
        num_generations: Number of generations per example
        
    Returns:
        List of lists containing generated code solutions for each example
    """
    llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        n=num_generations
    )
    code_prompts = [code_generation_format.format(problem=problem) for problem in problems]

    messages = [[{"role": "user", "content": prompt}] for prompt in code_prompts]
    outputs = llm.chat(messages, sampling_params)
    generated_code = [[one.text for one in output.outputs] for output in outputs]
    return generated_code


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model for code generation")
    parser.add_argument("--dataset_name", type=str, default="livecodebench", help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of generations per example")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--num_parallel", type=int, default=20, help="Number of parallel generations to run")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    if "livecodebench" in args.dataset_name:
        problems = [example['question_content'] for example in dataset]
        problems = [problem.split("Sample Input 1")[0].split("Example 1")[0].strip() for problem in problems]
        dataset_name = "livecodebench"
    elif "mbpp" in args.dataset_name:
        problems = [prepare_mbpp_prompt(example) for example in dataset]
        dataset_name = "mbpp"
    else:
        raise NotImplementedError("Dataset not supported")
    generated_code = generate_code(
        problems,
        args.model,
        temperature=args.temperature,
        num_generations=args.num_generations
    )
    dataset = dataset.add_column(name="generated_code", column=generated_code)
    generated_code = dataset["generated_code"]
    test_completions = [{"content": sample} for one in generated_code for sample in one]

    gt_rewards_kwargs = dict()
    gt_rewards_kwargs["verification_info"] = []
    for example in dataset:
        gt_tests = example["test_list"] + example["challenge_test_list"]
        gt_tests = {
            "language": "python",
            "test_cases": gt_tests,
        }
        gt_rewards_kwargs["verification_info"] += [gt_tests for _ in example["generated_code"]]
    gt_rewards = code_reward(test_completions, num_parallel=args.num_parallel, **gt_rewards_kwargs)
    gt_rewards = [gt_rewards[i:i+args.num_generations] for i in range(0, len(gt_rewards), args.num_generations)]
    dataset = dataset.add_column(name="gt_rewards", column=gt_rewards)
    num_passes = sum(1 for one in gt_rewards if sum(one) > 0)
    print(f"pass@{args.num_generations}: {num_passes / len(gt_rewards)}")

    execution_rewards_kwargs = dict()
    execution_rewards_kwargs["verification_info"] = []
    for example in dataset:
        gt_tests = example["test_list"] + example["challenge_test_list"]
        gt_tests = {
            "language": "python",
            "test_cases": [],
        }
        execution_rewards_kwargs["verification_info"] += [gt_tests for _ in example["generated_code"]]
    execution_rewards = code_reward(test_completions, num_parallel=args.num_parallel, **execution_rewards_kwargs)
    execution_rewards = [execution_rewards[i:i+args.num_generations] for i in range(0, len(execution_rewards), args.num_generations)]
    dataset = dataset.add_column(name="execution_rewards", column=execution_rewards)

    model_name = args.model.split("/")[-1]
    output_dataset_name = f"wentingzhao/{dataset_name}_{model_name}_temp{args.temperature}_num{args.num_generations}_generated_code"
    dataset.push_to_hub(output_dataset_name, split=args.dataset_split)