import argparse
import ast
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from verification_scaling.utils import (
    code_reward,
    prepare_mbpp_prompt,
    prepare_humaneval_prompt,
    prepare_livecodebench_prompt
)


code_generation_format = '''
Solve the following coding problem using the programming language python:

{problem}

Now solve the problem and return the code.
'''

def generate_code(problems, model, temperature, num_generations, thinking):
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
    outputs = llm.chat(messages, sampling_params=sampling_params, chat_template_kwargs={"enable_thinking": thinking})
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
    parser.add_argument("--thinking", action="store_true", help="Enable thinking for code generation")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    if "livecodebench" in args.dataset_name:
        problems = [prepare_livecodebench_prompt(example) for example in dataset]
        dataset_name = "livecodebench"
    elif "mbpp" in args.dataset_name:
        problems = [prepare_mbpp_prompt(example) for example in dataset]
        dataset_name = "mbpp"
    elif "humaneval" in args.dataset_name.lower():
        problems = [prepare_humaneval_prompt(example) for example in dataset]
        dataset_name = "humaneval"
    else:
        raise NotImplementedError("Dataset not supported")
    generated_code = generate_code(
        problems,
        args.model,
        temperature=args.temperature,
        num_generations=args.num_generations,
        thinking=args.thinking
    )
    dataset = dataset.add_column(name="generated_code", column=generated_code)
    generated_code = dataset["generated_code"]
    test_completions = [{"content": sample} for one in generated_code for sample in one]

    gt_rewards_kwargs = dict()
    gt_rewards_kwargs["verification_info"] = []
    for example in dataset:
        if "humaneval" in args.dataset_name.lower():
            gt_tests = [test.strip() for test in example["test"].split("\n") if test.strip().startswith("assert")]
            gt_tests = [test.replace("candidate(", example["entry_point"].strip()+"(") for test in gt_tests]
        elif "livecodebench" in args.dataset_name.lower():
            function_call = example["function_name"]
            gt_tests = []
            parsed_tests = ast.literal_eval(example["test"])
            starter_code = example["starter_code"]
            start = starter_code.find("(")
            end = starter_code.rfind(")")
            extracted_args = starter_code[start+1:end]
            num_args = extracted_args.count(":")
            assert num_args > 0, "No arguments found in starter code"
            for one in parsed_tests:
                if num_args == 1:
                    input_args = one["input"]
                else:
                    if '\n' in one["input"]:
                        input_args = one["input"].split("\n")
                        input_args = [arg.strip() for arg in input_args]
                    else:
                        input_args = ast.literal_eval(one["input"])
                    assert len(input_args) == num_args, "Number of arguments in input does not match number of arguments in starter code"
                    input_args = [str(arg) for arg in input_args]
                    input_args = ", ".join(input_args)
                gt_test = f"assert {function_call}({input_args}) == {one['output']}"
                gt_tests.append(gt_test)
        else:
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

    model_name = args.model.split("/")[-1]
    output_dataset_name = f"test-gen/{dataset_name}_{model_name}_t{args.temperature}_n{args.num_generations}_generated_code"
    dataset.push_to_hub(output_dataset_name, split=args.dataset_split)
