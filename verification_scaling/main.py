import argparse
from datasets import load_dataset
from verification_scaling.utils import code_reward, prepare_mbpp_prompt
from verification_scaling.code_generation import generate_code
from verification_scaling.test_case_generation import generate_tests

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model", type=str, required=True, help="Model for code generation")
    parser.add_argument("--verification_model", type=str, required=True, help="Model for test generation")
    parser.add_argument("--dataset_name", type=str, default="livecodebench", help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of generations per example")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--test_prompt_format", type=str, default="instruction_only", help="Prompt format for test generation")
    parser.add_argument("--output_dataset_name", type=str, default="wentingzhao/livecodebench_generated", help="Output dataset name")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True).select(range(10))
    if "livecodebench" in args.dataset_name:
        problems = [example['question_content'] for example in dataset]
        problems = [problem.split("Sample Input 1")[0].split("Example 1")[0].strip() for problem in problems]
    elif "mbpp" in args.dataset_name:
        problems = [prepare_mbpp_prompt(example) for example in dataset]
    else:
        raise NotImplementedError("Dataset not supported")
    generated_code = generate_code(
        problems,
        args.generation_model,
        temperature=args.temperature,
        num_generations=args.num_generations
    )
    dataset = dataset.add_column(name="generated_code", column=generated_code)
    test_completions = [{"content": sample} for one in generated_code for sample in one]
    tests = generate_tests(problems, args.test_prompt_format, args.verification_model)
    verification_info = []
    for test in tests:
        verification_info.append({
            "language": "python",
            "test_cases": test
        })
    dataset = dataset.add_column(name="verification_info", column=verification_info)
    reward_kwargs = dict()
    reward_kwargs["verification_info"] = []
    for example in dataset:
        reward_kwargs["verification_info"] += [example["verification_info"] for _ in example["generated_code"]]
    rewards = code_reward(test_completions, num_parallel=256, **reward_kwargs)
    rewards = [rewards[i:i+len(example["generated_code"])] for i in range(0, len(rewards), len(example["generated_code"]))]
    dataset = dataset.add_column(name="rewards", column=rewards)
    
    gt_rewards_kwargs = dict()
    gt_rewards_kwargs["verification_info"] = []
    for example in dataset:
        gt_tests = example["test_list"] + example["challenge_test_list"]
        gt_tests = {
            "language": "python",
            "test_cases": gt_tests,
        }
        gt_rewards_kwargs["verification_info"] += [gt_tests for _ in example["generated_code"]]
    gt_rewards = code_reward(test_completions, num_parallel=256, **gt_rewards_kwargs)
    gt_rewards = [gt_rewards[i:i+len(example["generated_code"])] for i in range(0, len(gt_rewards), len(example["generated_code"]))]
    dataset = dataset.add_column(name="gt_rewards", column=gt_rewards)
    dataset.push_to_hub(args.output_dataset_name, split=args.dataset_split)



if __name__ == "__main__":
    main()