import argparse
from datasets import load_dataset
from open_r1.rewards import code_reward
from code_generation import generate_code
from test_case_generation import generate_tests


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generation_model", type=str, required=True, help="Model for code generation")
    parser.add_argument("--verification_model", type=str, required=True, help="Model for test generation")
    parser.add_argument("--dataset_name", type=str, default="livecodebench", help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--num_generations", type=int, default=10, help="Number of generations per example")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--code_prompt_format", type=str, default="std", help="Prompt format for code generation")
    parser.add_argument("--test_prompt_format", type=str, default="instruction_only", help="Prompt format for test generation")
    parser.add_argument("--output_dataset_name", type=str, default="wentingzhao/livecodebench_generated", help="Output dataset name")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    if "livecodebench" in args.dataset_name:
        problems = [example['question_content'] for example in dataset]
        problems = [problem.split("Sample Input 1")[0].split("Example 1")[0].strip() for problem in problems]
    elif "mbpp" in args.dataset_name:
        problems = [example['text'] for example in dataset]
    else:
        raise NotImplementedError("Dataset not supported")
    generated_code = generate_code(
        problems,
        args.code_prompt_format,
        args.generation_model,
        temperature=args.temperature,
        num_generations=args.num_generations
    )
    dataset = dataset.add_column(name="generated_code", column=generated_code)
    test_inputs, test_outputs = generate_tests(problems, args.test_prompt_format, args.verification_model)
    verification_info = []
    for test_input, test_output in zip(test_inputs, test_outputs):
        verification_info.append({
            "language": "python",
            "test_cases": [
                {
                    "input": "\n".join(test_input),
                    "output": "\n".join(test_output),
                    "type": "stdin_stdout",
                }
            ],
        })
    dataset = dataset.add_column(name="verification_info", column=verification_info)
    test_completions = []
    reward_kwargs = dict()
    reward_kwargs["verification_info"] = []
    for example in dataset:
        test_completions += [[{"content": one}] for one in example["generated_code"]]
        reward_kwargs["verification_info"] += [example["verification_info"] for _ in example["generated_code"]]
    rewards = code_reward(test_completions, num_parallel=256, **reward_kwargs)
    rewards = [rewards[i:i+len(example["generated_code"])] for i in range(0, len(rewards), len(example["generated_code"]))]
    dataset = dataset.add_column(name="rewards", column=rewards)
    dataset.push_to_hub(args.output_dataset_name, split=args.dataset_split)


if __name__ == "__main__":
    main()