import argparse
from datasets import load_dataset
from verification_scaling.utils import code_reward

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--generated_code_dataset_name", type=str, required=True, help="Dataset name for generated code")
    parser.add_argument("--generated_code_dataset_split", type=str, default="test", help="Dataset split for generated code")
    parser.add_argument("--generated_tests_dataset_name", type=str, required=True, help="Dataset name for generated tests")
    parser.add_argument("--generated_tests_dataset_split", type=str, default="test", help="Dataset split for generated tests")
    parser.add_argument("--num_parallel", type=int, default=20, help="Number of parallel tests to run")
    return parser.parse_args()


def main():
    args = parse_args()
    generated_code_dataset = load_dataset(args.generated_code_dataset_name, split=args.generated_code_dataset_split, trust_remote_code=True)
    generated_tests_dataset = load_dataset(args.generated_tests_dataset_name, split=args.generated_tests_dataset_split, trust_remote_code=True)
    generated_code = generated_code_dataset["generated_code"]
    generated_tests = generated_tests_dataset["verification_info"]
    test_completions = [{"content": sample} for one in generated_code for sample in one]

    reward_kwargs = dict()
    reward_kwargs["verification_info"] = []
    for code, tests in zip(generated_code, generated_tests):
        reward_kwargs["verification_info"] += [tests for _ in code]
    rewards = code_reward(test_completions, num_parallel=args.num_parallel, **reward_kwargs)
    rewards = [float(reward) for reward in rewards]
    num_generations = len(rewards) // len(generated_code)
    rewards = [rewards[i:i+num_generations] for i in range(0, len(rewards), num_generations)]
    out_dataset = generated_code_dataset.add_column(name="rewards", column=rewards)
    out_dataset = out_dataset.add_column(name="verification_info", column=generated_tests_dataset["verification_info"])
    code_dataset_name = args.generated_code_dataset_name.split("/")[-1].replace("_generated_code", "")
    test_dataset_name = args.generated_tests_dataset_name.split("/")[-1].replace("_generated_tests", "")
    output_dataset_name = f"test-gen/code_{code_dataset_name}_tests_{test_dataset_name}"
    # HuggingFace dataset name limit is 96 characters
    if len(output_dataset_name) > 96:
        output_dataset_name = output_dataset_name.lower().replace("-instruct", "").replace("qwen2.5-coder-", "")
    out_dataset.push_to_hub(output_dataset_name, split=args.generated_code_dataset_split)


if __name__ == "__main__":
    main()