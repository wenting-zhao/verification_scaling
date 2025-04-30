import argparse
from datasets import load_dataset
from verification_scaling.utils import get_function_output

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for generated tests")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split for generated tests")
    parser.add_argument("--num_parallel", type=int, default=20, help="Number of parallel tests to run")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True).select(range(2))
    code = dataset["code"]
    verification_info = dataset["verification_info"]
    for info in verification_info:
        info["test_cases"] = [test.replace("assert ", "").split("==")[0].strip() for test in info["test_cases"]]
    kwargs = dict()
    kwargs["verification_info"] = verification_info
    outputs = get_function_output(code, num_parallel=args.num_parallel, **kwargs)
    print(outputs[0])


if __name__ == "__main__":
    main()