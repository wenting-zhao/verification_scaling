import argparse
import ast
import re
from copy import deepcopy
from datasets import load_dataset
from verification_scaling.utils import get_function_output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", type=str, required=True, help="Dataset name for generated tests")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split for generated tests")
    parser.add_argument("--num_parallel", type=int, default=20, help="Number of parallel tests to run")
    return parser.parse_args()

def replace_assert_output(assert_statement, new_value):
    """
    Replace the value after '==' in a single assert statement.

    Args:
        assert_statement (str): A single assert statement
        new_value (any): The new value to use after ==

    Returns:
        str: The updated assert statement
    """
    # Find where the == occurs and replace everything after it
    pattern = r'(assert.+?==\s*).*'
    try:
        # Try to evaluate as Python literal
        ast.literal_eval(new_value)
        # If it evaluates successfully, use it as-is (it's a valid Python literal)
        formatted_value = new_value
    except (ValueError, SyntaxError, TypeError):
        # If evaluation fails, treat as a regular string
        formatted_value = f'"{new_value}"'
    def replace_match(match):
        return match.group(1) + formatted_value
    return re.sub(pattern, replace_match, assert_statement)


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    code = dataset["code"]
    verification_info = deepcopy(dataset["verification_info"])
    for info in verification_info:
        info["test_cases"] = [test.replace("assert ", "").split("==")[0].strip() for test in info["test_cases"]]
    kwargs = dict()
    kwargs["verification_info"] = verification_info
    outputs = get_function_output(code, num_parallel=args.num_parallel, **kwargs)
    outputs = [ast.literal_eval(output.replace("\n ", " ")) if output != 0.0 and output is not None else [None] for output in outputs]
    new_verification_info = []
    for info, output in zip(dataset["verification_info"], outputs):
        new_info = dict()
        new_info["test_cases"] = [replace_assert_output(test, o) for test, o in zip(info["test_cases"], output) if o is not None]
        new_info["language"] = info["language"]
        new_verification_info.append(new_info)
    dataset = dataset.add_column(name="new_verification_info", column=new_verification_info)
    dataset.push_to_hub(args.dataset_name+"_updated")


if __name__ == "__main__":
    main()
