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


def extract_function_signature(code_block: str) -> str:
    """
    Extract the function signature line from a code block.

    Args:
        code_block: A string containing Python code with function definitions

    Returns:
        The function signature line (def line) of the first function found
    """
    # This looks for 'def', function name, parameters, and return type hint if present
    pattern = r'def\s+\w+\s*\([^)]*\)(?:\s*->\s*[^:]+)?:'

    match = re.search(pattern, code_block)
    if match:
        return match.group(0).rstrip(':')  # Remove the trailing colon
    return ""


def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    if "humaneval" in args.dataset_name.lower():
        code = []
        for example in dataset:
            function_signature = extract_function_signature(example["prompt"])
            code.append(function_signature + ":\n" + example["canonical_solution"])
    elif "livecodebench" in args.dataset_name.lower():
        code = load_dataset("test-gen/livecodebench-with-code", split=args.dataset_split)["code"]
    else:
        code = dataset["code"]
    verification_info = deepcopy(dataset["verification_info"])
    for info in verification_info:
        info["test_cases"] = [test.replace("assert ", "").split("==")[0].strip() for test in info["test_cases"]]
    kwargs = dict()
    kwargs["verification_info"] = verification_info
    outputs = get_function_output(code, num_parallel=args.num_parallel, **kwargs)
    outputs = [
        ast.literal_eval(output.replace("\n ", " "))
        if output != '0' and output is not None
        else [None]
        for output in outputs
    ]
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
