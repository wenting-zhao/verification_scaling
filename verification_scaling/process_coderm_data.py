import ast
import copy
import re
from datasets import (
    load_dataset,
    concatenate_datasets,
    Dataset,
    DatasetDict,
)


def is_literal_value(s):
    """Helper function to check if a string represents a literal value"""
    try:
        # Try to evaluate as a literal
        ast.literal_eval(s)
        return True
    except (ValueError, SyntaxError):
        return False


def process_unit_test(test_file_content):
    """
    Extract all assertEqual statements from a unittest file and convert them to assert statements.
    
    Args:
        test_file_content (str): The content of the unittest file
        
    Returns:
        list: A list of strings containing converted assert statements
    """
    # Extract the full assertEqual statements first
    assertEqual_pattern = r'self\.assertEqual\(([^()]*(?:\([^()]*(?:\([^()]*\)[^()]*)*\)[^()]*)*)\)'
    
    assertEqual_matches = re.findall(assertEqual_pattern, test_file_content)
    assert_statements = []
    
    for match in assertEqual_matches:
        # Track parentheses to find the boundary between first and second arguments
        paren_count = 0
        split_pos = -1
        
        for i, char in enumerate(match):
            if char == '(':
                paren_count += 1
            elif char == ')':
                paren_count -= 1
            elif char == ',' and paren_count == 0:
                split_pos = i
                break
        
        if split_pos != -1:
            left_arg = match[:split_pos].strip()
            right_arg = match[split_pos+1:].strip()
            assert_statements.append((left_arg, right_arg))
    
    return assert_statements


def main():
    dataset0 = load_dataset("KAKA22/CodeRM-UnitTest", split="train")
    dataset1 = load_dataset("KAKA22/CodeRM-UnitTest", split="test")
    dataset = concatenate_datasets([dataset0, dataset1])
    processed_dataset = []
    count = 0
    for example in dataset:
        unit_tests = process_unit_test(ast.literal_eval(example["unit_tests"])[0]["code"])
        if len(unit_tests) == 0:
            continue
        valid = True
        unit_tests_clean = copy.deepcopy(unit_tests)
        for test in unit_tests:
            # check if the left hand side contains a function call
            if "(" not in test[0] or ")" not in test[0]:
                valid = False
                break
            # check if the left hand side contains any undefined variables
            input_value = test[0]
            str = input_value.find('(')
            end = input_value.rfind(')')
            input_value = input_value[str:end+1]
            if not is_literal_value(input_value):
                valid = False
                break
            # check if the right hand side contains any undefined variables
            output_value = test[1]
            if not is_literal_value(output_value):
                valid = False
                break

        if valid:
            unit_tests = []
            for test in unit_tests_clean:
                if "error" not in test[1].lower() and len(test[1]) < 30:
                    test = "assert " + test[0] + "==" + test[1]
                    unit_tests.append(test)
            if len(unit_tests) < 3:
                continue
            elif len(unit_tests) == 3:
                challenge_test_list = []
            else:
                challenge_test_list = unit_tests[3:]
                unit_tests = unit_tests[:3]
            new_example = {
                "task_id": count,
                "text": example["question"],
                "code": example["code_ground_truth"],
                "test_list": unit_tests,
                "test_setup_code": "",
                "challenge_test_list": challenge_test_list,
            }
            processed_dataset.append(new_example)
            count += 1

    processed_dataset = Dataset.from_list(processed_dataset)
    mbpp = load_dataset("google-research-datasets/mbpp").cast(processed_dataset.features)
    processed_dataset = DatasetDict({
        "train": processed_dataset,
        "validation": mbpp["validation"]
    })
    processed_dataset.push_to_hub("test-gen/combined")


if __name__ == "__main__":
    main()
