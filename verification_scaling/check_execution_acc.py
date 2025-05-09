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
    return parser.parse_args()


def check_test_cases(predicted_tests, ground_truth_tests):
    # Parse the test cases into a more workable format
    def parse_test_case(test_case):
        try:
            # Extract the function call part
            func_call = test_case.split("==")[0].strip()
            # Extract the expected output
            expected_output = test_case.split("==")[1].strip()
        except:
            func_call = test_case
            expected_output = None
        return {
            "function_call": func_call,
            "expected_output": expected_output,
            "original": test_case
        }
    
    parsed_predicted = [parse_test_case(test) for test in predicted_tests]
    parsed_ground_truth = [parse_test_case(test) for test in ground_truth_tests]
    
    correct_count = 0
    results = []
    
    for pred in parsed_predicted:
        # Find matching ground truth test case with same input arguments
        matching_gt = None
        for gt in parsed_ground_truth:
            if pred["function_call"] == gt["function_call"]:
                matching_gt = gt
                break
        
        if matching_gt:
            # Compare expected outputs using literal_eval
            try:
                pred_output = ast.literal_eval(pred["expected_output"])
                gt_output = ast.literal_eval(matching_gt["expected_output"])
                is_correct = pred_output == gt_output
            except (SyntaxError, ValueError, TypeError):
                # If literal_eval fails, fall back to string comparison
                is_correct = pred["expected_output"] == matching_gt["expected_output"]
            
            if is_correct:
                correct_count += 1
                status = "Correct"
            else:
                status = "Incorrect"
        else:
            status = "No matching ground truth test case"
        
        results.append({
            "test_case": pred["original"],
            "status": status,
            "ground_truth": matching_gt["original"] if matching_gt else "N/A"
        })
    
    return correct_count, results

def main():
    args = parse_args()
    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    acc = 0
    for old, new in zip(dataset["verification_info"], dataset["new_verification_info"]):
        correct_count, results = check_test_cases(old["test_cases"], new["test_cases"])
        if len(old["test_cases"]) > 0:
            acc += correct_count / len(old["test_cases"])
    total_acc = acc / len(dataset)
    print(f"Accuracy: {total_acc} ({total_acc*100:.2f}%)")


if __name__ == "__main__":
    main()
