import argparse
import os
from datasets import Dataset, load_dataset

from generate_tests import instruction_only_format_no_few_shot
from utils import (
    prepare_mbpp_prompt,
    format_test_cases,
    sort_by_input_length,
    get_test_cases_with_unique_outputs,
    get_easy_test_cases_with_unique_outputs
)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataset_config', default='main')
    parser.add_argument('--easy', action='store_true')
    parser.add_argument('--unique', action='store_true')
    parser.add_argument('--bucket', action='store_true')
    parser.add_argument('--easy_unique', action='store_true')
    args = parser.parse_args()

    data_source = args.dataset

    dataset = load_dataset(data_source, args.dataset_config)

    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    if args.bucket:
        new_dataset = []
        def process_fn(dataset):
            for idx, example in enumerate(dataset):
                problem = prepare_mbpp_prompt(example)
                problem = instruction_only_format_no_few_shot.format(input=problem)
                tests = example["new_verification_info"]["test_cases"]
                if args.easy:
                    tests = sort_by_input_length(tests)
                    first_three_tests = tests[:3]
                elif args.unique:
                    first_three_tests = get_test_cases_with_unique_outputs(tests, 3)
                elif args.easy_unique:
                    first_three_tests = get_easy_test_cases_with_unique_outputs(tests, 3)
                else:
                    first_three_tests = tests[:3]
                other_tests = [test for test in tests if test not in first_three_tests]
                split_tests = [first_three_tests]
                for i in range(0, len(other_tests), 3):
                    split_tests.append(other_tests[i:i+3])
                if len(split_tests) >= 1:
                    new_problem = problem.replace("Test Cases:", "Easy Test Cases:")
                    new_dataset.append({
                        "data_source": data_source,
                        "extra_info": {
                            'split': 'train',
                            'index': idx,
                            "question": new_problem,
                            "answer": format_test_cases(split_tests[0]),
                        }
                    })
                    if len(split_tests) >= 2:
                        for i in range(1, len(split_tests)):
                            new_problem = problem.replace("Test Cases:", f"Hard Test Cases:")
                            new_dataset.append({
                                "data_source": data_source,
                                "extra_info": {
                                    'split': 'train',
                                    'index': idx,
                                    "question": new_problem,
                                    "answer": format_test_cases(split_tests[i]),
                                }
                            })
            return Dataset.from_list(new_dataset)
        train_dataset = process_fn(train_dataset)
        test_dataset = process_fn(test_dataset)

    else:
        # add a row to each data item that represents a unique id
        def make_map_fn(split, easy=False, unique=False, easy_unique=False):

            def process_fn(example, idx):
                problem = prepare_mbpp_prompt(example)
                problem = instruction_only_format_no_few_shot.format(input=problem)

                tests = example["new_verification_info"]["test_cases"]
                if easy:
                    tests = sort_by_input_length(tests)[:3]
                elif unique:
                    tests = get_test_cases_with_unique_outputs(tests, 3)
                elif easy_unique:
                    tests = get_easy_test_cases_with_unique_outputs(tests, 3)
                else:
                    tests = tests[:3]
                tests = format_test_cases(tests)
                data = {
                    "data_source": data_source,
                    "extra_info": {
                        'split': split,
                        'index': idx,
                        "question": problem,
                        "answer": tests,
                    }
                }
                return data

            return process_fn

        train_dataset = train_dataset.map(function=make_map_fn('train', easy=args.easy, unique=args.unique, easy_unique=args.easy_unique), with_indices=True)
        train_dataset = train_dataset.filter(lambda example: example["extra_info"]["answer"] != "")
        test_dataset = test_dataset.map(function=make_map_fn('test', easy=args.easy, unique=args.unique, easy_unique=args.easy_unique), with_indices=True)
        test_dataset = test_dataset.filter(lambda example: example["extra_info"]["answer"] != "")

    if args.easy:
        local_dir = args.local_dir + '_easy'
    elif args.unique:
        local_dir = args.local_dir + '_unique'
    elif args.easy_unique:
        local_dir = args.local_dir + '_easy_unique'
    else:
        local_dir = args.local_dir

    if args.bucket:
        local_dir = local_dir + '_bucket'

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

    print("average #tests:", sum(example["extra_info"]["answer"].count("<assertion>") for example in train_dataset) / len(train_dataset))
    print("average #tokens:", sum(len(example["extra_info"]["answer"].split()) for example in train_dataset) / len(train_dataset))