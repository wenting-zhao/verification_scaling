import re
import os
import datasets

from verl.utils.hdfs_io import copy, makedirs
import argparse

from generate_tests import instruction_only_format_no_few_shot
from utils import prepare_mbpp_prompt


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_dir', default='~/data')
    parser.add_argument('--dataset', required=True)
    parser.add_argument('--dataset_config', default='main')
    parser.add_argument('--easy', action='store_true')

    args = parser.parse_args()

    data_source = args.dataset

    dataset = datasets.load_dataset(data_source, args.dataset_config)

    train_dataset = dataset['train']
    test_dataset = dataset['validation']

    # add a row to each data item that represents a unique id
    def make_map_fn(split, easy=False):

        def process_fn(example, idx):
            problem = prepare_mbpp_prompt(example)
            problem = instruction_only_format_no_few_shot.format(input=problem)

            tests = example["new_verification_info"]["test_cases"]
            if easy:
                tests.sort(key=lambda x: len(x))
            tests = tests[:3]
            tests = ["<assertion>\n"+x.strip()+"\n</assertion>" for x in tests]
            tests = "\n".join(tests)

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

    train_dataset = train_dataset.map(function=make_map_fn('train', easy=args.easy), with_indices=True)
    test_dataset = test_dataset.map(function=make_map_fn('test', easy=args.easy), with_indices=True)

    if args.easy:
        local_dir = args.local_dir + '_easy'
    else:
        local_dir = args.local_dir

    train_dataset.to_parquet(os.path.join(local_dir, 'train.parquet'))
    test_dataset.to_parquet(os.path.join(local_dir, 'test.parquet'))

