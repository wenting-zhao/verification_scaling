# SPDX-License-Identifier: MIT
import argparse
import re
from datasets import load_dataset
from vllm import LLM, SamplingParams
import torch
from verification_scaling.utils import prepare_mbpp_prompt

instruction_only_format = '''
You are an expert at writing assertion test cases and below is a question with function signature and test cases. 
You must generate 10 assert test cases that will be used to evaluate the code solution's correctness. You must adhere to the provided function signature and test case format.
Here are some examples that you should use as a reference:

Question: 
from typing import Optional

def first_repeated_char(s: str) -> Optional[str]:
    """ 
    Find the first repeated character in a given string.
    
    >>> first_repeated_char("abbac")
    'a'
    """
        
Test Cases:
<assertion>
assert first_repeated_char("ccccc") == "c"
</assertion>
<assertion>
assert first_repeated_char("xvvdf") == "v"
</assertion>
<assertion>
assert first_repeated_char("egeagea") == "e"
</assertion>
<assertion>
assert first_repeated_char("rrrrea") == "r"
</assertion>
<assertion>
assert first_repeated_char("fa") == "None"
</assertion>
<assertion>
assert first_repeated_char("!@#$%^&*!") == "!"
</assertion>
<assertion>
assert first_repeated_char("abcdedcba") == "d"
</assertion>
<assertion>
assert first_repeated_char("") == "None"
</assertion>
<assertion>
assert first_repeated_char("aaaa") == "a"
</assertion>
<assertion>
assert first_repeated_char("a") == "None"
</assertion>

Question: 
def reverse_words(s: str) -> str:
    """ 
    Reverse words in a given string.
    
    >>> reverse_words("hi this is bob.")
    'bob. is this hi'
    """

Test Cases:
<assertion>
assert reverse_words("the") == "the"
</assertion>
<assertion>
assert reverse_words("no way, really?") == "really? way, no"
</assertion>
<assertion>
assert reverse_words("one two three four") == "four three two one"
</assertion>
<assertion>
assert reverse_words("fire away, questions please!!") == "please!! questions away, fire"
</assertion>
<assertion>
assert reverse_words("live, laughter and life.") == "life. and laughter live,"
</assertion>
<assertion>
assert reverse_words("     ") == ""
</assertion>
<assertion>
assert reverse_words("123 456 !@#") == "!@# 456 123"
</assertion>
<assertion>
assert reverse_words("hello
world") == "world hello"
</assertion>
<assertion>
assert reverse_words("  hello   world  ") == "world hello"
</assertion>
<assertion>
assert reverse_words("hello") == "hello"
</assertion>

Here are guidelines for writing the assertion test cases:
1. You must wrap each assertion test case with tags <assertion> and </assertion>.
2. Do not start the assert with any indents or spaces.
3. You must not import any unit testing libraries for the assertions such as "unittest" or "pytest".
4. Each assertion must be complete and immediately executable. Assume the code solution is provided, do not repeat it.
5. Avoid unnecessary string literals, incorrect escaping, wrapping in "```python" or other redundancies.
6. Remember, it is your responsibility to carefully read the question and generate test cases that will evaluate the correctness of the solution.

Here is the question you must provide assertion test cases for:



Question: {input}
Test Cases:
'''


instruction_solution_format = '''
You are an expert at writing assertion test cases and below is a question with function signature and completed code solution. 
You must generate 10 assert statements that will be used to evaluate the code solution's correctness which may or may not be correct.
Here are some examples that you should use as a reference:

Question: 
from typing import Optional

def first_repeated_char(s: str) -> Optional[str]:
    """ 
    Find the first repeated character in a given string.
    
    >>> first_repeated_char("abbac")
    'a'
    """
        
Solution:
from typing import Optional

def first_repeated_char(s: str) -> Optional[str]:
    """ 
    Find the first repeated character in a given string.
    
    >>> first_repeated_char("abbac")
    'a'
    """
    for index, c in enumerate(s):
        if s[:index + 1].count(c) > 1:
            return c
    return None
Test Cases:
<assertion>
assert first_repeated_char("ccccc") == "c"
</assertion>
<assertion>
assert first_repeated_char("xvvdf") == "v"
</assertion>
<assertion>
assert first_repeated_char("egeagea") == "e"
</assertion>
<assertion>
assert first_repeated_char("rrrrea") == "r"
</assertion>
<assertion>
assert first_repeated_char("fa") == "None"
</assertion>
<assertion>
assert first_repeated_char("!@#$%^&*!") == "!"
</assertion>
<assertion>
assert first_repeated_char("abcdedcba") == "d"
</assertion>
<assertion>
assert first_repeated_char("") == "None"
</assertion>
<assertion>
assert first_repeated_char("aaaa") == "a"
</assertion>
<assertion>
assert first_repeated_char("a") == "None"
</assertion>

Question: 
def reverse_words(s: str) -> str:
    """ 
    Reverse words in a given string.
    
    >>> reverse_words("hi this is bob.")
    'bob. is this hi'
    """

Solution:
def reverse_words(s: str) -> str:
    """ 
    Reverse words in a given string.
    
    >>> reverse_words("hi this is bob.")
    'bob. is this hi'
    """
    return ' '.join(reversed(s.split()))
Test Cases:
<assertion>
assert reverse_words("the") == "the"
</assertion>
<assertion>
assert reverse_words("no way, really?") == "really? way, no"
</assertion>
<assertion>
assert reverse_words("one two three four") == "four three two one"
</assertion>
<assertion>
assert reverse_words("fire away, questions please!!") == "please!! questions away, fire"
</assertion>
<assertion>
assert reverse_words("live, laughter and life.") == "life. and laughter live,"
</assertion>
<assertion>
assert reverse_words("     ") == ""
</assertion>
<assertion>
assert reverse_words("123 456 !@#") == "!@# 456 123"
</assertion>
<assertion>
assert reverse_words("hello
world") == "world hello"
</assertion>
<assertion>
assert reverse_words("  hello   world  ") == "world hello"
</assertion>
<assertion>
assert reverse_words("hello") == "hello"
</assertion>

Here are guidelines for writing the assertion test cases:
1. You must wrap each assertion test case with tags <assertion> and </assertion>.
2. Do not start the assert with any indents or spaces.
3. You must not import any unit testing libraries for the assertions such as "unittest" or "pytest".
4. Each assertion must be complete and immediately executable. Assume the code solution is provided, do not repeat it.
5. Avoid unnecessary string literals, incorrect escaping, wrapping in "```python" or other redundancies.
6. Remember, it is your responsibility to carefully read the question and generate test cases that will evaluate the correctness of the solution.

Here is the question and code solution you must provide assertion test cases for:



Question: {input}
Solution:
{code}
Test Cases:
'''

def extract_test_cases(content):
    if "</think>" in content:
        content = content.split("</think>")[-1]
    pattern = r"<assertion>(.*?)</assertion>"
    matches = re.findall(pattern, content, re.DOTALL)
    matches = [match.strip() for match in matches]
    return list(set(matches))


def generate_tests(problems, prompt_format, model):
    """Generate test cases for a dataset using vLLM in batch."""
    llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        temperature=0,
        max_tokens=512,
        n=1,    
        )
    formatted_prompts = []
    for problem in problems:
        if prompt_format == "instruction_only":
            formatted_input = instruction_only_format.format(input=problem)
        elif prompt_format == "instruction_solution":
            #formatted_input = instruction_solution_format.format(input=problem, code=code)
            raise NotImplementedError("Instruction solution format not implemented")
        else:
            raise ValueError("Invalid prompt format")
        formatted_prompts.append(formatted_input)
    messages = [[{"role": "user", "content": prompt}] for prompt in formatted_prompts]
    chat_outputs = llm.chat(messages, sampling_params)
    
    tests = []
    malformed_count = 0
    for output in chat_outputs:
        response = output.outputs[0].text
        current_tests = extract_test_cases(response)
        if current_tests == []:
            malformed_count += 1
        tests.append(current_tests)
    print(f"Ratio of malformed test cases: {malformed_count / len(chat_outputs)}")
    return tests


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True, help="Model for test generation")
    parser.add_argument("--dataset_name", type=str, default="livecodebench", help="Dataset name from HuggingFace")
    parser.add_argument("--dataset_split", type=str, default="test", help="Dataset split to use")
    parser.add_argument("--test_prompt_format", type=str, default="instruction_only", help="Prompt format for test generation")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset_name, split=args.dataset_split, trust_remote_code=True)
    if "livecodebench" in args.dataset_name:
        problems = [example['question_content'] for example in dataset]
        problems = [problem.split("Sample Input 1")[0].split("Example 1")[0].strip() for problem in problems]
        dataset_name = "livecodebench"
    elif "mbpp" in args.dataset_name:
        problems = [prepare_mbpp_prompt(example) for example in dataset]
        dataset_name = "mbpp"
    else:
        raise NotImplementedError("Dataset not supported")
    tests = generate_tests(problems, args.test_prompt_format, args.model)
    verification_info = []
    for test in tests:
        verification_info.append({
            "language": "python",
            "test_cases": test
        })
    dataset = dataset.add_column(name="verification_info", column=verification_info)
    model_name = args.model.split("/")[-1]
    output_dataset_name = f"wentingzhao/{dataset_name}_{model_name}_generated_tests"
    dataset.push_to_hub(output_dataset_name, split=args.dataset_split)