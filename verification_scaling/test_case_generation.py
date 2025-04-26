# SPDX-License-Identifier: MIT
import re
from vllm import LLM, SamplingParams
import torch
from verification_scaling.utils import cleanup

instruction_only_format = '''
You are an expert at writing assertion test cases and below is a question with function signature and completed code solution. 
You must generate 10 assert statements that will be used to evaluate the code solution's correctness which may or may not be correct.
Here are some examples that you should use as a reference:

Question: 
Find the first repeated character in a given string.

Test Cases:
<input>"ccccc"</input>
<output>"c"</output>

<input>"xvvdf"</input>
<output>"v"</output>

<input>"egeagea"</input>
<output>"e"</output>

<input>"rrrrea"</input>
<output>"r"</output>

<input>"fa"</input>
<output>"None"</output>

<input>"!@#$%^&*!"</input>
<output>"!"</output>

<input>"abcdedcba"</input>
<output>"d"</output>

<input>""</input>
<output>"None"</output>

<input>"aaaa"</input>
<output>"a"</output>

<input>"a"</input>
<output>"None"</output>

Question:
Reverse words in a given string.

Test Cases:
<input>"the"</input>
<output>"the"</output>

<input>"no way, really?"</input>
<output>"really? way, no"</output>

<input>"one two three four"</input>
<output>"four three two one"</output>

<input>"fire away, questions please!!"</input>
<output>"please!! questions away, fire"</output>

<input>"live, laughter and life."</input>
<output>"life. and laughter live,"</output>

<input>"     "</input>
<output>""</output>

<input>"123 456 !@#"</input>
<output>"!@# 456 123"</output>

<input>"hello
world"</input>
<output>"world hello"</output>

<input>"  hello   world  "</input>
<output>"world hello"</output>

<input>"hello"</input>
<output>"hello"</output>

Here are guidelines for writing the assertion test cases:
1. You must wrap each test case with tags <input> and </input> for the input and <output> and </output> for the output.
2. Do not start the input or output with any indents or spaces.
3. You must not import any unit testing libraries for the assertions such as "unittest" or "pytest".
4. Each test case must be complete and immediately executable. Assume the code solution is provided, do not repeat it.
5. Avoid unnecessary string literals, incorrect escaping, wrapping in "```python" or other redundancies.
6. Remember, it is your responsibility to carefully read the question and generate test cases that will evaluate the correctness of the solution.

Here is the question and code solution you must provide test cases for:



Question: {input}
Test Cases:
'''


instruction_solution_format = '''
You are an expert at writing assertion test cases and below is a question with function signature and completed code solution. 
You must generate 10 assert statements that will be used to evaluate the code solution's correctness which may or may not be correct.
Here are some examples that you should use as a reference:

Question: 
Find the first repeated character in a given string.

Solution:
```python
s = input()
for index, c in enumerate(s):
    if s[:index + 1].count(c) > 1:
        print(c)
        break
else:
    print(None)
```

Test Cases:
<input>"ccccc"</input>
<output>"c"</output>

<input>"xvvdf"</input>
<output>"v"</output>

<input>"egeagea"</input>
<output>"e"</output>

<input>"rrrrea"</input>
<output>"r"</output>

<input>"fa"</input>
<output>"None"</output>

<input>"!@#$%^&*!"</input>
<output>"!"</output>

<input>"abcdedcba"</input>
<output>"d"</output>

<input>""</input>
<output>"None"</output>

<input>"aaaa"</input>
<output>"a"</output>

<input>"a"</input>
<output>"None"</output>

Question:
Reverse words in a given string.

Solution:
```python
s = input()
print(' '.join(reversed(s.split())))
```

Test Cases:
<input>"the"</input>
<output>"the"</output>

<input>"no way, really?"</input>
<output>"really? way, no"</output>

<input>"one two three four"</input>
<output>"four three two one"</output>

<input>"fire away, questions please!!"</input>
<output>"please!! questions away, fire"</output>

<input>"live, laughter and life."</input>
<output>"life. and laughter live,"</output>

<input>"     "</input>
<output>""</output>

<input>"123 456 !@#"</input>
<output>"!@# 456 123"</output>

<input>"hello
world"</input>
<output>"world hello"</output>

<input>"  hello   world  "</input>
<output>"world hello"</output>

<input>"hello"</input>
<output>"hello"</output>

Here are guidelines for writing the test cases:
1. You must wrap each test case with tags <input> and </input> for the input and <output> and </output> for the output.
2. Do not start the input or output with any indents or spaces.
3. You must not import any unit testing libraries for the test cases such as "unittest" or "pytest".
4. Each test case must be complete and immediately executable. Assume the code solution is provided, do not repeat it.
5. Avoid unnecessary string literals, incorrect escaping, wrapping in "```python" or other redundancies.
6. Remember, it is your responsibility to carefully read the question and generate test cases that will evaluate the correctness of the solution.

Here is the question and code solution you must provide test cases for:



Question: {input}
Solution:
{code}
Test Cases:
'''

def extract_test_cases(content):
    if "</think>" in content:
        content = content.split("</think>")[-1]
    input_pattern = r"<input>(.*?)</input>"
    output_pattern = r"<output>(.*?)</output>"
    input_matches = re.findall(input_pattern, content, re.DOTALL)
    output_matches = re.findall(output_pattern, content, re.DOTALL)
    input_len = len(input_matches)
    output_len = len(output_matches)
    if input_len == 0 or output_len == 0:
        print(f"No input or output matches found in the content\n{content}")
        input_matches = []
        output_matches = []
    elif output_len > input_len:
        print(f"Output length {output_len} is greater than input length {input_len}\n{content}")
        input_matches = []
        output_matches = []
    else:
        min_len = min(input_len, output_len)
        input_matches = input_matches[:min_len]
        output_matches = output_matches[:min_len]
    return input_matches, output_matches


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
    
    test_inputs = []
    test_outputs = []
    malformed_count = 0
    for output in chat_outputs:
        response = output.outputs[0].text
        inputs, outputs = extract_test_cases(response)
        if inputs == []:
            malformed_count += 1
        test_inputs.append(inputs)
        test_outputs.append(outputs)
    print(f"Ratio of malformed test cases: {malformed_count / len(chat_outputs)}")
    cleanup()
    return test_inputs, test_outputs
