from vllm import LLM, SamplingParams
import torch
from verification_scaling.utils import cleanup
code_generation_std_format = '''
Solve the following coding problem using the programming language python:

{problem}

The input will be stdin and you should print your solution to stdout.

Now solve the problem and return the code.
'''

code_generation_function_format = '''
Solve the following coding problem using the programming language python:

{problem}

Now solve the problem and return the code.
'''

def generate_code(problems, prompt_format, model, temperature, num_generations):
    """Generate code solutions for a dataset using vLLM.
    
    Args:
        problems: List of problems
        model: vLLM model name or path
        temperature: Sampling temperature
        num_generations: Number of generations per example
        
    Returns:
        List of lists containing generated code solutions for each example
    """
    llm = LLM(model=model, tensor_parallel_size=torch.cuda.device_count())
    sampling_params = SamplingParams(
        temperature=temperature,
        max_tokens=512,
        n=num_generations
    )
    if prompt_format == "std":
        code_prompts = [code_generation_std_format.format(problem=problem) for problem in problems]
    elif prompt_format == "function":
        #code_prompts = [code_generation_function_format.format(problem=problem) for example in dataset]
        raise NotImplementedError("Function format not implemented")
    else:
        raise ValueError("Invalid prompt format")
    messages = [[{"role": "user", "content": prompt}] for prompt in code_prompts]
    outputs = llm.chat(messages, sampling_params)
    generated_code = [[one.text for one in output.outputs] for output in outputs]
    cleanup()
    return generated_code
