import asyncio
import contextlib
import gc
import json
import re
import torch

from dotenv import load_dotenv
from e2b_code_interpreter import AsyncSandbox

load_dotenv()


def cleanup() -> None:
    """Clean up resources associated with the given model."""
    if torch.cuda.is_available():
        from vllm.distributed.parallel_state import (
            destroy_model_parallel,
            destroy_distributed_environment,
        )
        destroy_model_parallel()
        destroy_distributed_environment()
        with contextlib.suppress(AssertionError):
            torch.distributed.destroy_process_group()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


def format_test_cases(tests):
    tests = [x.replace(" == ", "==").replace(" ==", "==").replace("== ", "==") for x in tests]
    tests = ["<assertion>\n"+x.strip()+"\n</assertion>" for x in tests]
    tests = "\n".join(tests)
    return tests


def sort_by_input_length(test_cases):
    def extract_input_length(test_case):
        # Extract everything between the outermost parentheses
        start_idx = test_case.find("(")
        end_idx = test_case.split("==")[0].rfind(")")

        if start_idx != -1 and end_idx != -1:
            # Get the entire input string
            input_str = test_case[start_idx+1:end_idx]
            return len(input_str)
        return 0

    # Sort the test cases based on the length of the input
    return sorted(test_cases, key=extract_input_length)


def get_output_to_test_cases_map(test_cases):
    """
    Group test cases by their expected outputs.

    Args:
        test_cases: List of test case assertions

    Returns:
        Dictionary mapping outputs to lists of test cases
    """
    output_to_test_cases = {}

    for test_case in test_cases:
        # Find the expected output (after == or =)
        if "==" in test_case:
            output = test_case.split("==")[1].strip()
        else:
            continue

        # Remove surrounding quotes if present
        output = output.strip('"\'')

        # Add test case to the list for this output
        if output not in output_to_test_cases:
            output_to_test_cases[output] = []
        output_to_test_cases[output].append(test_case)

    return output_to_test_cases


def get_test_cases_with_unique_outputs(test_cases, n):
    if len(test_cases) == 0:
        return []

    # Group test cases by their outputs
    output_to_test_cases = get_output_to_test_cases_map(test_cases)

    # Get unique outputs
    unique_outputs = list(output_to_test_cases.keys())

    # Initialize result list
    result = []

    # Fill result list with n test cases
    for i in range(n):
        if len(unique_outputs) == 0:
            break
        # Get the output to use (cycling through the unique outputs)
        output_index = i % len(unique_outputs)
        output = unique_outputs[output_index]

        # Get a test case for this output
        test_cases_for_output = output_to_test_cases[output]
        case_index = (i // len(unique_outputs)) % len(test_cases_for_output)
        result.append(test_cases_for_output[case_index])

    return result


def get_easy_test_cases_with_unique_outputs(test_cases, n):
    if len(test_cases) == 0:
        return []

    # Parse test cases to extract outputs and calculate input lengths
    parsed_tests = []

    for test in test_cases:
        # Extract output
        if "==" in test:
            output = test.split("==")[1].strip()
        else:
            continue

        # Remove quotes
        output = output.strip('"\'')

        # Extract arguments
        start = test.find("(")
        end = test.rfind(")")
        if start == -1 or end == -1:
            continue

        args = test[start+1:end]
        args_length = len(args)

        parsed_tests.append((test, output, args_length))

    # Sort all tests by argument length (shortest first)
    parsed_tests.sort(key=lambda x: x[2])

    # Take the first n tests with unique outputs
    result = []
    seen_outputs = set()

    for test, output, _ in parsed_tests:
        if output not in seen_outputs:
            result.append(test)
            seen_outputs.add(output)

            if len(result) >= n:
                break

    # If we still need more, add additional tests with already seen outputs
    if len(result) < n:
        output_counts = {}
        for test, output, _ in parsed_tests:
            if test in result:
                continue

            if output not in output_counts:
                output_counts[output] = 0

            # Only consider tests where we've already used this output
            if output in seen_outputs:
                output_counts[output] += 1
                if output_counts[output] == 1:  # Take the next shortest for each output
                    result.append(test)
                    if len(result) >= n:
                        break

    return result


def prepare_humaneval_prompt(example):
    return example['prompt']


def prepare_livecodebench_prompt(example):
    starter_code = example['starter_code'].replace("self, ", "").replace("self", "").replace("pass", "").strip()
    question_content = "\n".join([f"    {one}" for one in example['question_content'].split("\n")])
    prompt = f"{starter_code}\n    \"\"\"\n{question_content}\n    \"\"\"\n    pass"
    return prompt

def prepare_mbpp_prompt(example):
    prompt = generate_function_doc(example['test_list'][0], example['code'], example['text'])
    return prompt


def generate_function_doc(assertion, implementation, description):
    """
    Generates a documented function from an assertion, implementation and description.
    
    Args:
        assertion (str): Function assertion like 'assert func(args) == result'
        implementation (str): Function implementation code
        description (str): Text description of the function
        
    Returns:
        str: Formatted function with docstring containing description and example
    """
    # Extract function name and call from assertion
    assertion_pattern = r'assert\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\((.+?)\)\s*==\s*(.+)'
    assertion_match = re.search(assertion_pattern, assertion)

    if not assertion_match:
        return "Error: Could not parse the assertion correctly"

    func_name = assertion_match.group(1)
    func_args = assertion_match.group(2)
    expected_result = assertion_match.group(3)

    # Create the function call example for the docstring
    function_call = f"{func_name}({func_args})"

    # Find function signature in implementation
    impl_pattern = r'def\s+' + re.escape(func_name) + r'\s*\(\s*([^)]+)\s*\)'
    impl_match = re.search(impl_pattern, implementation)

    if not impl_match:
        return f"Error: Could not find implementation for function '{func_name}'"

    # Get the parameter names from the implementation
    param_str = impl_match.group(1)

    # Create the formatted function
    formatted_function = f"""def {func_name}({param_str}):
    \"\"\"
    {description}
    \"\"\"
        pass
    """

    return formatted_function


def extract_code(completion: str, language: str = "python") -> str:
    pattern = re.compile(rf"```{language}\n(.*?)```", re.DOTALL)
    matches = pattern.findall(completion)
    extracted_answer = matches[-1] if len(matches) >= 1 else ""
    return extracted_answer


# modified from https://github.com/huggingface/open-r1/blob/main/src/open_r1/rewards.py
def code_reward(completions, num_parallel: int = 2, **kwargs) -> list[float]:
    """Reward function that evaluates code snippets using the E2B code interpreter.

    Assumes the dataset contains a `verification_info` column with test cases.
    """

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        code = code + "\\n" + "\\n".join(test_cases)
        exec_timeout = 30
        try:
            process = subprocess.run(
                ["python3", "-c", code],
                text=True,
                capture_output=True,
                timeout=exec_timeout
            )
        except Exception:
            return 0
        if process.returncode != 0:  # Error in execution
            reward = 0
        else:
            reward = 1

        return reward
    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    code_snippets = [extract_code(completion["content"]) for completion in completions]
    verification_info = kwargs["verification_info"]
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_snippets, verification_info)
    ]

    language = verification_info[0]["language"]
    if not all(v["language"] == language for v in verification_info):
        raise ValueError("All verification_info must have the same language", verification_info)

    try:
        rewards = run_async_from_sync(scripts, language, num_parallel)
        rewards = [float(one) for one in rewards]
    except Exception as e:
        raise Exception(f"Error from E2B executor: {e}")

    return rewards


def get_function_output(code_list, num_parallel: int = 2, **kwargs) -> list[str]:
    """Get the output of the ground truth function given an input.

    Assumes the dataset contains a `verification_info` column with test cases.
    """

    # TODO: add support for other languages in E2B: https://e2b.dev/docs/code-interpreting/supported-languages
    """Returns a reward function that evaluates code snippets in a sandbox."""
    evaluation_script_template = """
    import subprocess
    import json

    def evaluate_code(code, test_cases):
        exec_timeout = 30
        outputs = []
        for test in test_cases:
            code_to_run = code + "\\n" + 'print('+test+')'
            try:
                process = subprocess.run(
                    ["python3", "-c", code_to_run],
                    text=True,
                    capture_output=True,
                    timeout=exec_timeout
                )
            except Exception:
                outputs.append(None)

            if process.returncode != 0:  # Error in execution
                outputs.append(None)
            else:
                outputs.append(process.stdout.strip())

        return outputs
    code_snippet = {code}
    test_cases = json.loads({test_cases})

    evaluate_code(code_snippet, test_cases)
    """
    scripts = [
        evaluation_script_template.format(code=json.dumps(code), test_cases=json.dumps(json.dumps(info["test_cases"])))
        for code, info in zip(code_list, kwargs["verification_info"])
    ]

    language = kwargs["verification_info"][0]["language"]
    if not all(v["language"] == language for v in kwargs["verification_info"]):
        raise ValueError("All verification_info must have the same language", kwargs["verification_info"])

    try:
        outputs = run_async_from_sync(scripts, language, num_parallel)
    except Exception as e:
        raise Exception(f"Error from E2B executor: {e}")

    return outputs


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[str]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[str]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> str:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 300
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return execution.text
        except (TypeError, ValueError):
            return '0'
        except asyncio.TimeoutError:
            print("Operation timed out")
            return '0'
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return '0'
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")
