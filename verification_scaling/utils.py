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
    
    >>> {function_call}
    {expected_result}
    \"\"\"
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
        exec_timeout = 5

        process = subprocess.run(
            ["python3", "-c", code],
            text=True,
            capture_output=True,
            timeout=exec_timeout
        )

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
    except Exception as e:
        print(f"Error from E2B executor: {e}")
        rewards = [0.0] * len(completions)

    return rewards


def run_async_from_sync(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    """Function wrapping the `run_async` function."""
    # Create a new event loop and set it
    try:
        # Run the async function and get the result
        rewards = asyncio.run(run_async(scripts, language, num_parallel))
    except Exception as e:
        print(f"Error from E2B executor async: {e}")
        raise e

    return rewards


async def run_async(scripts: list[str], language: str, num_parallel: int) -> list[float]:
    # Limit the number of concurrent tasks
    semaphore = asyncio.Semaphore(num_parallel)

    # Create a list of tasks for running scripts concurrently
    tasks = [run_script(script, language, semaphore) for script in scripts]

    # Wait for all tasks to complete and gather their results as they finish
    results = await asyncio.gather(*tasks)
    rewards = list(results)  # collect results

    return rewards


async def run_script(script: str, language: str, semaphore: asyncio.Semaphore) -> float:
    # We set a timeout margin, as the AsyncSandbox timeout does not seem to work
    # These values are based on running 256 examples with the gold solution
    # from open-r1/verifiable-coding-problems-python_decontaminated
    # see scripts/benchmark_e2b.py

    SANDBOX_TIMEOUT = 30
    MARGIN = 2
    REQUEST_TIMEOUT = SANDBOX_TIMEOUT - MARGIN
    ASYNCIO_TIMEOUT = SANDBOX_TIMEOUT + MARGIN

    async with semaphore:
        try:
            sandbox = await AsyncSandbox.create(timeout=SANDBOX_TIMEOUT, request_timeout=REQUEST_TIMEOUT)
            execution = await asyncio.wait_for(sandbox.run_code(script, language=language), timeout=ASYNCIO_TIMEOUT)
            return float(execution.text)
        except (TypeError, ValueError):
            return 0.0
        except asyncio.TimeoutError:
            print("Operation timed out")
            return 0.0
        except Exception as e:
            print(f"Error in `run_script` from E2B sandbox ID {sandbox.sandbox_id} : {e}")
            return 0.0
        finally:
            try:
                await sandbox.kill()
            except Exception as e:
                print(f"Error from E2B executor kill with sandbox ID {sandbox.sandbox_id} : {e}")