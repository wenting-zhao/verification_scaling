import contextlib
import gc
import torch


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


def extract_assert_test_cases(assert_statements):
    inputs = []
    outputs = []
    
    for statement in assert_statements:
        print(statement)
        # Skip the "assert " part
        statement = statement.replace("assert ", "")
        
        # Split by the equality operator
        parts = statement.split("==")
        
        # Extract the function call and expected output
        func_call = parts[0].strip()
        expected_output = parts[1].strip('"').strip()
        
        # Find the opening and closing parentheses of the function call
        open_paren_index = func_call.find("(")
        close_paren_index = func_call.rfind(")")
        
        if open_paren_index != -1 and close_paren_index != -1:
            # Extract just the arguments portion
            args_part = func_call[open_paren_index + 1:close_paren_index]
            
            # Add to our lists
            inputs.append(args_part)
            outputs.append(expected_output)
    
    return inputs, outputs