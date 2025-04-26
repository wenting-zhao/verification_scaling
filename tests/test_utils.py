from verification_scaling.utils import extract_assert_test_cases

def test_basic_extraction():
    assert_statements = [
        "assert remove_Occ(\"hello\",\"l\") == \"heo\"",
        "assert remove_Occ(\"abcda\",\"a\") == \"bcd\"",
        "assert remove_Occ(\"PHP\",\"P\") == \"H\""
    ]
    
    expected_inputs = ["\"hello\",\"l\"", "\"abcda\",\"a\"", "\"PHP\",\"P\""]
    expected_outputs = ["heo", "bcd", "H"]
    
    inputs, outputs = extract_assert_test_cases(assert_statements)
    
    assert inputs == expected_inputs
    assert outputs == expected_outputs

def test_different_function_name():
    assert_statements = [
        "assert add_numbers(5, 10) == 15",
        "assert add_numbers(0, 0) == 0",
        "assert add_numbers(-5, 5) == 0"
    ]
    
    expected_inputs = ["5, 10", "0, 0", "-5, 5"]
    expected_outputs = ["15", "0", "0"]
    
    inputs, outputs = extract_assert_test_cases(assert_statements)
    
    assert inputs == expected_inputs
    assert outputs == expected_outputs

def test_mixed_function_names():
    assert_statements = [
        "assert multiply(3, 4) == 12",
        "assert subtract(10, 5) == 5",
        "assert divide(20, 4) == 5"
    ]
    
    expected_inputs = ["3, 4", "10, 5", "20, 4"]
    expected_outputs = ["12", "5", "5"]
    
    inputs, outputs = extract_assert_test_cases(assert_statements)
    
    assert inputs == expected_inputs
    assert outputs == expected_outputs

def test_complex_arguments():
    assert_statements = [
        "assert process_list([1, 2, 3], sum) == 6",
        "assert format_data({\"name\": \"Alice\", \"age\": 30}) == \"Alice: 30\""
    ]
    
    expected_inputs = ["[1, 2, 3], sum", "{\"name\": \"Alice\", \"age\": 30}"]
    expected_outputs = ["6", "Alice: 30"]
    
    inputs, outputs = extract_assert_test_cases(assert_statements)
    
    assert inputs == expected_inputs
    assert outputs == expected_outputs
