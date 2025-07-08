"""
Custom assertion functions for testing.

This module provides custom assertions that are commonly used across
the test suite, following TDD principles and providing clear error messages.
"""

import json
import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Type, Union
from uuid import UUID

import pytest


def assert_valid_uuid(value: str, message: Optional[str] = None) -> None:
    """Assert that a string is a valid UUID."""
    try:
        UUID(value)
    except (ValueError, TypeError):
        msg = message or f"Expected valid UUID, got: {value}"
        pytest.fail(msg)


def assert_valid_email(value: str, message: Optional[str] = None) -> None:
    """Assert that a string is a valid email address."""
    email_pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if not re.match(email_pattern, value):
        msg = message or f"Expected valid email address, got: {value}"
        pytest.fail(msg)


def assert_valid_url(value: str, schemes: Optional[List[str]] = None, message: Optional[str] = None) -> None:
    """Assert that a string is a valid URL."""
    schemes = schemes or ['http', 'https']
    scheme_pattern = '|'.join(schemes)
    url_pattern = f'^({scheme_pattern})://[a-zA-Z0-9.-]+(?:\\.[a-zA-Z]{{2,}})?(?:/.*)?$'
    
    if not re.match(url_pattern, value):
        msg = message or f"Expected valid URL with schemes {schemes}, got: {value}"
        pytest.fail(msg)


def assert_valid_timestamp(value: str, message: Optional[str] = None) -> None:
    """Assert that a string is a valid ISO timestamp."""
    try:
        # Handle both with and without timezone
        if value.endswith('Z'):
            datetime.fromisoformat(value.replace('Z', '+00:00'))
        else:
            datetime.fromisoformat(value)
    except (ValueError, TypeError):
        msg = message or f"Expected valid ISO timestamp, got: {value}"
        pytest.fail(msg)


def assert_valid_json(value: str, message: Optional[str] = None) -> Dict[str, Any]:
    """Assert that a string is valid JSON and return the parsed data."""
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        msg = message or f"Expected valid JSON, got: {value}"
        pytest.fail(msg)


def assert_dict_structure(data: Dict[str, Any], 
                         required_keys: List[str], 
                         optional_keys: Optional[List[str]] = None,
                         message: Optional[str] = None) -> None:
    """Assert that a dictionary has the expected structure."""
    optional_keys = optional_keys or []
    
    # Check for missing required keys
    missing_keys = [key for key in required_keys if key not in data]
    if missing_keys:
        msg = message or f"Dictionary missing required keys: {missing_keys}"
        pytest.fail(msg)
    
    # Check for unexpected keys
    allowed_keys = set(required_keys + optional_keys)
    unexpected_keys = [key for key in data.keys() if key not in allowed_keys]
    if unexpected_keys:
        msg = message or f"Dictionary has unexpected keys: {unexpected_keys}"
        pytest.fail(msg)


def assert_dict_values_not_none(data: Dict[str, Any], 
                               keys: List[str], 
                               message: Optional[str] = None) -> None:
    """Assert that specific dictionary values are not None."""
    none_keys = [key for key in keys if key in data and data[key] is None]
    if none_keys:
        msg = message or f"Dictionary has None values for keys: {none_keys}"
        pytest.fail(msg)


def assert_list_all_type(data: List[Any], 
                        expected_type: Type, 
                        message: Optional[str] = None) -> None:
    """Assert that all items in a list are of the expected type."""
    for i, item in enumerate(data):
        if not isinstance(item, expected_type):
            msg = message or f"Item at index {i} has type {type(item).__name__}, expected {expected_type.__name__}"
            pytest.fail(msg)


def assert_list_not_empty(data: List[Any], message: Optional[str] = None) -> None:
    """Assert that a list is not empty."""
    if not data:
        msg = message or "Expected non-empty list"
        pytest.fail(msg)


def assert_list_length(data: List[Any], 
                      expected_length: int, 
                      message: Optional[str] = None) -> None:
    """Assert that a list has the expected length."""
    actual_length = len(data)
    if actual_length != expected_length:
        msg = message or f"Expected list length {expected_length}, got {actual_length}"
        pytest.fail(msg)


def assert_list_contains_item(data: List[Any], 
                             item: Any, 
                             message: Optional[str] = None) -> None:
    """Assert that a list contains a specific item."""
    if item not in data:
        msg = message or f"List does not contain item: {item}"
        pytest.fail(msg)


def assert_string_matches_pattern(value: str, 
                                 pattern: str, 
                                 message: Optional[str] = None) -> None:
    """Assert that a string matches a regex pattern."""
    if not re.match(pattern, value):
        msg = message or f"String '{value}' does not match pattern '{pattern}'"
        pytest.fail(msg)


def assert_string_contains_all(value: str, 
                              substrings: List[str], 
                              message: Optional[str] = None) -> None:
    """Assert that a string contains all specified substrings."""
    missing_substrings = [s for s in substrings if s not in value]
    if missing_substrings:
        msg = message or f"String does not contain substrings: {missing_substrings}"
        pytest.fail(msg)


def assert_string_not_empty(value: str, message: Optional[str] = None) -> None:
    """Assert that a string is not empty or whitespace-only."""
    if not value or value.isspace():
        msg = message or "Expected non-empty string"
        pytest.fail(msg)


def assert_numeric_range(value: Union[int, float], 
                        min_value: Optional[Union[int, float]] = None,
                        max_value: Optional[Union[int, float]] = None,
                        message: Optional[str] = None) -> None:
    """Assert that a numeric value is within the specified range."""
    if min_value is not None and value < min_value:
        msg = message or f"Value {value} is less than minimum {min_value}"
        pytest.fail(msg)
    
    if max_value is not None and value > max_value:
        msg = message or f"Value {value} is greater than maximum {max_value}"
        pytest.fail(msg)


def assert_positive_number(value: Union[int, float], message: Optional[str] = None) -> None:
    """Assert that a number is positive."""
    if value <= 0:
        msg = message or f"Expected positive number, got {value}"
        pytest.fail(msg)


def assert_non_negative_number(value: Union[int, float], message: Optional[str] = None) -> None:
    """Assert that a number is non-negative."""
    if value < 0:
        msg = message or f"Expected non-negative number, got {value}"
        pytest.fail(msg)


def assert_datetime_after(dt1: datetime, dt2: datetime, message: Optional[str] = None) -> None:
    """Assert that dt1 is after dt2."""
    if dt1 <= dt2:
        msg = message or f"Expected {dt1} to be after {dt2}"
        pytest.fail(msg)


def assert_datetime_before(dt1: datetime, dt2: datetime, message: Optional[str] = None) -> None:
    """Assert that dt1 is before dt2."""
    if dt1 >= dt2:
        msg = message or f"Expected {dt1} to be before {dt2}"
        pytest.fail(msg)


def assert_datetime_within_seconds(dt1: datetime, 
                                  dt2: datetime, 
                                  seconds: float, 
                                  message: Optional[str] = None) -> None:
    """Assert that two datetimes are within the specified number of seconds."""
    diff = abs((dt1 - dt2).total_seconds())
    if diff > seconds:
        msg = message or f"Datetimes differ by {diff} seconds, expected within {seconds} seconds"
        pytest.fail(msg)


# Agent-specific assertions
def assert_agent_config_valid(config: Dict[str, Any], message: Optional[str] = None) -> None:
    """Assert that an agent configuration is valid."""
    required_keys = ["agent_id", "agent_type", "max_retries", "timeout"]
    assert_dict_structure(config, required_keys, message=message)
    
    # Validate specific fields
    assert_valid_uuid(config["agent_id"], "agent_id must be a valid UUID")
    assert_string_not_empty(config["agent_type"], "agent_type cannot be empty")
    assert_positive_number(config["max_retries"], "max_retries must be positive")
    assert_positive_number(config["timeout"], "timeout must be positive")


def assert_email_data_valid(email_data: Dict[str, Any], message: Optional[str] = None) -> None:
    """Assert that email data is valid."""
    required_keys = ["from", "to", "subject", "body", "date"]
    assert_dict_structure(email_data, required_keys, message=message)
    
    # Validate specific fields
    assert_valid_email(email_data["from"], "Invalid 'from' email address")
    assert_valid_email(email_data["to"], "Invalid 'to' email address")
    assert_string_not_empty(email_data["subject"], "Email subject cannot be empty")
    assert_string_not_empty(email_data["body"], "Email body cannot be empty")
    assert_valid_timestamp(email_data["date"], "Invalid email date timestamp")


def assert_github_data_valid(github_data: Dict[str, Any], message: Optional[str] = None) -> None:
    """Assert that GitHub data is valid."""
    required_keys = ["repository", "sender"]
    assert_dict_structure(github_data, required_keys, message=message)
    
    # Validate repository structure
    repo_required = ["name", "full_name", "owner"]
    assert_dict_structure(github_data["repository"], repo_required, message="Invalid repository structure")
    
    # Validate sender structure
    sender_required = ["login"]
    assert_dict_structure(github_data["sender"], sender_required, message="Invalid sender structure")


def assert_ollama_response_valid(response: Dict[str, Any], message: Optional[str] = None) -> None:
    """Assert that an Ollama response is valid."""
    required_keys = ["model", "response", "done"]
    assert_dict_structure(response, required_keys, message=message)
    
    # Validate specific fields
    assert_string_not_empty(response["model"], "Model name cannot be empty")
    assert_string_not_empty(response["response"], "Response content cannot be empty")
    assert isinstance(response["done"], bool, "done field must be boolean")
    
    # Validate optional timing fields if present
    timing_fields = ["total_duration", "load_duration", "eval_duration"]
    for field in timing_fields:
        if field in response:
            assert_non_negative_number(response[field], f"{field} must be non-negative")


# HTTP-specific assertions
def assert_http_status_success(status_code: int, message: Optional[str] = None) -> None:
    """Assert that an HTTP status code indicates success (2xx)."""
    if not (200 <= status_code < 300):
        msg = message or f"Expected successful HTTP status code (2xx), got {status_code}"
        pytest.fail(msg)


def assert_http_status_error(status_code: int, message: Optional[str] = None) -> None:
    """Assert that an HTTP status code indicates an error (4xx or 5xx)."""
    if not (400 <= status_code < 600):
        msg = message or f"Expected error HTTP status code (4xx or 5xx), got {status_code}"
        pytest.fail(msg)


def assert_response_has_headers(response_headers: Dict[str, str], 
                               required_headers: List[str], 
                               message: Optional[str] = None) -> None:
    """Assert that HTTP response has required headers."""
    missing_headers = [h for h in required_headers if h.lower() not in [k.lower() for k in response_headers.keys()]]
    if missing_headers:
        msg = message or f"Response missing required headers: {missing_headers}"
        pytest.fail(msg)


# File system assertions
def assert_file_exists(file_path: str, message: Optional[str] = None) -> None:
    """Assert that a file exists."""
    from pathlib import Path
    path = Path(file_path)
    if not path.exists():
        msg = message or f"File does not exist: {file_path}"
        pytest.fail(msg)


def assert_file_not_exists(file_path: str, message: Optional[str] = None) -> None:
    """Assert that a file does not exist."""
    from pathlib import Path
    path = Path(file_path)
    if path.exists():
        msg = message or f"File should not exist: {file_path}"
        pytest.fail(msg)


def assert_file_contains(file_path: str, content: str, message: Optional[str] = None) -> None:
    """Assert that a file contains specific content."""
    assert_file_exists(file_path)
    with open(file_path, 'r') as f:
        file_content = f.read()
    if content not in file_content:
        msg = message or f"File {file_path} does not contain expected content: {content}"
        pytest.fail(msg)


def assert_directory_exists(dir_path: str, message: Optional[str] = None) -> None:
    """Assert that a directory exists."""
    from pathlib import Path
    path = Path(dir_path)
    if not path.exists() or not path.is_dir():
        msg = message or f"Directory does not exist: {dir_path}"
        pytest.fail(msg)


# Performance assertions
def assert_execution_time_under(max_seconds: float):
    """Decorator to assert that function execution time is under a threshold."""
    import time
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            start_time = time.time()
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            
            if execution_time > max_seconds:
                pytest.fail(f"Function {func.__name__} took {execution_time:.3f}s, expected under {max_seconds}s")
            
            return result
        return wrapper
    return decorator


def assert_memory_usage_under(max_mb: float):
    """Decorator to assert that function memory usage is under a threshold."""
    import psutil
    import os
    import functools
    
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            process = psutil.Process(os.getpid())
            start_memory = process.memory_info().rss / 1024 / 1024  # MB
            
            result = func(*args, **kwargs)
            
            end_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_increase = end_memory - start_memory
            
            if memory_increase > max_mb:
                pytest.fail(f"Function {func.__name__} used {memory_increase:.2f}MB, expected under {max_mb}MB")
            
            return result
        return wrapper
    return decorator


# Export all assertion functions
__all__ = [
    # Basic validation assertions
    "assert_valid_uuid",
    "assert_valid_email", 
    "assert_valid_url",
    "assert_valid_timestamp",
    "assert_valid_json",
    
    # Data structure assertions
    "assert_dict_structure",
    "assert_dict_values_not_none",
    "assert_list_all_type",
    "assert_list_not_empty",
    "assert_list_length",
    "assert_list_contains_item",
    
    # String assertions
    "assert_string_matches_pattern",
    "assert_string_contains_all",
    "assert_string_not_empty",
    
    # Numeric assertions
    "assert_numeric_range",
    "assert_positive_number",
    "assert_non_negative_number",
    
    # Datetime assertions
    "assert_datetime_after",
    "assert_datetime_before",
    "assert_datetime_within_seconds",
    
    # Domain-specific assertions
    "assert_agent_config_valid",
    "assert_email_data_valid",
    "assert_github_data_valid",
    "assert_ollama_response_valid",
    
    # HTTP assertions
    "assert_http_status_success",
    "assert_http_status_error",
    "assert_response_has_headers",
    
    # File system assertions
    "assert_file_exists",
    "assert_file_not_exists",
    "assert_file_contains",
    "assert_directory_exists",
    
    # Performance assertions
    "assert_execution_time_under",
    "assert_memory_usage_under",
]