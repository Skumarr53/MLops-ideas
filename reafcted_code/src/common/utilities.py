import os

def get_env_variable(var_name: str, default_value: str = None) -> str:
    """Get the environment variable or return a default value."""
    return os.getenv(var_name, default_value)
