"""Input validation."""
from helpers import check_type, check_length


def validate(x):
    check_type(x)
    if not check_length(x):
        return False
    return True
