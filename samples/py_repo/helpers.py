"""Small validation helpers."""


def check_type(x):
    if not isinstance(x, str):
        raise TypeError("bad input")


def check_length(x):
    return 0 < len(x) < 1000
