"""Input preprocessing: runs before the main function."""


def collapse_spaces(s):
    return " ".join(s.split())


def normalize(s):
    trimmed = s.strip()
    spaced = collapse_spaces(trimmed)
    return spaced.lower()
