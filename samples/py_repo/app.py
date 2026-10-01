"""The main function: read, preprocess, validate, render."""
from preprocess import normalize as norm
from validate import validate
from render import Renderer


def read_input():
    return input()


def main():
    raw = read_input()
    clean = norm(raw)
    if validate(clean):
        Renderer().draw([clean])
    return clean


if __name__ == "__main__":
    main()
