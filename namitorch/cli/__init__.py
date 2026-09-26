import argparse
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(prog="namitorch")
    parser.add_argument("command", choices=("prepare_corpus", "train"))
    argv = list(sys.argv[1:] if argv is None else argv)
    arguments = parser.parse_args(argv[:1])
    if arguments.command == "prepare_corpus":
        from .prepare_corpus import main as command
    else:
        from .train import main as command
    return command(argv[1:])


__all__ = ["main"]
