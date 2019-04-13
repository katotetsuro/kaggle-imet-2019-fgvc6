import dill
from pathlib import Path
from argparse import ArgumentParser


def main():
    args = ArgumentParser()
    args.add_argument('--out', default='model_defs', type=str)

    args = args.parse_args()

    p = Path(args.out)
