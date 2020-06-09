import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", default=1024, type=int)
    parser.add_argument("--epochs", "-e", default=10, type=int)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--test", action="store_true")
    return parser.parse_args()
