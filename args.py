import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", "-b", default=64, type=int)
    parser.add_argument("--epochs", "-e", default=100, type=int)
    parser.add_argument("--lr", default=0.00015, type=float)
    parser.add_argument("--device", default="cuda", type=str)
    parser.add_argument("--train", action="store_true")
    return parser.parse_args()
