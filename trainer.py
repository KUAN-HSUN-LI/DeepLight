from model import DeepLight, Discriminator
import torch


class Trainer():
    def __init__(self, args):
        self.batch_size = args.batch_size
        self.epochs = args.epochs
        self.deeplight = DeepLight()
        self.discriminator = Discriminator()

    def save(self):
        raise NotImplementedError
