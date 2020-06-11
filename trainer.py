from model import DeepLight, Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch


class Trainer():
    def __init__(self, train_data, args):
        self.device = args.device
        self.batch_size = args.batch_size
        self.train_data = train_data
        self.deeplight = DeepLight().to(self.device)
        self.discriminator = Discriminator().to(self.device)
        self.opt = torch.optim.AdamW(self.deeplight.parameters(), lr=args.lr)

    def run_epoch(self, epoch, training):
        self.deeplight.train(training)
        self.discriminator.train(training)
        if training:
            description = 'Train'
            dataset = self.train_data
            shuffle = True
        else:
            description = 'Valid'
            dataset = self.train_data
            shuffle = False

        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.batch_size,
                                shuffle=shuffle,
                                num_workers=4)

        trange = tqdm(enumerate(dataloader), total=len(dataloader), desc=description)
        loss = 0
        for idx, (raw, silver_ball, white_ball, yellow_ball) in trange:
            self._run_iter()

    def _run_iter(self):
        ...

    def save(self, path):
        torch.save({"deeplight": self.deeplight.state_dict(), "discriminator": self.discriminator.state_dict()}, path)
