from args import get_args
from utils import *
from dataset import DeepLightDataset
from trainer import Trainer
import os


args = get_args()
data = load_pkl("dataset.pkl")
train_data = DeepLightDataset(data)
trainer = Trainer(train_data, args)

if args.train:
    EPOCHS = args.epochs

    for epoch in range(EPOCHS):
        print(f"epoch{epoch}:")
        trainer.run_epoch(epoch, True)
        trainer.save(f"model/model-{epoch}.pkl")
else:
    ...
