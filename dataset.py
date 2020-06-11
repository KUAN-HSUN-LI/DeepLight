from torch.utils.data import Dataset
import torchvision.transforms.functional as TF
from torchvision import transforms
import random


class DeepLightDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])])

    def __len__(self):
        return len(self.datas)

    def __getitem__(self, idx):
        data = self.datas[idx]
        return self._transform(data)

    def _transform(self, data):
        raw = data['raw']
        silver_ball = data['silver_ball']
        white_ball = data['white_ball']
        yellow_ball = data['yellow_ball']

        if random.random() > 0.5:
            raw = TF.hflip(raw)
            silver_ball = TF.hflip(silver_ball)
            white_ball = TF.hflip(white_ball)
            yellow_ball = TF.hflip(yellow_ball)

        return self.transform(raw), self.transform(silver_ball), self.transform(white_ball), self.transform(yellow_ball)
