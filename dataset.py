from torch.utils.data import Dataset


class DeepLightDataset(Dataset):
    def __init__(self, datas):
        self.datas = datas
