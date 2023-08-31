import torch
from torch.utils.data import Dataset, DataLoader
from config.config import Config


def create_dataloader(dataset):
    return DataLoader(dataset=dataset, batch_size=Config().config["model"]["batch_size"])


class ChessDataset(Dataset):
    def __init__(self, input_list, target_list):
        self.input_list = input_list
        self.target_list = target_list
        self.device = Config().config["model"]["device"]

    def __len__(self):
        return len(self.input_list)

    def __getitem__(self, idx):
        return (torch.tensor(self.input_list[idx], dtype=torch.float, device=self.device),
                torch.tensor(self.target_list[idx], dtype=torch.float, device=self.device))



