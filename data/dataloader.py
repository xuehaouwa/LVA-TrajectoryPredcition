from torch.utils.data import Dataset
import torch


class TrajectoryDataLoader(Dataset):
    def __init__(self, obs, label):
        self.length = len(obs)
        self.obs = torch.Tensor(obs)
        self.label = torch.Tensor(label)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        obs = self.obs[idx]
        target = self.label[idx]

        return obs, target


