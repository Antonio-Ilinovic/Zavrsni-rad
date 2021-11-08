import torch

from PatchesDataset import PatchesDataset

model = torch.load('trained_model.pth')
model.eval()

patches_dataset = PatchesDataset()

