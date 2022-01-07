import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn
from torch.utils.data import DataLoader

import config
import hyperparameters
from PatchesDataset import PatchesDataset
from Network import Conv64Features


def set_learning_rate(optimizer, epoch):
    if epoch == hyperparameters.LR_CHANGE_AT_EPOCH:
        for param in optimizer.param_groups:
            param['lr'] = hyperparameters.LR_AFTER_10_EPOCHS


class TrainInfo:
    def __init__(self):
        self.train_loss_per_epoch = np.zeros((hyperparameters.EPOCHS, ))
        self.validation_loss_per_epoch = np.zeros((hyperparameters.EPOCHS, ))

    def set_train_loss(self, epoch, loss):
        self.train_loss_per_epoch[epoch] = loss

    def set_validation_loss(self, epoch, loss):
        self.validation_loss_per_epoch[epoch] = loss

    def save_train_loss(self):
        np.save("train_loss_per_epoch.npy", self.train_loss_per_epoch)

    def save_validation_loss(self):
        np.save("trained_models/grayscale_images/validation_loss_per_epoch.npy", self.validation_loss_per_epoch)


def train(model, optimizer, criterion, device, train_loader, print_every=1000):

    train_info = TrainInfo()
    count_train_global_batches = 0

    for epoch in range(hyperparameters.EPOCHS):
        print(f"epoch {epoch + 1}/{hyperparameters.EPOCHS}")
        set_learning_rate(optimizer, epoch)

        count_batches_in_current_epoch = 0
        loss_sum_in_current_epoch = 0.0

        model.train()
        for anchor_patch, positive_patch, negative_patch in train_loader:
            count_train_global_batches += 1
            count_batches_in_current_epoch += 1

            anchor_patch = anchor_patch.to(device)
            positive_patch = positive_patch.to(device)
            negative_patch = negative_patch.to(device)

            anchor_features = model(anchor_patch)
            positive_features = model(positive_patch)
            negative_features = model(negative_patch)

            optimizer.zero_grad()
            loss = criterion(anchor_features, positive_features, negative_features)
            loss.backward()
            optimizer.step()

            loss_sum_in_current_epoch += loss.item()

            if count_batches_in_current_epoch % print_every == 0:
                print(f"{hyperparameters.BATCH_SIZE * count_batches_in_current_epoch / len(train_dataset):.3f}  "
                      f"running_average_loss_in_current_epoch={loss_sum_in_current_epoch / count_batches_in_current_epoch:.3f} ")

        # izračunaj average loss trenutne epohe
        train_epoch_average_loss = loss_sum_in_current_epoch / count_batches_in_current_epoch
        train_info.set_train_loss(epoch, train_epoch_average_loss)

        torch.save(model, f"trained_model_{epoch}.pth")

    # spremi average loss treninga i validacije u fajlove
    train_info.save_train_loss()


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training is done on: {device}")

    model = Conv64Features(in_channels=1)
    model.to(device)

    train_dataset = PatchesDataset(train=True, grayscale=True)
    train_loader = DataLoader(dataset=train_dataset, batch_size=hyperparameters.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.TripletMarginLoss(margin=hyperparameters.MARGIN)
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparameters.LR)

    train(model, optimizer, criterion, device, train_loader)

