import torch
from torch import nn
from torch.utils.data import DataLoader

import Utils
import hyperparameters
from PatchesDataset import PatchesDataset
from Training import TrainInfo

if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluation is done on: {device}")

    validation_dataset = PatchesDataset(train=False)
    validation_loader = DataLoader(dataset=validation_dataset, batch_size=hyperparameters.BATCH_SIZE,
                              shuffle=True, num_workers=4, pin_memory=True)

    criterion = nn.TripletMarginLoss(margin=hyperparameters.MARGIN)

    train_info = TrainInfo()

    for epoch in range(hyperparameters.EPOCHS):
        print(f"epoch {epoch + 1}/{hyperparameters.EPOCHS} validation")
        model = Utils.load_model(f'D:/ZAVRSNI/Zavrsni-rad/trained_model_{epoch}.pth')

        count_batches_in_current_epoch = 0
        loss_sum_in_current_epoch = 0.0

        for anchor_patch, positive_patch, negative_patch in validation_loader:
            count_batches_in_current_epoch += 1

            anchor_patch = anchor_patch.to(device)
            positive_patch = positive_patch.to(device)
            negative_patch = negative_patch.to(device)

            anchor_features = model(anchor_patch)
            positive_features = model(positive_patch)
            negative_features = model(negative_patch)

            loss = criterion(anchor_features, positive_features, negative_features)

            loss_sum_in_current_epoch += loss.item()

            if count_batches_in_current_epoch % 1000 == 0:
                print(f"{hyperparameters.BATCH_SIZE * count_batches_in_current_epoch / len(validation_dataset):.3f}  "
                      f"running_average_loss_in_current_epoch={loss_sum_in_current_epoch / count_batches_in_current_epoch:.3f} ")

        # izračunaj average loss trenutne epohe
        validation_epoch_average_loss = loss_sum_in_current_epoch / count_batches_in_current_epoch
        train_info.set_validation_loss(epoch, validation_epoch_average_loss)

    # spremi average loss treninga i validacije u fajlove
    train_info.save_validation_loss()
