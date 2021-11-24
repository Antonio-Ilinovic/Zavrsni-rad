import torch
from torch import nn
from torch.utils.data import DataLoader

from PatchesDataset import PatchesDataset
from Network import Conv64Features


if __name__ == '__main__':

    hyperparameters = {
        'MAX_DISP': 192,
        'BATCH_SIZE': 128,
        'LR': 0.001,
        'FEATURES': 64,
        'KSIZE': 3,
        'PADDING': 0,
        'STEM_STRIDES': 1,
        'MARGIN': 0.2,
        'EPOCHS': 14,
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training is done on: {device}")

    model = Conv64Features()
    model.to(device)

    patches_dataset = PatchesDataset()
    patches_train_loader = DataLoader(dataset=patches_dataset, batch_size=hyperparameters.get('BATCH_SIZE'), shuffle=True)

    criterion = nn.TripletMarginLoss(margin=hyperparameters.get('margin'))
    optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparameters.get('lr'))

    loss_list = []
    cost_list = []
    BATCH_SIZE = hyperparameters.get('batch_size')


def train(hyperparameters, model, optimizer, criterion, device, dataloader_train):
    for epoch in range(hyperparameters.get('EPOCHS')):
        try:
            # epoch training
            model.train()

            for anchor_patch, positive_patch, negative_patch in dataloader_train:
                # put patches data to device (GPU)
                anchor_patch = anchor_patch.to(device)
                positive_patch = positive_patch.to(device)
                negative_patch = negative_patch.to(device)

                optimizer.zero_grad()

                # forward anchor, positive and negative patches through model
                # and view it as a 2D vector of dimensions [BATCH_SIZE, NUMBER_OF_FEATURES]
                model_forward = lambda patch : model(patch).view(patch.shape[0], -1)
                anchor_model_output = model_forward(anchor_patch)
                positive_model_output = model_forward(positive_patch)
                negative_model_output = model_forward(negative_patch)

                # calculate loss using loss function (torch.TripletMarginLoss)
                loss = criterion(anchor_model_output, positive_model_output, negative_model_output)
                # accumulate gradients
                loss.backward()
                # update optimizer parameters
                optimizer.step()

    torch.save(model, 'trained_model.pth')


for epoch in range(hyperparameters.get('epochs')):
    if epoch == 10:
        optimizer.param_groups[0]['lr'] = 0.0001

    cost = 0.0
    print(f"epoch {epoch+1}/{hyperparameters.get('epochs')}")

    count_batches = 0

    model.train()
    for reference_patch, positive_patch, negative_patch in patches_train_loader:
        count_batches += 1

        reference_patch = reference_patch.to(device)
        positive_patch = positive_patch.to(device)
        negative_patch = negative_patch.to(device)

        optimizer.zero_grad()

        reference_output = model(reference_patch)
        positive_output = model(positive_patch)
        negative_output = model(negative_patch)

        current_batch_size = reference_output.size(0)

        loss = criterion(
            reference_output.view(current_batch_size, -1),
            positive_output.view(current_batch_size, -1),
            negative_output.view(current_batch_size, -1))
        loss.backward()
        optimizer.step()
        #loss_list.append(loss.item())
        cost += loss.item()

        if count_batches % 1000 == 0:
            print(f"{BATCH_SIZE * count_batches / 16000000}  loss={loss.item()}")

    cost_list.append(cost)
    print(f" cost={cost}")
    torch.save(model, f"trained_model_{epoch}.pth")

