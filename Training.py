import torch
from torch import nn
from torch.utils.data import DataLoader

from PatchesDataset import PatchesDataset
from network import Conv64Features


hyperparameters = {
    'max_disp': 192,
    'batch_size': 128,
    'lr': 0.01,
    'features': 64,
    'ksize': 3,
    'padding': 0,
    'stem_strides': 1,
    'margin': 0.2,
    'epochs': 14,
}

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

model = Conv64Features()
model.to(device)

patches_dataset = PatchesDataset()
patches_train_loader = DataLoader(dataset=patches_dataset, batch_size=hyperparameters.get('batch_size'), shuffle=True)


def reference_patch_close_to_positive_patch_more_than_m_from_negative(reference_output, positive_output, negative_output, m=0.5):
    # parametri reference, positive i negative output su izlazi iz modela mreže.
    # Oni su vektori dimenzija (batch_size x 64)

    # L(R, P, Q) = max(0, m + h(R | θ) · h(Q | θ) − h(R | θ) · h(P | θ))
    # R - referentno okno (patch) sa lijeve slike
    # P - "točno" okno na desnoj slici
    # Q - "netočno" okno na desnoj slici
    # h - izlaz modela uz parametre θ.
    # Ovakva funkcija gubitka zahtjeva da je ugrađivanje referentnog okna
    # bliže ugrađivanju pozitivnog primjera za najmanje m.
    # Drugim riječima, funkcija gubitka implicira kako model uči samo za one primjere
    # u kojem je sličnost R i P manja od sličnosti između R i Q za više od m.

    # pošto nema dot-product za batch podataka, koristimo torch.bmm
    # https://discuss.pytorch.org/t/dot-product-batch-wise/9746
    batch_size = reference_output.shape[0]
    # koristim view metodu kako bih pravilno posložio batcheve podataka i mogao napraviti dot product za batch podataka
    dot_reference_positive = torch.sum(reference_output.view(batch_size, -1) * positive_output.view(batch_size, -1), axis=1)
    dot_reference_negative = torch.sum(reference_output.view(batch_size, -1) * negative_output.view(batch_size, -1), axis=1)

    similarity_tensor = m + dot_reference_negative - dot_reference_positive
    # ova linija je ekvivalentna: max(0.0, vrijednost)
    similarity_tensor[similarity_tensor < 0.0] = 0.0
    # vraćam sumu sličnosti, jer loss treba biti skalar. Je li to dobro na ovaj način?
    return similarity_tensor.sum()


#criterion = reference_patch_close_to_positive_patch_more_than_m_from_negative
criterion = nn.TripletMarginLoss(margin=hyperparameters.get('margin'))
optimizer = torch.optim.Adam(params=model.parameters(), lr=hyperparameters.get('lr'))

loss_list = []
cost_list = []
BATCH_SIZE = hyperparameters.get('batch_size')

for epoch in range(hyperparameters.get('epochs')):
    if epoch == 10:
        optimizer.param_groups[0]['lr'] = 0.001

    cost = 0.0
    print(f"epoch {epoch+1}/{hyperparameters.get('epochs')}")

    count_batches = 0

    model.train()
    for reference_patch, positive_patch, negative_patch in patches_train_loader:
        # ove linije su za printanje napretka epohe
        if count_batches % 10 == 0:
            print(BATCH_SIZE*count_batches/16000000)
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
        loss_list.append(loss.item())
        cost += loss.item()

    cost_list.append(cost)
    print(f" cost={cost}")
    torch.save(model, f"trained_model_{epoch}.pth")

# spremam istrenirani model za kasniju upotrebu
torch.save(model, 'trained_model.pth')
