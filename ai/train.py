import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# from torchvision import datasets
# from torchvision.transforms import ToTensor

from TinyModel import TradingModel


def get_codes():
    codes = pd.read_csv("train_data/survived_stocks.csv")
    return codes


def get_training_datas(codes):
    print("Start reading")
    datas_x = dict()
    datas_y = dict()
    for code in codes["code"]:
        datas_x[code] = pd.read_csv(f"train_data/_{code}_x.csv")
        datas_y[code] = pd.read_csv(f"train_data/_{code}_y.csv")
    print("Reading done")
    pass


def main():
    codes = get_codes()
    datas = get_training_datas(codes)
    return
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = TradingModel(num_products=codes.shape[0], hidden_size=128).to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    datas = []  # todo
    for X, y in enumerate(datas):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        print(f"loss: {loss:>7f}")


if __name__ == "__main__":
    main()
