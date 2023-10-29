import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader
# from torchvision import datasets
# from torchvision.transforms import ToTensor

from TinyModel import TradingModel


def get_num_codes():
    pd.read_csv("train_data/survived_stocks.csv")
    print(pd.shape[0])
    return pd.shape[0]

def main():
    device = "cuda" if torch.cuda.is_available() else "mps"
    model = TradingModel(num_products=get_num_codes(), hidden_size=128).to(device)
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

if __name__== "__main__":
    main()
