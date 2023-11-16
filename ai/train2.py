# import pandas as pd
import datetime
import glob
import os
import json

from tqdm import tqdm
import torch
from torch import nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR

# from torch.utils.data import DataLoader
from validation import val
from TinyModel import TradingModel

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"

print(f"Using device: {device}")


def load_js(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_codes_of_date(path):
    codes = [code for code in load_js(path).keys()]
    return codes


def get_date_info(path):
    date = path[14:24]
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return [date.month, date.day, date.weekday()]


def get_codes(start_date_str="19901101", end_date_str="20231109"):
    files = sorted(glob.glob(os.path.join("dataset", "daily", "*.json")))
    print(
        f"Found {len(files)} days, start finding from {start_date_str} to {end_date_str}"
    )
    train_dates = files[4500:6000]
    val_dates = files[6000:7000]
    test_dates = files[7000:-5]
    print(f"Train start: {train_dates[0]}")
    print(f"Train end: {train_dates[-1]}")
    print(f"Val start: {val_dates[0]}")
    print(f"Val end: {val_dates[-1]}")
    print(f"Test start: {test_dates[0]}")
    print(f"Test end: {test_dates[-1]}")
    codes = get_codes_of_date(test_dates[0])
    for file in [train_dates[0], val_dates[0], test_dates[0], test_dates[-1]]:
        codes = list(set(codes).intersection(set(get_codes_of_date(file))))
    print("Total codes: ", len(codes))
    codes = sorted(codes)
    return codes, files, 4500, 6000, 7000, len(files) - 5


def get_data(codes, files, start_idx, end_idx):
    print("Start reading data")
    datas_x = []
    for i in range(start_idx, end_idx):
        data = load_js(files[i])
        row = get_date_info(files[i])
        for code in codes:
            if code not in data:
                row += [0.0, 0.0, 0.0, 0.0]
            else:
                row += data[code][1:5]
        datas_x.append(row)
    datas_x = torch.tensor(datas_x)
    # print(datas_x.shape, datas_x.dtype)  # torch.Size([1500, 5071]) torch.float32
    datas_y_buy = []
    datas_y_price = []
    for i in range(start_idx, end_idx):
        path = files[i]
        path = path.replace("daily", "labels1")
        data = load_js(path)
        row_buy = []
        row_price = []
        for code in codes:
            if code not in data:
                row_buy.append(0)
                row_price.append(0.0)
            else:
                row_buy.append(data[code][0])
                row_price.append(data[code][1])
        datas_y_buy.append(row_buy)
        datas_y_price.append(row_price)
    datas_y_buy = torch.tensor(datas_y_buy, dtype=torch.float32)
    datas_y_price = torch.tensor(datas_y_price)
    print(
        datas_y_buy.shape, datas_y_buy.dtype
    )  # torch.Size([1500, 5071]) torch.float32
    print(
        datas_y_price.shape, datas_y_price.dtype
    )  # torch.Size([1500, 5071]) torch.float32
    return datas_x, datas_y_buy, datas_y_price


def trading_loss_function(
    buy_decision_prob, buy_label, expected_sell_price, price_label
):
    buy_decision_prob = buy_decision_prob.flatten()
    buy_label = buy_label.flatten()
    t = buy_label.shape[0]
    s = torch.sum(buy_label).item()
    weight = buy_label * (1 - 2 * (s / t)) + (s / t)
    buy_decision_loss = F.binary_cross_entropy(
        buy_decision_prob,
        buy_label,
        weight,
    )
    return buy_decision_loss, None

    sell_executed = buy_label == 1

    sell_price_loss = F.mse_loss(expected_sell_price, price_label)
    sell_price_loss = torch.where(
        sell_executed,
        sell_price_loss,
        torch.tensor(0.0, device=device, requires_grad=True),
    )
    sell_price_loss = torch.mean(sell_price_loss)

    return buy_decision_loss, sell_price_loss


def train(model, train_x, buy_label, price_label, optimizer, epoch):
    model.train()
    optimizer.zero_grad()
    buy_decision_prob, expected_sell_price = model(train_x)
    # print("Model output:", buy_decision_prob.shape, expected_sell_price.shape)
    buy_decision_loss, sell_price_loss = trading_loss_function(
        buy_decision_prob, buy_label, expected_sell_price, price_label
    )
    # loss = buy_decision_loss + sell_price_loss
    loss = buy_decision_loss
    loss.backward()
    optimizer.step()
    # print(
    #     f"{epoch} loss: {loss:>7f}, buy_decision_loss:{buy_decision_loss.item():>7f},sell_price_loss:{sell_price_loss.item():>7f}"
    # )
    print(f"{epoch} loss: {loss:>7f}")
    return loss


def main():
    codes, files, train_start, val_start, test_start, test_end = get_codes()
    print(len(codes))
    train_x, train_b, train_p = get_data(codes, files, train_start, val_start)
    val_x, val_b, val_p = get_data(codes, files, val_start, test_start)
    test_x, test_b, test_p = get_data(codes, files, test_start, test_end)
    model = TradingModel(len(codes), train_x.shape[1], 2048).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    scheduler = MultiStepLR(optimizer, milestones=[1000, 5000, 12000], gamma=0.1)
    epoch = 20000
    for i in range(1, epoch + 1):
        loss = train(
            model,
            train_x.to(device),
            train_b.to(device),
            train_p.to(device),
            optimizer,
            i,
        )
        if i % 1000 == 0:
            ckpt_path = os.path.join("ai", "weights", f"{i}_{loss:>3f}.pth")
            print(f"Saving model to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)
            val(
                model,
                codes,
                files,
                val_start,
                test_start,
                torch.cat((train_x, val_x), 0).to(device).clone(),
            )
        scheduler.step()
    val(
        model,
        codes,
        files,
        test_start,
        test_end,
        torch.cat((train_x, val_x, test_x), 0).to(device).clone().to(device),
    )


if __name__ == "__main__":
    main()
