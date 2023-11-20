# from torchvision.transforms import ToTensor
import os
import glob
import json
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torcheval.metrics import BinaryAccuracy, BinaryAUROC
import torch.nn as nn

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"


def load_js(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def get_codes_of_date(path):
    codes = [code for code in load_js(path).keys()]
    return codes


def get_codes(start_date_str, end_date_str):
    files = sorted(glob.glob(os.path.join("dataset", "daily", "*.json")))
    print(
        f"Found {len(files)} days, start finding from {start_date_str} to {end_date_str}"
    )
    files = [
        file
        for file in files
        if os.path.basename(file).split(".")[0] >= start_date_str
        and os.path.basename(file).split(".")[0] <= end_date_str
    ]
    print(f"Found {len(files)} days")
    codes1 = get_codes_of_date(files[0])
    codes2 = get_codes_of_date(files[-6])
    codes = list(set(codes1).intersection(set(codes2)))
    print("Total codes: ", len(codes))
    codes = sorted(codes)
    return codes, files


def get_date_info(path):
    date = path[14:24]
    date = datetime.datetime.strptime(date, "%Y-%m-%d")
    return [date.month, date.day, date.weekday()]


class SlidWindowDataset(Dataset):
    def __init__(
        self,
        start_date_str="2016-11-01",
        end_date_str="2023-10-25",
        windows_size=15,
    ):
        self.window_size = windows_size
        self.codes, files = get_codes(start_date_str, end_date_str)
        x_cache = "x_" + start_date_str + end_date_str + ".pt"
        y_b_cache = "y_b_" + start_date_str + end_date_str + ".pt"
        y_p_cache = "y_p_" + start_date_str + end_date_str + ".pt"
        if (
            os.path.exists(x_cache)
            and os.path.exists(y_b_cache)
            and os.path.exists(y_p_cache)
        ):
            print("Read from cache")
            self.x = torch.load(x_cache)
            self.y_b = torch.load(y_b_cache)
            self.y_p = torch.load(y_p_cache)
            print(torch.sum(self.y_b).item(), self.y_b.shape[0] * self.y_b.shape[1])
            return
        datas_x = []
        for file in files:
            data = load_js(file)
            row = get_date_info(file)
            for code in self.codes:
                if code not in data:
                    row += [0.0, 0.0, 0.0, 0.0]
                else:
                    row += data[code][1:5]
            datas_x.append(row)
        self.x = torch.tensor(datas_x)

        datas_y_buy = []
        datas_y_price = []
        for path in files:
            path = path.replace("daily", "labels2")
            data = load_js(path)
            row_buy = []
            row_price = []
            for code in self.codes:
                if code not in data:
                    row_buy.append(0)
                    row_price.append(0.0)
                else:
                    row_buy.append(data[code][0])
                    row_price.append(data[code][1])
            datas_y_buy.append(row_buy)
            datas_y_price.append(row_price)
        self.y_b = torch.tensor(datas_y_buy, dtype=torch.float32)
        self.y_p = torch.tensor(datas_y_price)
        print(self.x.shape, self.x.dtype)
        print(self.y_b.shape, self.y_b.dtype)
        print(self.y_p.shape, self.y_p.dtype)
        torch.save(self.x, x_cache)
        torch.save(self.y_b, y_b_cache)
        torch.save(self.y_p, y_p_cache)
        print(torch.sum(self.y_b).item(), self.y_b.shape[0] * self.y_b.shape[1])
        print("Data cache saved")

    def __len__(self):
        return self.x.shape[0] - self.window_size + 1

    def __getitem__(self, idx):
        x = self.x[idx : idx + self.window_size]
        y_b = self.y_b[idx + self.window_size - 1]
        y_p = self.y_p[idx + self.window_size - 1]
        return x, y_b, y_p

    def get_num_of_stocks(self):
        return len(self.codes)


def get_slid_window_dataloader(start_date, end_date, ws, bs):
    return DataLoader(
        SlidWindowDataset(
            start_date_str=start_date, end_date_str=end_date, windows_size=ws
        ),
        batch_size=bs,
    )


class TradingModel(nn.Module):
    def __init__(self, num_stocks, input_size, hidden_size=2048, num_layers=1):
        super(TradingModel, self).__init__()
        print(f"Model info: {num_stocks} {input_size} {hidden_size} {num_layers}")
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc_buy = nn.Linear(hidden_size, num_stocks)  # 决定是否购买
        self.fc_price = nn.Linear(hidden_size, num_stocks)  # 决定卖出价格

    def forward(self, x):
        # x: batch_size x sequence_length x input_size
        output, (h_n, c_n) = self.lstm(x)

        # Take the last time step output
        output = output[:, -1, :]

        if not self.training:
            buy_prob = torch.sigmoid(self.fc_buy(output))
        else:
            buy_prob = self.fc_buy(output)

        expected_price = torch.relu(self.fc_price(output))
        return buy_prob, expected_price


def trading_loss_function(
    buy_decision_prob, buy_label, expected_sell_price, price_label
):
    buy_decision_prob = buy_decision_prob.flatten()
    buy_label = buy_label.flatten()
    buy_decision_loss = F.binary_cross_entropy_with_logits(buy_decision_prob, buy_label)
    sell_executed = buy_label == 1.0

    sell_price_loss = F.mse_loss(expected_sell_price, price_label)
    sell_price_loss = torch.where(
        sell_executed,
        sell_price_loss,
        torch.tensor(0.0, device=device, requires_grad=True),
    )
    sell_price_loss = torch.mean(sell_price_loss)

    return buy_decision_loss, sell_price_loss


def main():
    # Train start: dataset\daily\2007-01-08.json
    # Train end: dataset\daily\2015-04-01.json
    train_dataset = SlidWindowDataset("2010-08-11", "2014-11-03")
    metric_acc = BinaryAccuracy()
    metric_auc = BinaryAUROC()
    model = TradingModel(
        train_dataset.get_num_of_stocks(), train_dataset[0][0].shape[1], 4096, 1
    ).to(device)
    model.train()
    data_loader = DataLoader(train_dataset, batch_size=256)
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.0000001)
    scheduler = MultiStepLR(optimizer, milestones=[50000, 10000, 15000], gamma=0.2)
    epoch = 20000
    for i in range(1, epoch + 1):
        metric_acc.reset()
        metric_auc.reset()
        ave_loss = 0.
        ave_loss_b = 0.
        ave_loss_p = 0.
        for step, (x, y_b, y_p) in enumerate(data_loader):
            x = x.to(device)
            y_b = y_b.to(device)
            y_p = y_p.to(device)
            pre_probs, pre_prices = model(x)
            buy_decision_loss, sell_price_loss = trading_loss_function(
                pre_probs, y_b, pre_prices, y_p
            )
            # loss = buy_decision_loss + 0.01 * sell_price_loss
            loss = buy_decision_loss
            loss.backward()
            optimizer.step()
            metric_acc.update(torch.sigmoid(pre_probs.flatten()), y_b.flatten())
            metric_auc.update(torch.sigmoid(pre_probs.flatten()), y_b.flatten())
            ave_loss_b += buy_decision_loss.detach().cpu().item()
            ave_loss_p += sell_price_loss.detach().cpu().item()
        ave_loss_b /= len(data_loader)
        ave_loss_p /= len(data_loader)
        ave_loss = ave_loss_b + ave_loss_p
        scheduler.step()
        print(
            f"epoch: {i}, b_loss:{ave_loss_b}, p_loss:{ave_loss_p} ACC: {metric_acc.compute()} AUC: {metric_auc.compute()}"
        )
        if i % 1000 == 0:
            ckpt_path = os.path.join("st2", f"{i}_{loss:>3f}.pth")
            print(f"Saving model to {ckpt_path}")
            torch.save(model.state_dict(), ckpt_path)
    print("st1 done")


if __name__ == "__main__":
    main()
