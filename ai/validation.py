import json
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm


def get_date_from_file(file):
    return file[14:24]


class CustomWriter:
    def __init__(self, files, title) -> None:
        current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S_")
        self.folder = os.path.join("val_results", current_time + title)
        os.mkdir(self.folder)

        self.dates = sorted([get_date_from_file(file) for file in files])
        self.idxs = [i for i in range(len(self.dates))]
        self.idxs = dict(zip(self.dates, self.idxs))

        self.buys_count = [0] * len(self.dates)
        self.sells_count = [0] * len(self.dates)

        self.balances = []
        self.totals = []

        self.vars = dict()

    def export_pic(self, x_data, x_label, y_data, y_label, save_name):
        plt.figure(figsize=(10, 6))
        plt.plot(x_data, y_data)
        plt.gca().xaxis.set_major_locator(ticker.AutoLocator())
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(y_label)
        plt.grid(True)
        plt.savefig(os.path.join(self.folder, save_name))
        plt.close()

    def close(self):
        self.export_pic(self.dates, "Date", self.totals, "Total", "Total.png")
        self.export_pic(self.dates, "Date", self.balances, "Balance", "Balance.png")
        self.export_pic(
            self.dates, "Date", self.buys_count, "buys_count", "buys_count.png"
        )
        self.export_pic(
            self.dates, "Date", self.sells_count, "sells_count", "sells_count.png"
        )

        fig, ax = plt.subplots()
        ax.axis("tight")
        ax.axis("off")
        data = []
        for k, v in self.vars.items():
            data.append([k, v.item()])
        table = ax.table(cellText=data)
        plt.savefig(os.path.join(self.folder, "vars.png"))
        plt.close()

    def add_holds(self, holds):
        pass

    def add_var(self, k, v):
        self.vars[k] = v

    def add_balance(self, date, balance):
        self.balances.append(balance)

    def add_total(self, date, total):
        self.totals.append(total)

    def add_buy(self, date, code, price, count=100):
        self.buys_count[self.idxs[get_date_from_file(date)]] += 1

    def add_sell(self, date, code, price, count=100):
        self.sells_count[self.idxs[get_date_from_file(date)]] += 1

    def add_prob_dist(self, buy_decision_probs):
        # 计算概率的分布
        buy_decision_probs = buy_decision_probs.flatten()
        probabilities, bins = np.histogram(
            buy_decision_probs.cpu().detach().numpy(), bins=100, range=(0, 1)
        )

        # 绘制直方图
        plt.figure(figsize=(10, 6))
        plt.bar(
            bins[:-1],
            probabilities,
            width=bins[1] - bins[0],
            color="skyblue",
            edgecolor="black",
        )
        plt.xlabel("Probability")
        plt.ylabel("Frequency")
        plt.title("Probability Distribution of Elements in the Tensor")
        plt.grid(True)
        plt.savefig(os.path.join(self.folder, "prob_dist.png"))
        # plt.show()
        plt.close()


def load_js(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def val(
    model,
    codes,
    files,
    start_idx,
    end_idx,
    total_x,
    title="test",
    metric_acc=None,
    metric_auc=None,
    buy_label=None,
    fc1=None,
    fc2=None,
):
    writer = CustomWriter(files[start_idx:end_idx], title)

    balance = 10000.0
    start_balance = balance
    v1 = 0.6

    num_days = end_idx - start_idx
    model.eval()

    holds = dict()
    values_holds = 0.0
    buy_decision_probs, expected_sell_prices = model(total_x)
    buy_decision_probs = buy_decision_probs[-num_days:]
    writer.add_prob_dist(buy_decision_probs)
    metric_acc.reset()
    metric_acc.update(buy_decision_probs.flatten(), buy_label.flatten())
    writer.add_var("acc", metric_acc.compute())
    metric_auc.reset()
    metric_auc.update(buy_decision_probs.flatten(), buy_label.flatten())
    writer.add_var("auc", metric_auc.compute())

    for i, (file) in tqdm(enumerate(files[start_idx:end_idx]), total=num_days):
        price_of_the_day = load_js(file)
        values_holds = 0.0
        buy_decision_prob = buy_decision_probs[i]
        holds, balance = sell_core(price_of_the_day, balance, holds, writer, file, fc2)
        codes_probs = list(zip(codes, buy_decision_prob.tolist()))
        want_codes = fc1(codes_probs, price_of_the_day, balance)
        holds, balance = buy_core(
            want_codes, price_of_the_day, balance, holds, writer, file
        )
        # expected_sell_price = expected_sell_prices[i]
        # for i in range(len(buy_decision_prob)):
        #     if (
        #         buy_decision_prob[i] > v1
        #         and codes[i] in price_of_the_day
        #         and num_op * price_of_the_day[codes[i]][1] <= balance
        #     ):
        #         balance -= num_op * price_of_the_day[codes[i]][1]
        #         holds.append(
        #             [
        #                 codes[i],
        #                 price_of_the_day[codes[i]][1],
        #                 # expected_sell_price[i].item(),
        #                 price_of_the_day[codes[i]][1] * 1.03,
        #                 file,
        #                 buy_decision_prob[i].item(),
        #             ]
        #         )
        #         values_holds += price_of_the_day[codes[i]][2] * num_op
        #         # print(f"Buy {codes[i]} on {price_of_the_day[codes[i]][1]}!")
        #         writer.add_buy(
        #             file,
        #             codes[i],
        #             price_of_the_day[codes[i]][1],
        #             num_op,
        #         )
        values_holds = cal_holds(holds)
        writer.add_balance(file, balance)
        writer.add_total(file, balance + values_holds)
    print(f"Balance: {balance}")
    print(f"Value of holds: {values_holds}")
    print(f"总资产:  {balance + values_holds}")
    print(f"经过 {end_idx - start_idx + 1} 个交易日")
    print(f"总收益率: {(balance + values_holds) / start_balance - 1.0}")
    writer.close()


def cal_holds(holds):
    res = 0.0
    for code in holds:
        res += holds[code][5] * holds[code][1]
    return res


# files = [
#     "dataset/daily/1990-12-19.json",
#     "dataset/daily/1990-12-20.json",
#     "dataset/daily/1990-12-21.json",
#     "dataset/daily/1990-12-24.json",
#     "dataset/daily/1990-12-25.json",
#     "dataset/daily/1990-12-26.json",
#     "dataset/daily/1990-12-27.json",
#     "dataset/daily/1990-12-28.json",
#     "dataset/daily/1990-12-31.json",
#     "dataset/daily/1991-01-02.json",
# ]
# writer = CustomWriter(files)


def sell_core(price_of_the_day, balance, holds, writer, date, fc2):
    to_sell = fc2(price_of_the_day, holds)
    for code, sell_price in to_sell:
        balance += sell_price * holds[code][1]
        writer.add_sell(
            date,
            code,
            sell_price,
            holds[code][1],
        )
    for code, _ in to_sell:
        holds.pop(code)
    return holds, balance


def buy_core(want_codes, price_of_the_day, balance, holds, writer, date):
    for code in holds:
        holds[code][4] += 1
        if code in price_of_the_day:
            holds[code][5] = price_of_the_day[code][2]
    for code, num_op, prob, alpha in want_codes:
        balance -= num_op * price_of_the_day[code][1]
        if code in holds:
            buy_price, mount, expect_p, hold_prob, days, newest_p = holds[code]
            new_buy_price = buy_price * mount + num_op * price_of_the_day[code][1]
            mount += num_op
            new_buy_price /= mount
            holds[code] = [
                new_buy_price,  # buy price
                mount,  # mount
                price_of_the_day[code][1] * alpha,  # expect sell price
                prob,  # prob
                0,  # days
                price_of_the_day[code][2],  # newest price
            ]

        else:
            holds[code] = [
                price_of_the_day[code][1],  # buy price
                num_op,  # mount
                price_of_the_day[code][1] * alpha,  # expect sell price
                prob,  # prob
                0,  # days
                price_of_the_day[code][2],  # newest price
            ]
        # values_holds += price_of_the_day[code][2] * num_op
        # print(f"Buy {codes[i]} on {price_of_the_day[codes[i]][1]}!")
        writer.add_buy(
            date,
            code,
            price_of_the_day[code][1],
            num_op,
        )
    return holds, balance


# buy_price, mount, expect_sell_price, prob, days
def decision_sell(price_of_the_day, holds):
    to_sell = []
    for code in holds:
        if code in price_of_the_day and price_of_the_day[code][3] >= holds[code][2]:
            sell_price = holds[code][2]
            to_sell.append([code, sell_price])
        elif code in price_of_the_day and holds[code][4] >= 10:
            sell_price = holds[code][2] = price_of_the_day[code][1]
            to_sell.append([code, sell_price])
    return to_sell


def decision_buy(codes_probs, price_of_the_day, balance):
    num_op = 1
    want_buy = []
    codes_probs = [
        (code, prob)
        for code, prob in codes_probs
        if code in price_of_the_day and prob > 0.65
    ]
    codes_probs = sorted(codes_probs, key=lambda x: -x[1])
    for code, prob in codes_probs:
        num_op = int(prob * 10)
        if num_op * price_of_the_day[code][1] <= balance:
            balance -= num_op * price_of_the_day[code][1]
            want_buy.append(
                [
                    code,
                    num_op,
                    prob,
                    1.02,
                ]
            )
    return want_buy


if __name__ == "__main__":
    ckpts = [
        "2000_0.359695.pth",
        "3000_0.328994.pth",
        "4000_0.298951.pth",
        "5000_0.270292.pth",
        "6000_0.260832.pth",
        "7000_0.249412.pth",
        "8000_0.237103.pth",
        "9000_0.225180.pth",
    ]
    from train2 import get_codes, get_data

    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    from torcheval.metrics import BinaryAccuracy, BinaryAUROC

    codes, files, train_start, val_start, test_start, test_end = get_codes()
    print(len(codes))
    train_x, train_b, train_p = get_data(codes, files, train_start, val_start, "train")
    val_x, val_b, val_p = get_data(codes, files, val_start, test_start, "val")
    test_x, test_b, test_p = get_data(codes, files, test_start, test_end, "test")
    metric_acc = BinaryAccuracy()
    metric_auc = BinaryAUROC()
    from TinyModel import TradingModel

    model = TradingModel(len(codes), train_x.shape[1], 4096).to(device)
    only_test = True
    for ckpt_path in ckpts:
        model.load_state_dict(torch.load(os.path.join("ai", "weights", ckpt_path)))
        val(
            model,
            codes,
            files,
            val_start,
            test_start,
            torch.cat((train_x, val_x), 0).to(device).clone(),
            os.path.basename(ckpt_path),
            metric_acc,
            metric_auc,
            val_b,
            decision_buy,
            decision_sell,
        )
        val(
            model,
            codes,
            files,
            val_start,
            test_end,
            torch.cat((train_x, val_x, test_x), 0).to(device).clone(),
            os.path.basename(ckpt_path),
            metric_acc,
            metric_auc,
            torch.cat((val_b, test_b), 0).to(device).clone(),
            decision_buy,
            decision_sell,
        )
