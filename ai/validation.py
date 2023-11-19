import json
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


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
        self.export_pic(self.dates, "Date", self.buys_count, "buys_count", "buys_count.png")
        self.export_pic(self.dates, "Date", self.sells_count, "sells_count", "sells_count.png")

        fig, ax = plt.subplots()
        ax.axis('tight')
        ax.axis('off')
        data = []
        for k, v in self.vars:
            data.append([k, v])
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
        probabilities, bins = np.histogram(buy_decision_probs.cpu().detach().numpy(), bins=100, range=(0, 1))

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


def val(model, codes, files, start_idx, end_idx, total_x, title = "test", metric_acc=None, metric_auc=None, buy_label=None):
    writer = CustomWriter(files[start_idx:end_idx], title)

    balance = 10000.0
    start_balance = balance
    num_op = 1
    v1 = 0.6

    num_days = end_idx - start_idx
    model.eval()

    holds = []  # code, buy_price, target_sell_price, date, prob, hold days
    values_holds = 0.0
    buy_decision_probs, expected_sell_prices = model(total_x)
    buy_decision_probs = buy_decision_probs[-num_days:]
    writer.add_prob_dist(buy_decision_probs)
    metric_acc.reset()
    metric_acc.update(buy_decision_prob.flatten(), buy_label.flatten())
    writer.add_var("acc", metric_acc.compute())
    metric_auc.reset()
    metric_auc.update(buy_decision_prob.flatten(), buy_label.flatten())
    writer.add_var("auc", metric_auc.compute())

    for i, (file) in tqdm(enumerate(files[start_idx:end_idx]), total=num_days):
        price_of_the_day = load_js(file)
        to_remove = []
        values_holds = 0.0
        for hold in holds:
            if hold[0] in price_of_the_day and price_of_the_day[hold[0]][2] >= hold[2]:
                balance += hold[2] * num_op
                to_remove.append(hold)
                writer.add_sell(
                    file,
                    hold[0],
                    hold[2],
                    num_op,
                )
            elif hold[0] in price_of_the_day and hold[5] >= 5:
                balance += price_of_the_day[hold[0]][2] * num_op
                to_remove.append(hold)
                writer.add_sell(
                    file,
                    hold[0],
                    hold[2],
                    num_op,
                )
            elif hold[0] in price_of_the_day:
                values_holds += price_of_the_day[hold[0]][1] * num_op
        # print("163 ", holds)
        for hold in to_remove:
            holds.remove(hold)
        for j in range(len(holds)):
            holds[j][5]+=1
        # print(holds)
        buy_decision_prob = buy_decision_probs[i]
        codes_probs = list(zip(codes, buy_decision_prob.tolist()))
        codes_probs = [(code, prob) for code, prob in codes_probs if code in price_of_the_day and prob>0.65]
        codes_probs = sorted(codes_probs, key=lambda x:-x[1])
        for code, prob in codes_probs[:10]:
            if (num_op * price_of_the_day[code][1] <= balance):
                balance -= num_op * price_of_the_day[code][1]
                holds.append(
                    [
                        code,
                        price_of_the_day[code][1],
                        # expected_sell_price[i].item(),
                        price_of_the_day[code][1] * 1.03 ,
                        file,
                        prob,
                        0,
                    ]
                )
                values_holds += price_of_the_day[code][2] * num_op
                # print(f"Buy {codes[i]} on {price_of_the_day[codes[i]][1]}!")
                writer.add_buy(
                    file,
                    code,
                    price_of_the_day[code][1],
                    num_op,
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
        writer.add_balance(file, balance)
        writer.add_total(file, balance + values_holds)
    print(f"Balance: {balance}")
    values_holds = 0.
    for hold in holds:
        values_holds += hold[1]*num_op
    print(f"Value of holds: {values_holds}")
    print(f"总资产:  {balance + values_holds}")
    print(f"经过 {end_idx - start_idx + 1} 个交易日")
    print(f"总收益率: {(balance + values_holds) / start_balance - 1.0}")
    writer.close()


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
