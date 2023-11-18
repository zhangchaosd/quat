import json
from datetime import datetime
import os
import torch
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter


def get_date_from_file(file):
    return file[14:24]


class CustomWriter:
    def __init__(self, files) -> None:
        current_time = datetime.now().strftime("%Y_%m_%d__%H_%M_%S")
        self.folder = os.path.join("val_results", current_time)
        self.writer = SummaryWriter(self.folder)

        files = sorted([get_date_from_file(file) for file in files])
        self.idxs = [i for i in range(len(files))]
        self.idxs = dict(zip(files, self.idxs))

        self.buys_count = [0] * len(files)
        self.sells_count = [0] * len(files)
        # print(self.idxs)

    def close(self):
        self.writer.close()

    def add_holds(self, holds):
        pass

    def add_balance(self, date, balance):
        idx = self.idxs[get_date_from_file(date)]
        self.writer.add_scalar("Balance", balance, idx)
        self.writer.add_scalar("Buys", self.buys_count[idx], idx)
        self.writer.add_scalar("Sells", self.sells_count[idx], idx)

    def add_total(self, date, balance):
        idx = self.idxs[get_date_from_file(date)]
        self.writer.add_scalar("Total", balance, idx)

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


def load_js(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def val(model, codes, files, start_idx, end_idx, total_x):
    writer = CustomWriter(files[start_idx:end_idx])

    balance = 10000.0
    start_balance = balance
    num_op = 1
    v1 = 0.6

    num_days = end_idx - start_idx
    model.eval()

    holds = []  # code, buy_price, target_sell_price, date, prob
    values_holds = 0.0
    buy_decision_probs, expected_sell_prices = model(total_x)
    buy_decision_probs = buy_decision_probs[-num_days:]
    writer.add_prob_dist(buy_decision_probs)

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
                    codes[i],
                    hold[2],
                    num_op,
                )
            elif hold[0] in price_of_the_day:
                values_holds += price_of_the_day[hold[0]][1] * num_op
        # print("163 ", holds)
        for hold in to_remove:
            holds.remove(hold)
        # print(holds)
        buy_decision_prob = buy_decision_probs[i]
        codes_probs = list(zip(codes, buy_decision_prob.tolist()))
        codes_probs = [(code, prob) for code, prob in codes_probs if code in price_of_the_day and prob>0.6]
        codes_probs = sorted(codes_probs, key=lambda x:-x[1])
        for code, prob in codes_probs[-5:]:
            if (num_op * price_of_the_day[code][1] <= balance):
                balance -= num_op * price_of_the_day[code][1]
                holds.append(
                    [
                        code,
                        price_of_the_day[code][1],
                        # expected_sell_price[i].item(),
                        price_of_the_day[code][1] * 1.06,
                        file,
                        prob,
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
