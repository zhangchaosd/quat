import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# from torchvision import datasets
# from torchvision.transforms import ToTensor

from TinyModel import TradingModel

device = "cuda" if torch.cuda.is_available() else "mps"
print(device)


def get_codes():
    codes = pd.read_csv("train_data/survived_stocks.csv")
    return codes


def get_training_datas():
    print("Start reading")
    x = pd.read_csv(f"train_data/100806_230918_x.csv")
    y = pd.read_csv(f"train_data/100806_230918_y.csv")
    # totensor
    x = x.drop(columns=["date", "date2"])
    y = y.drop(columns=["date", "date2", "month", "day", "weekday"])
    y_buy_decision = y.iloc[:, ::2]
    y_sell_price = y.iloc[:, 1::2]
    x = torch.tensor(x.values).to(torch.float)
    y_buy_decision = torch.tensor(y_buy_decision.values).to(torch.float)
    y_sell_price = torch.tensor(y_sell_price.values).to(torch.float)
    x = x.unsqueeze(1)
    x = x.unsqueeze(1)
    y_buy_decision = y_buy_decision.unsqueeze(1)
    y_sell_price = y_sell_price.unsqueeze(1)
    print(x.shape)
    print(y_buy_decision.shape)
    print(y_sell_price.shape)
    print("Reading done")
    return x, y_buy_decision, y_sell_price


def trading_loss_function(
    buy_decision_prob, actual_buy_decision, expected_sell_price, actual_sell_decision
):
    # 买入决策损失（二元交叉熵）
    buy_decision_loss = nn.BCELoss()(buy_decision_prob, actual_buy_decision)

    # 检查是否有成交的实际价格
    sell_executed = actual_buy_decision == 1.0
    # print(sell_executed)

    sell_decision_loss = nn.MSELoss()(expected_sell_price, actual_sell_decision)
    sell_decision_loss = torch.where(
        sell_executed, sell_decision_loss, torch.tensor(0.0,device=device , requires_grad=True)
    )
    # print(buy_decision_loss, sell_decision_loss)
    sell_decision_loss = torch.mean(sell_decision_loss)

    return buy_decision_loss + sell_decision_loss

    # 如果有成交，则计算盈利；如果没有成交，则为0
    profit = torch.where(
        sell_executed, expected_sell_price - purchase_price, torch.tensor(0.0)
    )

    # 盈利最大化损失
    profit_loss = -torch.mean(profit)

    # 执行可能性损失
    execution_loss = torch.mean(
        (actual_prices - expected_sell_price).pow(2) * (1 - sell_executed)
    )

    # 将买入决策损失、盈利最大化损失和执行可能性的损失结合起来
    combined_loss = buy_decision_loss + profit_loss + execution_loss

    return combined_loss


def main():
    codes = get_codes()
    print(codes.shape[0])
    x, y_buy_decision, y_sell_price = get_training_datas()
    epoch = 10


    model = TradingModel(
        num_products=codes.shape[0], input_size=x.shape[3], hidden_size=128
    ).to(device)
    x = x.to(device)
    y_buy_decision = y_buy_decision.to(device)
    y_sell_price = y_sell_price.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    for i in range(epoch):
        for j in range(x.shape[0]):
            input = x[j].to(device)
            buy_decision_label = y_buy_decision[j].to(device)
            sell_price_label = y_sell_price[j].to(device)

            # Compute prediction error
            buy_prob, expected_price = model(input)
            # print(buy_prob.shape, expected_price.shape)

            loss = trading_loss_function(
                buy_prob, buy_decision_label, expected_price, sell_price_label
            )

            # Backpropagation
            loss.backward(retain_graph=True)
            optimizer.step()
            optimizer.zero_grad()
            print(f"loss: {loss:>7f}")


if __name__ == "__main__":
    main()
