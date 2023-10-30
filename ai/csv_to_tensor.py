import pandas as pd
import torch

from functools import reduce


def get_codes():
    codes = pd.read_csv("train_data/survived_stocks.csv")
    return codes


def get_training_datas(codes):
    print("Start reading")

    codes = codes["code"]
    codes = sorted(codes)
    dates = pd.read_csv(f"train_data/_{codes[0]}_x.csv")[["date"]]
    dates["date2"] = pd.to_datetime(dates["date"])

    # 提取月、日和星期信息，并添加为新列（使用数字表示）
    dates["month"] = dates["date2"].dt.month
    dates["day"] = dates["date2"].dt.day
    dates["weekday"] = dates["date2"].dt.dayofweek  # 星期几，0表示周一，6表示周日
    datas_x = [dates]
    datas_y = [dates]
    for code in codes:
        x = pd.read_csv(f"train_data/_{code}_x.csv")[
            ["date", "open", "close", "high", "low"]
        ]
        x.columns = [col + f"_{code}" if col != "date" else col for col in x.columns]
        datas_x.append(x)

        y = pd.read_csv(f"train_data/_{code}_y.csv")[
            ["date", "buy_label", "sell_price_label"]
        ]
        y.columns = [col + f"_{code}" if col != "date" else col for col in y.columns]
        datas_y.append(y)
    merged_x = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"), datas_x
    )
    merged_x.fillna(0, inplace=True)
    merged_x = merged_x.drop(merged_x.tail(5).index)
    merged_y = reduce(
        lambda left, right: pd.merge(left, right, on="date", how="outer"), datas_y
    )
    merged_y = merged_y.drop(merged_y.tail(5).index)
    print("Reading done")
    merged_x.to_csv("train_data/100806_230918_x.csv", index=False)
    merged_y.to_csv("train_data/100806_230918_y.csv", index=False)
    print("Saved")


def save(data):
    torch.save(data, "train_data/_tensor_x.pt")


codes = get_codes()
get_training_datas(codes)
