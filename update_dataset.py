import os
import glob
import json
import time
from datetime import datetime, timedelta
from tqdm import tqdm
import akshare as ak
import pandas as pd

import numpy as np


def get_today_stocks():
    today = datetime.now().strftime("%Y%m%d")
    # file = glob.glob("dataset/daily/codes_*.csv")
    file = glob.glob(os.path.join("dataset", "daily", "codes_*.csv"))
    if len(file) != 0:
        last_date = os.path.basename(file[0])[6:14]
        if last_date == today:
            stock_list = pd.read_csv(file[0])["代码"]
            return stock_list
        os.remove(file[0])
    print("Updating codes...")
    stock_list = ak.stock_zh_a_spot()["代码"]
    spath = os.path.join("dataset", "daily", "codes_" + today + ".csv")
    stock_list.to_csv(spath, encoding="utf_8_sig", index=False)
    return stock_list


def save_daily_stock_data_to_csv(code):
    data = ak.stock_zh_a_hist(
        symbol=code[2:], period="daily", start_date="19690101", end_date="20231109"
    )
    spath = os.path.join("dataset", "by_codes", f"{code}.csv")
    data.to_csv(spath, encoding="utf_8_sig", index=False)


def save_stocks_by_code(codes):
    for code in tqdm(codes):
        while True:
            try:
                save_daily_stock_data_to_csv(code)
            except:
                time.sleep(2)
                continue
            break


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)


def fc_core(codes, datas, start_date_str="19901101", end_date_str="20231109"):
    iters = [0 for i in range(len(codes))]

    # Convert strings to datetime objects
    start_date = datetime.strptime(start_date_str, "%Y%m%d")
    end_date = datetime.strptime(end_date_str, "%Y%m%d")

    # Use a while loop to iterate from start_date to end_date
    current_date = start_date
    while current_date <= end_date:
        # Append current_date to the list
        date = current_date.strftime("%Y-%m-%d")
        print(date)
        data_today = {}
        for i in range(len(codes)):
            if iters[i] not in datas[i].index:
                continue
            row_data = datas[i].loc[iters[i]]
            if row_data["日期"] != date:
                continue
            data_today[codes[i]] = list(row_data)
            iters[i] += 1
        if data_today != {}:
            filename = os.path.join("dataset", "daily", f"{date}.json")
            with open(filename, "w") as f:
                json.dump(data_today, f, cls=NpEncoder)

        # Move to the next day
        current_date += timedelta(days=1)


def save_stocks_by_date(codes, start_date_str="19901101", end_date_str="20231109"):
    datas = []
    for code in tqdm(codes):
        print(code)
        if os.path.getsize(os.path.join("dataset", "by_codes", f"{code}.csv")) > 5:
            datas.append(pd.read_csv(os.path.join("dataset", "by_codes", f"{code}.csv")))
        else:
            datas.append(pd.DataFrame())
    print("read done")
    fc_core(codes, datas, start_date_str, end_date_str)


def rm_(d):
    return d[:4] + d[5:7] + d[8:10]


def update(codes):
    today = datetime.now().strftime("%Y-%m-%d")
    file = glob.glob(os.path.join("dataset", "daily", "*.json"))
    file = sorted(file)
    if len(file) != 0:
        last_date = os.path.basename(file[-1])[:10]
        if last_date == today:
            print("Up to date")
            return
        print(f"Try update {last_date} to {today}")
        datas = []
        for code in codes:
            while True:
                try:
                    data = ak.stock_zh_a_hist(
                        symbol=code[2:],
                        period="daily",
                        start_date=rm_(last_date),
                        end_date=rm_(today),
                    )
                    datas.append(data)
                except:
                    time.sleep(2)
                    continue
                break
        fc_core(codes, datas, rm_(last_date), rm_(today))


def construct_labels1(folder="labels1"):
    data_path = os.path.join("dataset", "daily")
    label1_path = os.path.join("dataset", folder)
    files = glob.glob(os.path.join(data_path, "*.json"))
    files = sorted(files)
    datas = []
    print("Start loading daily datas")
    for file in tqdm(files):
        with open(file, "r") as f:
            datas.append(json.load(f))
    window_size = 5
    print("Start construct label1")
    for i in tqdm(range(len(files) - window_size)):
        codes = list(datas[i].keys())
        label = {}
        for code in codes:
            buy = 0
            target_price = datas[i][code][3] * 1.03  # max
            for j in range(1, window_size + 1):
                if (code in datas[i + j]
                    and datas[i + j][code][1] >= target_price
                ):
                    buy = 1
                    target_price = datas[i + j][code][1]  # open
            label[code] = [buy, target_price if buy else 0.0]
        if label != {}:
            with open(os.path.join(label1_path, os.path.basename(files[i])), "w") as f:
                json.dump(label, f)

def get_label1_datas():
    pass


codes = get_today_stocks()
save_stocks_by_code(codes)
save_stocks_by_date(codes)
# update(codes)
construct_labels1()
