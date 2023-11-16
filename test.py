import akshare as ak
import os
import json
import glob
import pandas as pd


with open("dataset/daily/2023-11-09.json", "r") as f:
    d = json.load(f)
print(len(d.keys()))

today = "20231110"
file = glob.glob("dataset/daily/codes_*.csv")
stock_list = pd.read_csv(file[0])["代码"]

stocks = list(stock_list)
print(len(stocks))

for code in stocks:
    if code not in d.keys():
        print(code)
