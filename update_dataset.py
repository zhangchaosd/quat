import os
import glob
import datetime
import time

from tqdm import tqdm
import akshare as ak
import pandas as pd

save_path = "dataset/daily"

# 保存为csv文件，encoding="utf_8_sig"确保csv文件可以正常显示中文
# spath = os.path.join(save_path, "data.csv")
# df.to_csv(spath, encoding="utf_8_sig", index=False)


def get_today_stocks():
    today = datetime.datetime.now().strftime("%Y%m%d")
    file = glob.glob("dataset/daily/codes_*.csv")
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



def parse(code):
    data = ak.stock_zh_a_hist(
        symbol=code[2:], period="daily", start_date="19690101", end_date="20231109"
    )
    spath = os.path.join("dataset", "by_codes", f"{code}.csv")
    data.to_csv(spath, encoding="utf_8_sig", index=False)


codes = get_today_stocks()

# codes = ["sz301468", "sz301469", "sz301486", "sz301487"]
for code in tqdm(codes):
    while True:
        try:
            parse(code)
        except:
            time.sleep(2)
            continue
        break
# datas = map(parse, codes)
