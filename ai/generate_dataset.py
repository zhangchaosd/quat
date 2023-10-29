import pandas as pd
import baostock as bs
from tqdm import tqdm
import numpy as np

# from datetime import datetime, timedelta


# def is_trading_day(date):
#     # print(date)
#     rs = bs.query_trade_dates(start_date=date, end_date=date)
#     data_list = []
#     while (rs.error_code == "0") & rs.next():
#         data_list.append(rs.get_row_data())
#     # print(date)
#     return data_list[0][1] == "1"


# def get_next_day(today):
#     date_obj = datetime.strptime(today, "%Y-%m-%d")
#     new_date_obj = date_obj + timedelta(days=1)
#     tomorrow = new_date_obj.strftime("%Y-%m-%d")
#     return tomorrow


# def get_next_trading_day(today):
#     tomorrow = get_next_day(today)
#     while not is_trading_day(tomorrow):
#         tomorrow = get_next_day(tomorrow)
#     return tomorrow


# def go_to_future(today, days):
#     while days > 0:
#         today = get_next_trading_day(today)
#         days -= 1
#     return today


# def get_period(date2, date1):
#     # print(date2, date1)
#     ans = 0
#     while date1 != date2:
#         date1 = get_next_day(date1)
#         if is_trading_day(date1):
#             ans += 1
#     return ans


def construct_labels(product_data, code):
    # 设定预期卖出价格的计算窗口
    sell_window = 5

    # 确保数据按日期排序
    product_data = product_data.sort_values("date")

    # 使用rolling和min来计算未来sell_window天内的最低价格
    product_data["min_future_low"] = (
        product_data["high"]
        .shift(-sell_window)
        .rolling(window=sell_window, min_periods=1)
        .min()
    )
    # product_data["target_sell_price"] = product_data["high"] * 1.01
    # print(product_data)
    # 向量化条件逻辑来确定是否买入
    product_data["buy_label"] = np.where(
        product_data["min_future_low"] >= product_data["high"] * 1.01, 1, 0
    )
    # print(product_data)

    # 创建最终的DataFrame，只包含需要的列
    labels_df = product_data[["date", "buy_label"]].copy()
    labels_df["sell_price_label"] = product_data["min_future_low"]
    labels_df["hold_period"] = 0  # 如果需要计算持有期，则在此处进行计算
    labels_df["code"] = code

    # 最终DataFrame的列可能需要根据你的具体需要进行调整
    return labels_df


def get_day_k(code, start_date, end_date):
    #### 获取沪深A股历史K线数据 ####
    # 详细指标参数，参见“历史行情指标参数”章节；“分钟线”参数与“日线”参数不同。“分钟线”不包含指数。
    # 分钟线指标：date,time,code,open,high,low,close,volume,amount,adjustflag
    # 周月线指标：date,code,open,high,low,close,volume,amount,adjustflag,turn,pctChg
    rs = bs.query_history_k_data_plus(
        code,
        "date,code,open,high,low,close,preclose,volume,amount,adjustflag,turn,tradestatus,pctChg,isST",
        start_date=start_date,
        end_date=end_date,
        frequency="d",
        adjustflag="3",
    )
    if rs.error_code != "0":
        print("query_history_k_data_plus respond error_code:" + rs.error_code)
        print("query_history_k_data_plus respond  error_msg:" + rs.error_msg)
    data_list = []
    while (rs.error_code == "0") & rs.next():
        # 获取一条记录，将记录合并在一起
        data_list.append(rs.get_row_data())
    result = pd.DataFrame(data_list, columns=rs.fields)
    result = result.astype(
        {"open": "float", "close": "float", "high": "float", "low": "float"}
    )
    return result


def get_stocks(data):
    stock_rs = bs.query_all_stock(data)
    stock_df = stock_rs.get_data()
    return stock_df
    print(type(stock_df))
    # print(stock_df['1000'])
    data_df = pd.DataFrame()
    ts = []
    for code in stock_df["code"]:
        ts.append(code)
        # k_rs = bs.query_history_k_data_plus(code, "date,code,open,high,low,close", date, date)
        # data_df = data_df.append(k_rs.get_data())
    return ts


def create_survived_stocks():
    df1 = get_stocks("2010-08-06")
    df2 = get_stocks("2018-09-25")
    df3 = get_stocks("2023-09-25")
    df_merged = pd.merge(pd.merge(df1, df2, on="code"), df3, on="code").iloc[:, :3]
    ts = []
    for code in df_merged["code"]:
        ts.append(code)
    df_merged = df_merged[df_merged["tradeStatus_x"] == "1"]  # 去掉停牌的股票
    df_merged.to_csv("train_data/survived_stocks.csv", index=False)
    print(len(ts))
    return ts


def main():
    bs.login()
    codes = create_survived_stocks()
    for code in tqdm(codes):
        # print(code)
        x = get_day_k(code, "2010-08-06", "2023-09-25")
        y = construct_labels(x, code)
        x.to_csv(f"train_data/_{code}_x.csv", index=False)
        y.to_csv(f"train_data/_{code}_y.csv", index=False)
        # break
    bs.logout()


if __name__ == "__main__":
    main()
