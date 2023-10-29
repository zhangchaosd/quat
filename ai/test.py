import baostock as bs
import pandas as pd


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

bs.login()

def create_survived_stocks():
    df1 = get_stocks("2010-08-06")
    df2 = get_stocks("2018-09-25")
    df3 = get_stocks("2023-09-25")
    df_merged = pd.merge(pd.merge(df1, df2, on='code'), df3, on='code').iloc[:, :3]
    ts = []
    for code in df_merged["code"]:
        ts.append(code)
    df_merged = df_merged[df_merged['tradeStatus_x'] == '1']  # 去掉停牌的股票
    df_merged.to_csv('train_data/survived_stocks.csv', index=False)
    print(len(ts))
    return ts
codes = create_survived_stocks()
bs.logout()