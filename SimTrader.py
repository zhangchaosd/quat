import baostock as bs

import pandas as pd

from Trader import Trader
from Monster import Monster


class SimTrader(Trader):
    def __init__(self, start_date, end_date) -> None:
        super().__init__()
        lg = bs.login()
        if lg.error_code != "0":
            print("login respond error_code:" + lg.error_code)
            print("login respond  error_msg:" + lg.error_msg)
        self.monster = Monster(self.place_order)
        self.trading_dates = self.get_trading_dates(start_date, end_date)

        self.orders_waiting = []
        self.orders_success = []
        self.balance = 1000.0
        self.holds = []

    def __del__(self) -> None:
        bs.logout()

    def reset(self):
        pass

    def place_order(self, code, price, buy):
        pass

    def run(self):
        for date in self.trading_dates:
            print("try ", date)
            codes = self.get_codes_of_date(date)
            print(codes)
            times = self.get_times_of_date(date)
            print(times)
            exit()
            prices = dict.fromkeys(times, {})
            print(prices)
            for code in codes:
                for time_price in self.get_prices(code, date):
                    prices[time_price[0]][code] = time_price[1:]
            for time in times:
                # check orders
                self.monster.update(time, prices[time])
            break

    def get_times_of_date(self, date):
        code = self.get_codes_of_date(date)[0]
        times = []
        for time_price in self.get_prices(code, date):
            # print(time_price)
            if time_price[0] not in times:
                times.append(time_price[0])
        print(times)
        times = sorted(times)
        return times

    def get_prices(self, code, date):
        print(code, date)
        rs = bs.query_history_k_data_plus(
            "sz.399658",
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            # "time,open,high,low,close",
            start_date=date,
            end_date=date,
            frequency="5",
            # adjustflag="3",
        )
        if rs.error_code != "0":
            print(f"get_prices error {rs.error_code}")
            print(f"get_prices error {rs.error_msg}")
        data_list = []
        print(rs.error_code)
        while (rs.error_code == "0") & rs.next():
            print("has data")
            data_list.append(rs.get_row_data())
        # print(f"Get {code} prices done")
        # result = pd.DataFrame(data_list, columns=rs.fields)
        print(data_list)
        return data_list

    def get_trading_dates(self, start_date, end_date):
        rs = bs.query_trade_dates(start_date=start_date, end_date=end_date)
        if rs.error_code != "0":
            print("query_trade_dates respond error_code:" + rs.error_code)
            print("query_trade_dates respond  error_msg:" + rs.error_msg)

        trading_dates = []
        while (rs.error_code == "0") & rs.next():
            line = rs.get_row_data()
            if line[1] == "1":
                trading_dates.append(line[0])
        print(
            f"From {start_date} to {end_date} total found {len(trading_dates)} trading dates"
        )
        return trading_dates

    def get_codes_of_date(self, date):
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        codes = list(stock_df["code"])
        # print(codes)
        return codes
