from Monster import Monster

import baostock as bs
from datetime import datetime, timedelta

import pandas as pd

from Trader import Trader


def get_stocks():
    ts = pd.read_csv("train_data/survived_stocks.csv")["code"]
    # print(len(ts))
    # print(ts)
    return ts


class SimTrader(Trader):
    def __init__(self, start_date, end_date) -> None:
        super().__init__()
        self.monster = Monster(self.place_order)
        self.start_date = start_date
        self.end_date = end_date
        # codes
        self.codes = get_stocks()
        print(self.codes[:10])
        lg = bs.login()
        if lg.error_code != "0":
            print("login respond error_code:" + lg.error_code)
            print("login respond  error_msg:" + lg.error_msg)

        self.prices = {}
        for code in self.codes[:10]:
            self.prices[code] = self.get_prices(code)
        print(self.prices)

    def __del__(self) -> None:
        bs.logout()

    def reset(self):
        pass

    def place_order(self, code, price, buy):
        pass

    def run(self):
        # Convert strings to datetime objects
        start_date = datetime.strptime(self.start_date_str, "%Y-%m-%d")
        end_date = datetime.strptime(self.end_date_str, "%Y-%m-%d")

        # Initialize a list to hold all dates
        date_list = []

        # Use a while loop to iterate from start_date to end_date
        current_date = start_date
        while current_date <= end_date:
            # Append current_date to the list
            date_list.append(current_date.strftime("%Y-%m-%d"))
            # Move to the next day
            current_date += timedelta(days=1)

    def get_prices(self, code):
        # print(f"Now try to get prices: {code}, {self.start_date}, {self.end_date}")
        rs = bs.query_history_k_data_plus(
            code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=self.start_date,
            end_date=self.end_date,
            frequency="d",
            # adjustflag="3",
        )
        data_list = []
        print(rs.error_code)
        while (rs.error_code == "0") & rs.next():
            data_list.append(rs.get_row_data())
        # print(f"Get {code} prices done")
        result = pd.DataFrame(data_list, columns=rs.fields)
        return result
