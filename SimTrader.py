import baostock as bs

# import pandas as pd

from Trader import Trader


class SimTrader(Trader):
    def __init__(self, start_date, end_date) -> None:
        super().__init__()
        self.start_date = start_date
        self.end_date = end_date
        lg = bs.login()
        if lg.error_code != "0":
            print("login respond error_code:" + lg.error_code)
            print("login respond  error_msg:" + lg.error_msg)

        code = "sh.600000"
        print(f"Now try to get prices: {code}, {start_date}, {end_date}")
        rs = bs.query_history_k_data_plus(
            code,
            "date,time,code,open,high,low,close,volume,amount,adjustflag",
            start_date=start_date,
            end_date=end_date,
            frequency="5",
            # adjustflag="3",
        )
        print("Get prices done")
        self.data_lists = dict()
        self.data_lists[code] = []
        self.iters = dict()
        self.iters[code] = 0
        while (rs.error_code == "0") & rs.next():
            # 获取一条记录，将记录合并在一起
            self.data_lists[code].append(rs.get_row_data())

    def __del__(self) -> None:
        bs.logout()

    def get_price(self, code):
        if self.iters[code] >= len(self.data_lists[code]):
            return False, 0.0
        price = self.data_lists[code][self.iters[code]][3]
        self.iters[code] += 1
        return True, eval(price)

    def reset(self):
        pass
