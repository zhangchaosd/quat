import baostock as bs
import pandas as pd

#### 登陆系统 ####
lg = bs.login()

rs = bs.query_history_k_data_plus(
    "sh.000001",
    # "date,time,code,open,high,low,close,volume,amount,adjustflag",
    "date,open,high,low,close",
    start_date="2015-03-02",
    end_date="2015-03-03",
    frequency="d",
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
print(data_list[:10])

#### 登出系统 ####
bs.logout()
