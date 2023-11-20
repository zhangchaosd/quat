import glob
import os
import json
from TimeMachine import TimeMachine
from SimTrader import SimTrader
from Monster2 import Monster2

# http://baostock.com/
"""

假设最开始一万块钱
验证日期 2015-01-01-2022-12-31
测试日期 2023-01-01-2023-10-19


需要可视化的数据：
验证时间内总收益率
折线图：横轴分别是日，周，月
总收益率：横轴是日

输出log：
买卖时间，数量

"""


def run(start_date, end_date):
    # trader = SimTrader(start_date, end_date)
    # trader.run()
    # trader.show()
    # del trader
    monster = Monster2()
    files = sorted(glob.glob(os.path.join("dataset", "daily", "*.json")))
    idx1 = files.index(os.path.join("dataset", "daily", start_date + ".json"))
    idx2 = files.index(os.path.join("dataset", "daily", end_date + ".json"))
    for path in files[idx1 : idx2 + 1]:
        with open(path, "r") as f:
            data = json.load(f)
        monster.update_daily_data(os.path.basename(path).split(".")[0], data)


def main():
    print("Start validation")
    run("2015-03-02", "2023-01-03")
    print("Validation done")
    print("Start test")
    # run("2023-01-03", "2023-10-28")
    print("Test done")


if __name__ == "__main__":
    main()
