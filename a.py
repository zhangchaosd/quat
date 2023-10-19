from Monster import Monster
from TimeMachine import TimeMachine

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


def main():
    print("main")
    monster = Monster()
    tm = TimeMachine(monster)
    tm.run()
    print("Done")


if __name__ == "__main__":
    main()
