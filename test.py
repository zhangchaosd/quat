import akshare as ak

# 获取东方财富网-沪深京 A 股-实时行情
df = ak.stock_zh_a_spot_em()
# 保存为csv文件，encoding="utf_8_sig"确保csv文件可以正常显示中文
spath = r"./data.csv"
df.to_csv(spath, encoding="utf_8_sig", index=False)

