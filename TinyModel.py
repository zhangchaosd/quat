import torch
import torch.nn as nn

class TradingModel(nn.Module):
    def __init__(self, num_products, hidden_size):
        super(TradingModel, self).__init__()
        self.lstm = nn.LSTM(input_size=num_products, hidden_size=hidden_size, batch_first=True)
        self.fc_buy = nn.Linear(hidden_size, num_products)  # 决定是否购买
        self.fc_price = nn.Linear(hidden_size, num_products)  # 决定卖出价格

    def forward(self, x):
        # x: batch_size x sequence_length x num_products
        _, (hidden, _) = self.lstm(x)
        
        # 只使用最后一个时间步的隐藏状态
        hidden = hidden[-1]
        
        # 二分类输出：是否购买（采用sigmoid激活）
        buy_prob = torch.sigmoid(self.fc_buy(hidden))
        
        # 回归输出：期望卖出价格（可以采用ReLU激活以保证价格非负）
        expected_price = torch.relu(self.fc_price(hidden))
        
        return buy_prob, expected_price

# 参数
num_products = 100  # 商品种类数量
hidden_size = 128  # LSTM隐藏层大小

# 实例化模型
model = TradingModel(num_products=num_products, hidden_size=hidden_size)

# 示例输入数据
# 假设我们有10天的数据，每天的数据包含100个商品的价格
sequence_length = 10
batch_size = 1
x_dummy = torch.randn(batch_size, sequence_length, num_products)

# 模型预测
buy_prob, expected_price = model(x_dummy)

print(x_dummy.shape)
print(buy_prob,expected_price)