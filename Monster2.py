import torch
from ai.TinyModel import TradingModel


class Monster2:
    def __init__(self) -> None:
        model = TradingModel(3189, 7787, 2048)  # TODO
        # model.load_state_dict(torch.load(f"ai/weights/100.pth"))
        model.eval()

    def update(self, time, prices):
        print("In monster: ", time, prices)
        pass

    def update_daily_data(self, date_str, price_of_the_day):
        print(date_str)
        pass
