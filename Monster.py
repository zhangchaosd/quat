import torch
from ai.TinyModel import TradingModel


class Monster:
    def __init__(self, place_order_func) -> None:
        self.place_order_func = place_order_func

        model = TradingModel(3189, 7787, 2048)  # TODO
        # model.load_state_dict(torch.load(f"ai/weights/100.pth"))
        model.eval()

    def update(self, time, prices):
        pass
