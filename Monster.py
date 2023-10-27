class Monster:
    def __init__(self, buy_func, sel_func) -> None:
        self.buy = buy_func
        self.sel = sel_func

    def update(self, code, price):
        pass
