class Monster:
    def __init__(self, balance=10000.0) -> None:
        self.balance = balance

    def update(self, price):
        pass

    def show(self):
        print(self.balance)
