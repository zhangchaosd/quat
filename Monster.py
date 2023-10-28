import random


class Monster:
    def __init__(self, get_price, buy_func, sel_func) -> None:
        self.get_price = get_price
        self.buy = buy_func
        self.sel = sel_func

        self.balance = 0.0

        self.holds = dict()
        code = "sh.600000"
        self.holds[code] = []
        # code: [buy price, mount, target sell price]

    def update(self):
        code = "sh.600000"
        res, price = self.get_price("sh.600000")
        if not res:
            return res
        if random.random() < 0.01:
            self.holds[code].append([price, 100, price * 1.03])
            self.balance -= price * 100
        new_holds = []
        for order in self.holds[code]:
            if order[2] > price:
                new_holds.append(order)
            else:
                self.balance += price * order[1]
        self.holds[code] = new_holds
        self.last_price = price
        return res

    def show(self):
        print("###############################################")
        print(f"Balance: {self.balance}")
        print(f"Holds({self.last_price}):")
        total = 0.0
        for order in self.holds["sh.600000"]:
            print("sh.600000", order)
            total += self.last_price * order[1]
        print(f"Now holds: {total}")
        print(f"Total: {total + self.balance}")
        print("###############################################")
