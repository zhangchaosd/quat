class TimeMachine:
    def __init__(
        self, monster, trader, validation_start=20150101, test_start=20230101
    ) -> None:
        self.monster = monster
        self.trader = trader

    def run(self):
        # run val
        self.trader.show()
        # run test
        self.trader.show()
        pass
