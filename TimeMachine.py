from datetime import date, timedelta, datetime


class TimeMachine:
    def __init__(
        self, monster, trader, validation_start="2015-03-02", test_start="2023-01-03"
    ) -> None:
        self.monster = monster
        self.trader = trader

        self.train_start = datetime.strptime("2008-08-08", "%Y-%m-%d")
        self.train_start = "2008-08-08"
        self.validation_start = datetime.strptime(validation_start, "%Y-%m-%d")
        self.validation_start = validation_start
        self.test_start = datetime.strptime(test_start, "%Y-%m-%d")
        self.test_start = test_start
        self.test_end = datetime.strptime("2023-10-28", "%Y-%m-%d")
        self.test_end = "2023-10-28"

    def run(self):
        self.enumerate_days(self.validation_start, self.test_start)
        self.enumerate_days(self.test_start, self.test_end)

    def enumerate_days(self, start_date, end_date):
        self.trader.reset()

        prices = self.trader.get_price("sh.600000", start_date, end_date)
        if len(prices) > 0:
            for line in prices:
                # date time code open high low close volume amount adjustflag
                self.monster.update(line[2], line[3])
        else:
            print(f"{start_date}-{end_date} get no data!")
        self.trader.show()
