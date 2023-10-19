class TimeMachine:
    def __init__(self, monster, validation_start=20150101, test_start=20230101) -> None:
        self.monster = monster

    def run(self):
        # run val
        self.monster.show()
        # run test
        self.monster.show()
        pass
