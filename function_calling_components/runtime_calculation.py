import time


class RuntimeCalculation:
    def __init__(self):
        self.start_time = 0.0
        self.end_time = 0.0
        self.runtimes = []

    def start(self):
        self.start_time = time.time()

    def end(self):
        self.end_time = time.time()
        self.runtimes.append(self.end_time - self.start_time)

    def calculate_runtime(self):
        return self.runtimes[-1] if self.runtimes else 0.0

    def calculate_total_runtime(self):
        return sum(self.runtimes)

    def get_runtime_summary(self, pred_filename):
        info = {
            "filename": pred_filename,
            "total_runtime": self.calculate_total_runtime()
        }
        return info
