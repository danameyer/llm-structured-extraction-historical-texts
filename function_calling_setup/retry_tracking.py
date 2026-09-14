MAX_ATTEMPTS = 3


class RetryTracker:

    def __init__(self):
        self.attempts = 0
        self.success = False

    def reset(self):
        self.attempts = 0
        self.success = False

    def record_attempt(self):
        self.attempts += 1

    def mark_success(self):
        self.success = True

    def get_summary(self, filename: str) -> dict:
        return {
            "filename": filename,
            "attempts": self.attempts,
            "retries": max(self.attempts - 1, 0),
            "success": self.success,
            "max_attempts": MAX_ATTEMPTS,
        }