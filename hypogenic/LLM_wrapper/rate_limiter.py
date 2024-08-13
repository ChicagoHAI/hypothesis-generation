import threading
import time
import logging

logger = logging.getLogger(name="RateLimiter")


class RateLimiter:
    def __init__(self, min_backoff=1.0, max_backoff=60.0):
        self.min_backoff = self.backoff_time = min_backoff
        self.max_backoff = max_backoff
        self.lock = threading.Lock()

    def reset(self):
        with self.lock:
            self.backoff_time = self.min_backoff

    def backoff(self):
        logger.warning(f"Backing off for {self.backoff_time:.1f} seconds")
        time.sleep(self.backoff_time)
        with self.lock:
            self.backoff_time = min(self.backoff_time * 2, self.max_backoff)
        logger.debug(f"Setting backoff time to {self.backoff_time:.1f} seconds")

    def add_event(self):
        with self.lock:
            self.backoff_time = max(self.min_backoff, self.backoff_time * 0.75)
