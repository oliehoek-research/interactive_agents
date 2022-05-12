import time

class Stopwatch:
    """A simple timer class for convenient benchmarking"""

    def __init__(self):
        self._started = None
        self._elapsed = 0
    
    def start(self):
        if self._started is None:
            self._started = time.time()

    def restart(self):
        self._elapsed = 0
        self._started = time.time()

    def stop(self):
        stopped = time.time()
        if self._started is not None:
            self._elapsed += stopped - self._started
            self._started = None
    
    def reset(self):
        self._elapsed = 0
        self._started = None

    def elapsed(self):
        return self._elapsed
