from contextlib import ContextDecorator
import time

class ExecutionTimer(ContextDecorator):
    def __enter__(self):
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end_time = time.time()
        self.execution_time = self.end_time - self.start_time

    def get_execution_time(self):
        return time.time() - self.start_time