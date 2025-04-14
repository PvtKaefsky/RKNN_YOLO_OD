import threading
import torch
from queue import Queue
from typing import List, Tuple


class TensorRTPool:
    def __init__(self, model_class, engine_path: str, num_workers: int = 2):
        self.queue = Queue()
        self.lock = threading.Lock()
        self.models = [model_class(engine_path) for _ in range(num_workers)]
        for i in range(num_workers):
            self.queue.put(i)

    def infer(self, image: torch.Tensor) -> List:
        idx = self.queue.get()
        image = torch.from_numpy(image)
        try:
            model = self.models[idx]
            output = model(image)
        finally:
            self.queue.put(idx)
        return output
