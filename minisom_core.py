"""
Pure NumPy SOM — no external dependencies.
"""
import numpy as np
from collections import defaultdict


class MiniSom:
    def __init__(self, x, y, input_len, sigma=1.0, learning_rate=0.5, random_seed=42):
        self._x = x
        self._y = y
        self._input_len = input_len
        self._sigma = sigma
        self._lr = learning_rate
        rng = np.random.RandomState(random_seed)
        self._weights = rng.rand(x, y, input_len)
        norms = np.linalg.norm(self._weights, axis=-1, keepdims=True)
        self._weights /= np.where(norms == 0, 1, norms)

    def random_weights_init(self, data):
        rng = np.random.RandomState(0)
        for i in range(self._x):
            for j in range(self._y):
                self._weights[i, j] = data[rng.randint(len(data))]

    def winner(self, x):
        diff = self._weights - x
        dist = np.linalg.norm(diff, axis=-1)
        return np.unravel_index(dist.argmin(), dist.shape)

    def _gaussian(self, c, sigma):
        cols = np.arange(self._y)
        rows = np.arange(self._x)
        xx, yy = np.meshgrid(cols, rows)
        d = 2 * sigma ** 2
        return np.exp(-((xx - c[1]) ** 2 + (yy - c[0]) ** 2) / d)

    def update(self, x, win, t, max_t):
        eta = self._lr * np.exp(-t / max_t)
        sig = self._sigma * np.exp(-t / max_t)
        g = self._gaussian(win, sig) * eta
        self._weights += g[:, :, np.newaxis] * (x - self._weights)

    def train(self, data, num_iter):
        rng = np.random.RandomState(42)
        for t in range(num_iter):
            x = data[rng.randint(len(data))]
            self.update(x, self.winner(x), t, num_iter)

    def get_weights(self):
        return self._weights

    def distance_map(self):
        um = np.zeros((self._x, self._y))
        for i in range(self._x):
            for j in range(self._y):
                nbrs = []
                if i > 0:        nbrs.append(self._weights[i-1, j])
                if i < self._x-1: nbrs.append(self._weights[i+1, j])
                if j > 0:        nbrs.append(self._weights[i, j-1])
                if j < self._y-1: nbrs.append(self._weights[i, j+1])
                if nbrs:
                    um[i, j] = np.mean([np.linalg.norm(self._weights[i, j] - n) for n in nbrs])
        mx = um.max()
        return um / mx if mx > 0 else um

    def win_map(self, data):
        wm = defaultdict(list)
        for idx, x in enumerate(data):
            wm[self.winner(x)].append(idx)
        return wm

    def quantization_error(self, data):
        return np.mean([np.linalg.norm(x - self._weights[self.winner(x)]) for x in data])
