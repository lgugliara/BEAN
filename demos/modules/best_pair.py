import math

def best_pair(n: int):
    root = math.sqrt(n)
    a = max([d for d in range(1, int(root)+1) if n % d == 0], default=1)
    return a, n // a