import torch


def run(x, x_keep, p):
    return (x * x_keep) * (1.0 / (1 - p))
