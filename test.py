import numpy as np
import matplotlib.pyplot as plt
import math
import torch

u = 0  # 均值μ
sig = math.sqrt(0.05)  # 标准差δ

x = np.array([-3, -2.667, -2.333, -2, -1.667, -1.333, -1, -0.667, -0.333, 0, 0.333, 0.667, 1, 1.333, 1.667, 2, 2.333, 2.667, 3])
x = np.array([-3, -2, -1, 0, 1, 2, 3])
y_sig = np.exp(-(x - u) ** 2 / (2 * sig ** 2)) / (math.sqrt(2 * math.pi) * sig)
print(np.sum(y_sig))
y_sig = y_sig / np.sum(y_sig)
# y_sig = torch.softmax(torch.tensor(y_sig), dim = 0)  # 按列SoftMax,列和为1
# print(y_sig)
print("=" * 20)
print(y_sig)