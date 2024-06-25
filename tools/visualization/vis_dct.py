import numpy as np
import matplotlib.pyplot as plt


def build_filter(freq: int, size: int, t_length):
    pos = np.arange(t_length) / t_length * size
    dct_filter = np.cos(np.pi * freq * (pos + 0.5) / size)
    if freq == 0:
        return dct_filter * np.sqrt(1 / size)
    else:
        return dct_filter * np.sqrt(2 / size)


def vis_dct():
    # 可视化dct变换，包含多个频率分量
    t = np.linspace(0, 2, 500, endpoint=False)
    for freq in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        b = build_filter(freq[0], 2, 500) * build_filter(freq[1], 2, 500)
        plt.figure(figsize=(6, 6))
        plt.plot(t, b)
        plt.xlim(0, 2)  # 限制x轴范围为[0, 2]
        plt.ylim(-1, 1)  # 限制y轴范围为[-1, 1]
        plt.axis('off')
        plt.savefig(f'./dct_pic/freq-{freq[0]}-{freq[1]}')
        plt.show()


if __name__ == '__main__':
    vis_dct()
