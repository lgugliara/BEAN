import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

def binary_to_array(val, n_bits):
    """Intero -> array binario di lunghezza n_bits."""
    return np.fromiter(format(val, f'0{n_bits}b'), dtype=np.int8)

def crop_or_pad_center(arr, width):
    """Tronca o pad a larghezza fissa 'width', centrando il contenuto."""
    n = arr.size
    if n == width:
        return arr
    elif n > width:  # tronca
        start = (n - width) // 2
        return arr[start:start + width]
    else:  # pad
        pad_left = (width - n) // 2
        pad_right = width - n - pad_left
        return np.pad(arr, (pad_left, pad_right), constant_values=0)

def binary_mean_anim_truncated(a, b, n_bits=4, steps=20, view_bits=64, interval=600):
    a = max(0.0, min(1 - 1e-12, a))
    b = max(0.0, min(1 - 1e-12, b))

    a_val = int(a * (2**n_bits))
    b_val = int(b * (2**n_bits))

    frames = []
    labels = []
    cur_bits = n_bits

    for step in range(steps):
        n_bits_out = cur_bits * 2

        a_scaled = a_val << cur_bits
        b_scaled = b_val << cur_bits
        prod = (a_scaled * b_scaled) >> (2 * cur_bits)
        prod_bits_arr = binary_to_array(prod, n_bits_out)

        # sempre centrato e di lunghezza fissa
        fixed = crop_or_pad_center(prod_bits_arr, view_bits)
        frames.append(fixed)

        prod_float = prod / (2**n_bits_out)
        labels.append((n_bits_out, prod_float))

        a_val, b_val = b_val, prod
        cur_bits = n_bits_out

    # setup animazione
    fig, ax = plt.subplots(figsize=(10, 3))
    img = ax.imshow(np.zeros((1, view_bits)), aspect='auto', vmin=0, vmax=1)
    ax.set_title("Binary mean (central truncated)")

    def update(i):
        data = np.vstack(frames[:i + 1])  # ora tutte lunghezza = view_bits
        img.set_data(data)
        bits, val = labels[i]
        ax.set_ylabel(f"step {i+1}/{steps}\nQ{bits} ≈ {val:.8f}")
        ax.set_xlim(view_bits/2 - 10, view_bits)
        ax.set_ylim(data.shape[0]/2, -data.shape[0]/2)
        return [img]

    anim = FuncAnimation(fig, update, frames=len(frames),
                         interval=interval, blit=True, repeat=True)
    plt.tight_layout()
    plt.show()

# Esempio
binary_mean_anim_truncated(0.8125, 0.375, n_bits=2, steps=16, view_bits=2048, interval=100)
