import numpy as np
from modules.best_pair import best_pair       # per disporre subplots

np.random.seed(7)
debug_bits = False
random_input = True

N_tokens = 64
N_features = 3
N_heads = 4
d_head = N_features  # features per head (3D for plotting)
look_feature = 0  # which feature to print

E = None
W_Q, W_K, W_V = None, None, None
Q, K, V = None, None, None
K_T = None
A, S, C, O = None, None, None, None

variances = []
n_col, n_row = best_pair(N_heads)

# ----------------- Debugging bits function -----------------

from modules.printer import prit

def print_input():
    print("\n--- Input, shape:", E.shape)
    if debug_bits:
        prit("I", E[0, look_feature])
def print_weights():
    print("\n--- Weights, shape:", W_Q.shape)
    if debug_bits:
        for h in range(N_heads):
            print(f"HEAD {h}")
            prit("W_Q", W_Q[h, 0, look_feature])
            prit("W_K", W_K[h, 0, look_feature])
            prit("W_V", W_V[h, 0, look_feature])
def print_heads():
    print("\n--- QKV, shape:", Q.shape)
    if debug_bits:
        for h in range(N_heads):
            print(f"HEAD {h}")
            prit("Q", Q[h, 0, look_feature])
            prit("K", K[h, 0, look_feature])
            prit("V", V[h, 0, look_feature])
def print_scores():
    print("\n--- Scores, shape:", S.shape)
    if debug_bits:
        for h in range(N_heads):
            print(f"HEAD {h}")
            prit("S", S[h, 0, 0])
def print_attention():
    print("\n--- Attention, shape:", A.shape)
    if debug_bits:
        for h in range(N_heads):
            print(f"HEAD {h}")
            prit("A", A[h, 0, 0])
def print_curl():
    print("\n--- Curl, shape:", C.shape)
    if debug_bits:
        for h in range(N_heads):
            print(f"HEAD {h}")
            prit("C", C[h, 0, look_feature])
def print_output():
    print("\n--- Output, shape:", O.shape)
    if debug_bits:
        for h in range(N_heads):
            print(f"HEAD {h}")
            prit("O", O[h, 0, look_feature])

def print_all():
    print_input()
    print_weights()
    print_heads()
    print_scores()
    print_attention()
    print_curl()
    print_output()

# ----------------- Part 1a: Initialization (input and weights) -----------------

def generate_input():
    global E
    E = np.random.uniform(-1, 1, (N_tokens, N_features)).astype(np.float32)  # (N_tokens, N_features)
def generate_weights():
    global W_Q, W_K, W_V
    W_Q = np.random.uniform(-1, 1, (N_heads, N_features, d_head)).astype(np.float32)    # (N_heads, N_features, d_head)
    W_K = np.random.uniform(-1, 1, (N_heads, N_features, d_head)).astype(np.float32)
    W_V = np.random.uniform(-1, 1, (N_heads, N_features, d_head)).astype(np.float32)
def generate_heads():
    global Q, K, V, K_T
    Q = np.einsum('nf,hdf->hnd', E, W_Q)    # (N_heads, N_tokens, d_head)
    K = np.einsum('nf,hdf->hnd', E, W_K)
    V = np.einsum('nf,hdf->hnd', E, W_V)
    # Q = E @ W_Q     # (N_heads, N_tokens, d_head)
    # K = E @ W_K
    # V = E @ W_V
    K_T = K.transpose(0, 2, 1)          # (N_heads, d_head, N_tokens)

# ----------------- Part 1b: Forward pass function -----------------

from modules.softmax import softmax
from modules.bin import bean

def generate_scores():
    global S
    S = (Q @ K_T) / np.sqrt(d_head)     # (N_heads, N_tokens, N_tokens)
def step():
    global Q, K, V
    global A, S, C, O

    A = softmax(S)                      # (N_heads, N_tokens, N_tokens)
    C = bean(Q, K)                      # (N_heads, N_tokens, N_tokens, d_head, d_head)
    O = A @ V                           # (N_heads, N_tokens, d_head)
def forward(randomize=False, debug_mode=False):
    global E
    global W_Q, W_K, W_V
    global Q, K, V
    global K_T
    global A, S, C, O

    if randomize:
        generate_input()
        generate_heads()
        generate_scores()
        step()
        
    # C = np.cross(Q[:, :, None, :], K[:, None, :, :], axis=-1)   # (N_heads, N_tokens, N_tokens, d_head)
    # C = wedge(Q, K)                    # (N_heads, N_tokens, N_tokens, d_head, d_head)

    if debug_mode:
        print_all()

    return O, C, A

# ----------------- Part 2a: Animated stepper (token-wise) -----------------

from matplotlib.animation import FuncAnimation
import matplotlib.pyplot as plt

fig, axs = plt.subplots(n_row, n_col, figsize=(6, 6))
plt.suptitle(f"Attention matrices per head (N_tokens={N_tokens}, N_heads={N_heads})", fontsize=12)

def init():
    generate_input()
    generate_weights()
    generate_heads()
    forward(randomize=True, debug_mode=True)
def animate_scalar(frame, randomize=random_input):
    global O, C, A
    
    k = frame % N_tokens
    O, C, A = forward(randomize)
    variances.append(np.var(O, axis=(1,2)))     # (N_heads,)

    for ax, k in zip(axs.flat, range(N_heads)):
        ax.cla()
        ax.imshow(A[k, :, :], aspect='auto', cmap='viridis')
        ax.set_title(f"Head {k+1}", fontsize=7)

    return axs

anim = FuncAnimation(fig, init_func=init, func=animate_scalar, repeat=False, interval=120, blit=False, cache_frame_data=False, save_count=0)

plt.tight_layout()
plt.show()

# ------------------ Part 2b: Plot variance over time -----------------

plt.figure(figsize=(8, 4))
plt.plot(np.vstack(variances))                  # (N_tokens, N_heads)
plt.title(f"Varianza di E nel tempo (N_tokens={N_tokens}, N_heads={N_heads})")
plt.tight_layout()
plt.show()

# ------------------ Part 3: Visualize attention vector field -----------------

# from mpl_toolkits.mplot3d import Axes3D  # serve per bar3d
# 
# # A_h = np.random.uniform(-1, 1, (N_tokens, N_tokens, d_head))
# ds_min = 0.1  # altezza minima per non sparire
# 
# fig, axs = plt.subplots(n_row, n_col, figsize=(6, 6))
# 
# # loop su griglia (i,j)
# def animate_attention_vector(frame):
#     k = frame % N_tokens
#     O, T, A, Q, K, V = multi_head_attention_step()
#     n = range(N_tokens)
#     
#     for i in n:
#         for j in n:
#             values = T[i, j]                # vettore (d_head,)
#             z = np.arange(d_head)             # livelli verticali 0..d_head-1
#             colors = plt.cm.RdBu((values - values.min()) / (np.ptp(values) + 1e-9))
#             scale = np.abs(values) + ds_min  # scala proporzionale al valore
#             # bar3d: x,y base, z base, larghezze, altezza, colore
#             ax.bar3d(
#                 np.full(d_head, i),           # x
#                 np.full(d_head, j),           # y
#                 z,                            # z (livello feature)
#                 scale, scale, scale,
#                 color=colors,
#                 shade=True
#             )
# 
#     ax.set_xlabel("token i")
#     ax.set_ylabel("token j")
#     ax.set_zlabel("feature k")
# 
# anim = FuncAnimation(fig, animate_attention_vector, interval=1000, blit=False, cache_frame_data=False, save_count=0)
# 
# plt.tight_layout()
# plt.show()
