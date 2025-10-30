import numpy as np
import matplotlib.pyplot as plt

# Dati dummy
N_token, N_feature = 8, 3      # (N_tokens, d_head)
A_h = np.random.uniform(-1, 1, (N_token, N_token, N_feature))

fig = plt.figure(figsize=(7,6))
ax = fig.add_subplot(111, projection='3d')
ax.set_title("Stack completo delle feature (una colonna per (i,j))")

# offset per spaziatura
dx = dy = 0.8
ds_min = 0.1  # altezza minima per non sparire

# loop su griglia (i,j)
for i in range(N_token):
    for j in range(N_token):
        values = A_h[i, j]                # vettore (d_head,)
        z = np.arange(N_feature)             # livelli verticali 0..d_head-1
        colors = plt.cm.RdBu((values - values.min()) / (np.ptp(values) + 1e-9))
        scale = np.abs(values) + ds_min  # scala proporzionale al valore
        # bar3d: x,y base, z base, larghezze, altezza, colore
        ax.bar3d(
            np.full(N_feature, i),           # x
            np.full(N_feature, j),           # y
            z,                            # z (livello feature)
            scale, scale, scale,
            color=colors,
            shade=True
        )

ax.set_xlabel("token i")
ax.set_ylabel("token j")
ax.set_zlabel("feature k")
plt.tight_layout()
plt.show()
