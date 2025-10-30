import struct
import numpy as np

def bean(a, b, n_bits=16):
    """
    Media binaria normalizzata tra due numeri (o array) in [0,1).
    Simula il comportamento Qn→Q2n→Qn: convergenza discreta.
    """
    # 1️⃣ conversione in rappresentazione Qn (int)
    scale = 1 << n_bits
    a_i = np.clip((a * scale).astype(int), 0, scale - 1)
    b_i = np.clip((b * scale).astype(int), 0, scale - 1)
    
    # 2️⃣ prodotto binario → "intersezione strutturale"
    merged = (a_i * b_i) >> (n_bits // 2)
    
    # 3️⃣ normalizzazione di ritorno in [0,1)
    return merged / scale