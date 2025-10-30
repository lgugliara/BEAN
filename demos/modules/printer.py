import struct
import numpy as np

def bits(f):
    # 1. Converte il float a 32-bit (np.float32) e poi ottiene 
    #    la sua rappresentazione intera non firmata a 32 bit (>I).
    #    Questo è il valore binario grezzo del float.
    [d] = struct.unpack(">I", struct.pack(">f", np.float32(f)))
    
    # 2. Converte l'intero 'd' in una stringa binaria di 32 cifre (032b).
    binary_string = f"{d:032b}"
    
    # 3. Inserisce gli spazi nelle posizioni corrette (1, 8, 23).
    #    - bit 0: Segno
    #    - bit 1 a 8: Esponente
    #    - bit 9 a 31: Mantissa
    sign = binary_string[0]
    exponent = binary_string[1:9]
    mantissa = binary_string[9:]
    
    return f"{sign} {exponent} {mantissa}"

# si chiama prit perché è print+bit :)
def prit(desc, bit):
    print(desc, bits(bit))