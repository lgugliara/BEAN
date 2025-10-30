# 🫘 BEAN — Binary Mean Demos

Questa cartella contiene alcuni esempi di utilizzo e visualizzazione della *Binary Mean* (`bean`), una funzione che combina due tensori binari o reali tramite media informazionale bit-wise invece che aritmetica.

---

## ⚙️ Concetto base

<img width="889" height="221" alt="demo_arithmetic" src="https://raw.githubusercontent.com/lgugliara/BEAN/refs/heads/main/demo_arithmetic.png" />

In un sistema di rappresentazione binaria normalizzata, la media aritmetica tende a perdere informazione (collassa i bit discordanti).  
La *Binary Mean* invece conserva la struttura di ogni valore, fondendo le configurazioni binarie dei due input secondo una regola di equilibrio locale sui bit più significativi (MSB).

> In pratica: non calcoliamo una media tra numeri, ma tra **configurazioni** di bit.  
> Il risultato rappresenta la "posizione intermedia informazionale" tra due stati.

---

## 📂 Struttura

```
demos/
 ├── modules/
 ├── cube_attention.py         # Attenzione vettoriale su cubo 3D
 ├── demo_arithmetic.py        # Confronto media aritmetica vs. binaria
 ├── demo_attention.py         # Attenzione multi-head con bean(Q,K)
 ├── demo_arithmetic.png       # Output grafico della demo aritmetica
 ├── cube_attention.png        # Visualizzazione della cube attention
 └── varianza.png              # Evoluzione della varianza nel tempo
```

---

## 🧪 Contenuto delle demo

### `demo_arithmetic.py`
Mostra la differenza tra:
- **media classica**: `(A + B) / 2`
- **binary mean**: `bean(A, B)`  
che opera direttamente sui bit normalizzati dei due input.

L’output (`demo_arithmetic.png`) evidenzia pattern discreti e discontinuità logaritmiche dovute ai salti di MSB (bit più significativo).

---

### `demo_attention.py`
Esegue una simulazione di attenzione multi-head sostituendo il prodotto scalare classico con:
```python
C = bean(Q, K)
```
Il risultato è una mappa di coerenza binaria che mantiene struttura direzionale tra i token, evitando la saturazione tipica dell’attenzione continua.

---

### `modules/cube_attention.py`
Contiene una versione sperimentale di attenzione 3D (cube attention),  
in cui la *binary mean* viene usata per combinare vettori Q–K–V in uno spazio discreto tridimensionale.

---

## 📊 Output

<img width="700" height="600" alt="cube_attention" src="https://raw.githubusercontent.com/lgugliara/BEAN/refs/heads/main/cube_attention.png" />

- **`demo_arithmetic.png`** → mostra la *Binary Mean (central truncated)*: una colonna di coerenza a sinistra e una serie di scatti logaritmici verso destra, tipici della migrazione dei bit più significativi.  
- **`cube_attention.png`** → rappresentazione 3D delle teste di attenzione binaria.  
- **`varianza.png`** → andamento temporale della varianza dei token in output.

---

## 🧩 Note

- Tutto è implementato in **NumPy puro**, senza framework esterni.  
- I dati non sono pensati per training ma per **visualizzazione e analisi del comportamento informazionale**.  
- I valori risultanti sono sempre normalizzati in `[0, 1]` o `[-1, 1]` a seconda del contesto.  
- L’obiettivo è mostrare come il comportamento di *bean(Q,K)* produca pattern discreti e strutturati, interpretabili come *coerenza topologica* tra token.

---

> “Non calcoliamo somme. Calcoliamo stati possibili.”  
> — *Binary Mean Experiments, 2025*
