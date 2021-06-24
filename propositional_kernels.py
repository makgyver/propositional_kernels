import numpy as np
from typing import Tuple


def KLIN(X: np.ndarray) -> Tuple[np.ndarray, int]:
	return [X @ X.T, X.shape[1]]

def KXX(KA: np.ndarray) -> np.ndarray:
	n = KA.shape[0]
	d = np.array([[KA[i, i] for i in range(n)]])
	U = np.ones((1, n))
	return (d.T @ U)

def KNOT(K: np.ndarray, n: int) -> Tuple[np.ndarray, int]:
	dim = K.shape[0]
	Q = np.array([[K[i, i] + K[j, j] for i in range(dim)] for j in range(dim)])
	return [n + K - Q, n]
	
def KAND(KA: np.ndarray, na: int, 
         KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return [KA * KB, na * nb]

def KOR(KA: np.ndarray, na: int, 
        KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return [na*nb \
			- KAND(*KNOT(KXX(KA),na), *KNOT(KXX(KB),nb))[0] \
			- KAND(*KNOT(KXX(KA),na), *KNOT(KXX(KB),nb))[0].T \
			+ KAND(*KNOT(KA,na), *KNOT(KB,nb))[0] \
			, na * nb]

def KXOR(KA: np.ndarray, na: int, 
         KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return [2*KAND(KA,na,KB,nb)[0] \
			+ KAND(KA,na,*KNOT(KB,nb))[0] \
			+ KAND(*KNOT(KA,na),KB,nb)[0] \
			- KAND(KA,na,KXX(KB).T,nb)[0] \
			+ KAND(KXX(KA),na,KXX(KB).T,nb)[0] \
			- KAND(KXX(KA),na,KB,nb)[0] \
			- KAND(KA,na,KXX(KB),nb)[0] \
			+ KAND(KXX(KA).T,na,KXX(KB),nb)[0] \
			- KAND(KXX(KA).T,na,KB,nb)[0] \
			, na * nb]

def KIMP(KA: np.ndarray, na: int, 
         KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KOR(KNOT(KA, na)[0], na, KB, nb)
	
def KBIMP(KA: np.ndarray, na: int, 
          KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KOR(KA, na, KNOT(KB, nb)[0], nb)

def KNIMP(KA: np.ndarray, na: int, 
          KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KAND(KA, na, *KNOT(KB, nb))

def KEQ(KA: np.ndarray, na: int, 
        KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KNOT(*(KXOR(KA, na, KB, nb)))

def KNOR(KA: np.ndarray, na: int, 
         KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KNOT(*(KOR(KA, na, KB, nb)))

def KNAND(KA: np.ndarray, na: int, 
         KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KNOT(*(KAND(KA, na, KB, nb)))

def KNBIMP(KA: np.ndarray, na: int, 
           KB: np.ndarray, nb: int) -> Tuple[np.ndarray, int]:
	return KNOT(*(KOR(KA, na, KNOT(KB, nb)[0], nb)))
