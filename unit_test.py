import numpy as np
from numpy.testing import assert_array_equal
from propositional_kernels import *

# Logical Rules

# ⊤
def r_true():
    return 1

# ⊥
def r_false():
    return 0

# a, b
def r_id(a):
    return a

# ¬a, ¬b
def r_not(a):
	return 1 - a

# a ∧ b
def r_and(a, b):
	return int(a * b)

# a ∨ b
def r_or(a, b):
	return int(max(a, b))

# a ⊕ b
def r_xor(a, b):
	return int(a != b)

# a ↔︎ b
def r_eq(a, b):
	return int(a == b)

# a ⊼ b
def r_nand(a, b):
	return r_not(r_and(a, b))

# a ⊽ b
def r_nor(a, b):
	return r_not(r_or(a, b))

# a → b
def r_imp(a, b):
	return int(not a or b)

# b → a
def r_bimp(a, b):
	return int(not b or a)

# a ↛ b
def r_nimp(a, b):
	#return int(not b and a)
	return r_not(r_imp(a, b))

# b ↛ a
def r_nbimp(a, b):
	#return int(not a and b)
	return r_not(r_bimp(a, b))


def explicit_kernel(XA, XB, rule):
	rep = np.array([[rule(XA[j, i1], XB[j, i2]) for i1 in range(XA.shape[1])
												for i2 in range(XB.shape[1])]
												for j in range(XA.shape[0])]).T
	return rep.T @ rep


def unit_test(XA, XB):
	KA, na = KLIN(XA)
	KB, nb = KLIN(XB)
	assert_array_equal(KNOT   (KA, na)[0], (1 - XA) @ (1 - XA).T)
	assert_array_equal(KNOT   (KB, nb)[0], (1 - XB) @ (1 - XB).T)
	assert_array_equal(KAND   (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_and))
	assert_array_equal(KOR    (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_or))
	assert_array_equal(KIMP   (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_imp))
	assert_array_equal(KNIMP  (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_nimp))
	assert_array_equal(KXOR   (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_xor))
	assert_array_equal(KEQ    (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_eq))
	assert_array_equal(KBIMP  (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_bimp))
	assert_array_equal(KNOR   (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_nor))
	assert_array_equal(KNAND  (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_nand))
	assert_array_equal(KNBIMP (KA, na, KB, nb)[0], explicit_kernel(XA, XB, r_nbimp))


if __name__ == "__main__":
	
	XA = np.array([
		[1,0,1,1,0,0,1,0,0,1,0,1,0,1,1],
		[1,1,0,0,1,0,1,0,1,1,1,0,0,1,0],
		[0,0,1,0,0,1,0,1,0,1,1,0,0,1,1],
		[1,0,0,1,1,0,0,0,1,0,1,1,0,0,1],
	])

	XB = np.array([
		[1,0,1,1,0,0,1,0,1,1],
		[0,1,0,1,1,1,0,0,1,0],
		[1,0,1,0,1,1,0,0,1,1],
		[0,1,1,0,0,0,1,0,1,1],
	])

	unit_test(XA, XB)
	print("Everything is ok!")