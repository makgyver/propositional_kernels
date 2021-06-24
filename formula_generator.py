from typing import Callable
import numpy as np
from unit_test import *

UNI_KERNEL = {
    "id": KLIN,
    "not": KNOT
}

BIN_KERNEL = {
	"and": KAND, 
	"or": KOR, 
	"xor": KXOR, 
	"eq": KEQ, 
	"nand": KNAND, 
	"nor": KNOR, 
	"imp": KIMP, 
	"bimp": KBIMP, 
	"nimp": KNIMP, 
	"nbimp": KNBIMP
}

BOOL_UNI_FUN = {
    "id": r_id,
    "not": r_not
}

BOOL_BIN_FUN = {
	"and": r_and, 
	"or": r_or, 
	"xor": r_xor, 
	"eq": r_eq, 
	"nand": r_nand, 
	"nor": r_nor, 
	"imp": r_imp, 
	"bimp": r_bimp, 
	"nimp": r_nimp, 
	"nbimp": r_nbimp
}


class ParsingTreeNode:
    def __call__(self):
        raise NotImplementedError()
    
    def __str__(self):
        raise NotImplementedError()

    def __len__(self):
        raise NotImplementedError()


class TerminalNode(ParsingTreeNode):
    def __init__(self, val: int):
        self.val = val
    
    def __call__(self, x: np.ndarray) -> int:
        assert 0 <= self.val < len(x)
        return x[self.val]
    
    def __str__(self):
        return "x_%d" %self.val
    
    def __len__(self):
        return 1


class UnaryOpNode(ParsingTreeNode):
    def __init__(self,
                 fun: str,
                 operand: ParsingTreeNode):
        super(UnaryOpNode, self).__init__()
        self.fun = fun
        self.operand = operand
    
    def __call__(self, x: np.ndarray) -> int:
        return BOOL_UNI_FUN[self.fun](self.operand(x))
    
    def __str__(self) -> str:
        return self.fun + "(" + str(self.operand) + ")"
    
    def __len__(self) -> int:
        return len(self.operand)


class BinaryOpNode(ParsingTreeNode):
    def __init__(self,
                 fun: str,
                 left: ParsingTreeNode,
                 right: ParsingTreeNode):
        super(BinaryOpNode, self).__init__()
        self.fun = fun
        self.left = left
        self.right = right
    
    def __call__(self, x: np.ndarray) -> int:
        return BOOL_BIN_FUN[self.fun](self.left(x), self.right(x))

    def __str__(self) -> str:
        return  "(" + str(self.left) + " " + self.fun + " " + str(self.right) + ")"
    
    def __len__(self) -> int:
        return len(self.left) + len(self.right)


def generate_formula(n: int,
                     p: float=.2,
                     eta: float=.5,
                     t: int=0):
    #pt = np.exp(eta*t) / (1. + np.exp(eta*t))
    pt = 1. - p**(eta * t)

    if np.random.random() < pt:
        if np.random.random() < .5: 
            return TerminalNode(np.random.randint(0, n))
        else: 
            return UnaryOpNode("not", TerminalNode(np.random.randint(0, n)))
    else:
        fun = np.random.choice(list(BOOL_BIN_FUN.keys()) + ["not"])
        if fun == "not": return UnaryOpNode("not", generate_formula(n, p, eta, t+1))
        else: return BinaryOpNode(fun, generate_formula(n, p, eta, t+1), generate_formula(n, p, eta, t+1))


def define_kernel(tree, X):
    if isinstance(tree, TerminalNode):
        return KLIN(X)
    elif isinstance(tree, UnaryOpNode):
        return UNI_KERNEL[tree.fun](*define_kernel(tree.operand, X))
    elif isinstance(tree, BinaryOpNode):
        return BIN_KERNEL[tree.fun](*define_kernel(tree.left, X), *define_kernel(tree.right, X))
    else:
        raise ValueError("Unexpected Parsing Tree Node type %s!" %type(tree))


def generate_kernel(X, p=.2, eta=.7):
    fun = generate_formula(X.shape[1], p, eta)
    return define_kernel(fun, X), fun


def ttt(x, z):
    n = len(x)
    A = (n + x @ z - x @ x - z @ z) * (n**2 + (x @ z)**2)
    B = (x @ x) * (n - x @ x) * (x @ x - x @ z)
    C = (z @ z) * (n - z @ z) * (z @ z - x @ z)
    return A + B + C

if __name__ == "__main__":

    XA = np.array([
        [1,0,1,1,0,0,1,0,0,1,0,1,0,1,1],
        [1,1,0,0,1,0,1,0,1,1,1,0,0,1,0],
        [0,0,1,0,0,1,0,1,0,1,1,0,0,1,1],
        [1,0,0,1,1,0,0,0,1,0,1,1,0,0,1],
    ])

    #tree = BinaryOpNode("or", BinaryOpNode("and", TerminalNode(0), UnaryOpNode("not", TerminalNode(1))), UnaryOpNode("not", TerminalNode(2)))
    #tree = generate_formula(XA.shape[1])
    (K, nfeat), tree = generate_kernel(XA)#define_kernel(tree, XA)
    print(tree)
    #print(tree(XA[2]))
    print(K)
    print(len(tree))

    tree = BinaryOpNode("or", BinaryOpNode("and", TerminalNode(0), UnaryOpNode("not", TerminalNode(1))), UnaryOpNode("not", TerminalNode(2)))

    K, _ = define_kernel(tree, XA[2:3])
    print(K)
    print(ttt(XA[2], XA[2]))


    # L = [len(generate_formula(XA.shape[1], p=.4, eta=.7)) for i in range(1000)]
    # import matplotlib.pyplot as plt
    # plt.hist(L)
    # plt.show()
