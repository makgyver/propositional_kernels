import sys
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import classification_report, roc_auc_score
from tqdm import tqdm
from formula_generator import *

def get_label(X, tree):
    return np.array([tree(x) for x in X])


def get_dataset(n):
    return np.array([([0]*(2**i) + [1]*(2**i))*(2**(n-i-1)) for i in range(n)][::-1]).T


def svm(Ktr, ytr, Kte, yte, C=10**4):
    clf = SVC(C=C, kernel="precomputed", probability=True)
    clf.fit(Ktr, ytr)
    y_pred = clf.predict(Kte)
    y_prob = clf.predict_proba(Kte)[:,1]
    report = classification_report(yte, y_pred, output_dict=True)
    report["auc"] = roc_auc_score(yte, y_prob)
    return report


def split(y, test_size=0.3):
    f = np.where(y == 0)[0]
    t = np.where(y == 1)[0]
    np.random.shuffle(f)
    np.random.shuffle(t)
    tr_idx = np.concatenate([f[:-round(len(f)*test_size)],
                             t[:-round(len(t)*test_size)]])
    te_idx = np.concatenate([f[-round(len(f)*test_size):],
                             t[-round(len(t)*test_size):]])
    return tr_idx, te_idx


def get_subset(X, y, n):
    f = np.where(y == 0)[0]
    t = np.where(y == 1)[0]
    assert len(f) > 1 and len(t) > 1
    idf = np.random.choice(f, 2)
    idt = np.random.choice(t, 2)
    idx = np.concatenate([idf, idt, np.random.choice(list(set(range(len(y))) - set(idf) - set(idt)), n-4)])
    return X[idx], y[idx]


if __name__ == "__main__":

    if len(sys.argv) == 1:
        print("*** experiments.py usage ***\n")
        print("  python experiments.py N [S [F T]]\n")
        print("   - N : number of features")
        print("   - S (default: 2^N/10): subset size")
        print("   - F (default: 0) : start number of reps")
        print("   - T (default: 100) : end number of reps\n")
        exit(0)

    N = int(sys.argv[1]) #if len(sys.argv) > 1 else 10
    S = int(sys.argv[2]) if len(sys.argv) > 2 else 2**N // 10
    F = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    T = int(sys.argv[4]) if len(sys.argv) > 4 else 100
    if S > 2**N: S = 2**N

    np.random.seed(42)

    PROB = .4
    ETA = .7

    print("############ SETTINGS ############")
    print("  N=%d (%d)" %(N, 2**N))
    print("  S=%d" %S)
    print("  p=%.2f" %PROB)
    print("  η=%.2f" %ETA)
    print("  F-T=%d-%d" %(F, T))
    print("##################################\n")
    sys.stdout.flush()

    tauc = []
    auc = []
    for e in tqdm(range(F+1, T+1)):
        np.random.seed(98765+e*1000)
        while True:
            tree = generate_formula(N, p=PROB, eta=ETA)
            X = get_dataset(N)
            y = get_label(X, tree)
            nt, nf = np.where(y == 0)[0], np.where(y == 1)[0]
            if len(nt) > 1 and len(nf) > 1: break
        print(tree, "\n")

        tr_idx, te_idx = split(y)
        n_te = len(te_idx)
        Xtr, ytr, Xte, yte = X[tr_idx], y[tr_idx], X[te_idx], y[te_idx]
        Xtr, ytr = get_subset(Xtr, ytr, S)
        Xsub = np.vstack([Xtr, Xte])
        K, nfeats = define_kernel(tree, Xsub)
        Ktr, Kte = K[:-n_te][:, :-n_te], K[-n_te:][:, :-n_te]
        report = svm(Ktr, ytr, Kte, yte)
        tauc.append(report["auc"])

        tmp = []
        np.random.seed(123456+e*1000)
        for i in range(10):
            (K, _), fun = generate_kernel(Xsub, p=PROB, eta=ETA)
            #print(fun)
            Ktr, Kte = K[:-n_te][:, :-n_te], K[-n_te:][:, :-n_te]
            report = svm(Ktr, ytr, Kte, yte)
            tmp.append(report["auc"])
        
        auc.append(np.mean(tmp))

        if e % 10 == 0:
            print("TARGET AUC:", np.mean(tauc))
            print("OTHER AUC:", np.mean(auc))
        sys.stdout.flush()

    print("TARGET AUC:", np.mean(tauc))
    print("OTHER AUC:", np.mean(auc))
