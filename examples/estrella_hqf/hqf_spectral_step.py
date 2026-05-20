import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA


class HQFSpectralEmbeddingStep:
    """
    Minimal spectral operator step for MMCTAgent.
    """

    def __init__(self, k=12, sigma=0.1, n_eigs=20, seed=0):
        self.k = k
        self.sigma = sigma
        self.n_eigs = n_eigs
        self.seed = seed

    def sierpinski(self, level=5):
        pts = np.array([
            [0.0, 0.0],
            [1.0, 0.0],
            [0.5, np.sqrt(3) / 2]
        ], dtype=float)

        for _ in range(level):
            pts = np.vstack([
                pts / 2.0,
                (pts + np.array([1.0, 0.0])) / 2.0,
                (pts + np.array([0.5, np.sqrt(3) / 2])) / 2.0
            ])

        return np.unique(pts, axis=0)

    def build_laplacian(self, X):
        n = X.shape[0]
        if n < 3:
            raise ValueError("Need at least 3 points to build a meaningful graph.")

        k = min(self.k + 1, n)
        nbrs = NearestNeighbors(n_neighbors=k).fit(X)
        dist, idx = nbrs.kneighbors(X)

        rows, cols, vals = [], [], []
        for i in range(n):
            for j, d in zip(idx[i][1:], dist[i][1:]):
                w = np.exp(-(d * d) / (2.0 * self.sigma * self.sigma))
                rows.append(i)
                cols.append(j)
                vals.append(w)

        W = sp.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
        W = 0.5 * (W + W.T)  # true symmetrization

        d = np.asarray(W.sum(axis=1)).ravel()
        inv = 1.0 / np.sqrt(d + 1e-12)
        Dinv = sp.diags(inv)
        L = sp.eye(n, format="csr") - Dinv @ W @ Dinv
        return L

    def labels(self, X):
        x, y = X[:, 0], X[:, 1]
        r = np.sqrt(x * x + y * y)
        t = np.arctan2(y, x + 1e-8)
        return (
            ((t > 0) & (r > 0.4)).astype(int)
            + 2 * ((t < 0) & (r > 0.4)).astype(int)
        )

    def run(self, level=5):
        rng = np.random.default_rng(self.seed)

        X = self.sierpinski(level)
        y = self.labels(X)

        if len(np.unique(y)) < 2:
            raise ValueError("Not enough label diversity for classification.")

        L = self.build_laplacian(X)

        n = X.shape[0]
        k = min(self.n_eigs, n - 2)
        if k < 2:
            raise ValueError("n_eigs is too small relative to the number of points.")

        vals, vecs = spla.eigsh(L, k=k, which="SA")
        E = vecs

        idx = np.arange(n)
        tr, te = train_test_split(
            idx,
            test_size=0.3,
            random_state=self.seed,
            stratify=y
        )

        clf = LogisticRegression(max_iter=2000, random_state=self.seed)
        clf.fit(E[tr], y[tr])
        pred = clf.predict(E[te])

        clf2 = LogisticRegression(max_iter=2000, random_state=self.seed)
        Xp = PCA(n_components=min(2, X.shape[1])).fit_transform(X)
        clf2.fit(Xp[tr], y[tr])
        pred_pca = clf2.predict(Xp[te])

        gaps = np.diff(np.sort(vals))
        gpos = gaps[gaps > 0]
        entropy = float(
            -np.sum((gpos / (np.sum(gpos) + 1e-15)) * np.log(gpos / (np.sum(gpos) + 1e-15) + 1e-15))
        ) if len(gpos) else 0.0

        return {
            "accuracy": float(accuracy_score(y[te], pred)),
            "f1_macro": float(f1_score(y[te], pred, average="macro")),
            "balanced_accuracy": float(balanced_accuracy_score(y[te], pred)),
            "accuracy_pca": float(accuracy_score(y[te], pred_pca)),
            "f1_macro_pca": float(f1_score(y[te], pred_pca, average="macro")),
            "mean_gap": float(gaps.mean()) if len(gaps) else 0.0,
            "entropy_gap": entropy,
            "n_points": int(len(X)),
            "n_eigs": int(k),
            "n_classes": int(len(np.unique(y)))
        }
