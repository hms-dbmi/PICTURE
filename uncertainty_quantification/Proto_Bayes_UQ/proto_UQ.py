# HMS - DBMI 
# Author: Junhan Zhao. Ph.D. for PICTURE
# Done: For PICTURE
# Pending: Verify untested edge cases



import numpy as np
import scanpy as sc
import anndata as ad
import scipy.stats
from multiprocessing import Pool

class Cluster:
    def __init__(self, data, neighbors=20, threshold=1, permute_num=500, n_jobs=1):
        self.threshold = threshold
        self.permute_num = permute_num
        self.n_jobs = n_jobs

        adata = ad.AnnData(X=data)
        sc.pp.neighbors(adata, n_neighbors=neighbors, method="umap")
        self.A = adata.obsp["connectivities"]
        self.Dinv = 1 / self.A.sum(axis=1).A1

    def form_clusters(self, threshold=None):
        A = self.A * (threshold or self.threshold)
        n = A.shape[0]
        self.labels = -np.ones(n)
        label = 0
        remain, first = n, True

        while remain > 0:
            seed = np.random.randint(remain) if first else idx_unlabeled[np.argmin(heat)]
            first = False
            self.labels[seed] = label
            idx_unlabeled = np.flatnonzero(self.labels == -1)
            cluster = (self.labels == label)
            thresh = self.Dinv[idx_unlabeled]

            while True:
                heat = np.asarray(A[cluster][:, idx_unlabeled].mean(0)).ravel()
                burned = heat > thresh
                if not burned.any(): break
                cluster[idx_unlabeled[burned]] = True
                idx_unlabeled, thresh = idx_unlabeled[~burned], thresh[~burned]

            self.labels[cluster] = label
            remain -= cluster.sum()
            label += 1

        return self.labels

    def _burn_cluster(self, seeds):
        A = self.A * self.threshold
        burn = np.zeros((A.shape[0], len(seeds)), dtype=int)
        burn[seeds, np.arange(len(seeds))] = 1

        for i in range(len(seeds)):
            idx = np.flatnonzero(burn[:, i] == 0)
            thresh = self.Dinv[idx]
            while True:
                heat = np.asarray(A[burn[:, i] == 1][:, idx].mean(0)).ravel()
                burned = heat > thresh
                if not burned.any(): break
                burn[idx[burned], i] = 1
                idx, thresh = idx[~burned], thresh[~burned]

        return burn

    def run(self, parallel=False, permute_num=None, n_jobs=None, seed=0, init_pos=None):
        np.random.seed(seed)
        if permute_num: self.permute_num = permute_num
        if n_jobs: self.n_jobs = n_jobs
        if self.permute_num > self.A.shape[0]:
            self.permute_num = self.A.shape[0]

        init = init_pos or np.random.randint(self.A.shape[0])
        seeds = [init] + list(np.random.randint(self.A.shape[0], size=self.permute_num - 1))

        if parallel:
            with Pool(self.n_jobs) as p:
                parts = np.array_split(seeds, self.n_jobs)
                burns = np.concatenate(p.map(self._burn_cluster, parts), axis=1)
        else:
            burns = np.zeros((self.A.shape[0], self.permute_num), dtype=int)
            for i, s in enumerate(seeds):
                burns[:, i] = self._burn_cluster([s]).flatten()

        burns[burns == 0] = -1
        self.MC = burns
        for i, s in enumerate(seeds):
            burns[:, i][burns[:, i] >= 0] = self.labels[s]

        self._compute_uncertainty()
        self._compute_pval()

    def _compute_uncertainty(self):
        self.uncertainty = np.zeros(self.MC.shape[0])
        for i in range(self.MC.shape[0]):
            labels = self.MC[i, self.MC[i] >= 0].astype(int)
            if labels.size > 0:
                p = np.bincount(labels) / labels.size
                self.uncertainty[i] = scipy.stats.entropy(p)
        self.uncertainty = np.nan_to_num(self.uncertainty)

    def _compute_pval(self):
        self.pval = np.array([
            1 - np.mean(row[row >= 0] == self.labels[i]) if (row >= 0).sum() > 0 else 1
            for i, row in enumerate(self.MC)
        ])
        self.pval = np.nan_to_num(self.pval)

#u_train should be the embedding either HD or LD.
cl = Cluster(data=u_train, neighbors=200, threshold=50)
cl.form_clusters()  
cl.run()            

# Results
cl.labels
cl.uncertainty
cl.pval