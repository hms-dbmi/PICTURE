import numpy as np
import scanpy as sc
import anndata as ad
from multiprocessing import Pool

class Cluster:
    
    def __init__(self, neighbors=20, threshold=1, permute_num=500):
        
        self.neighbors = neighbors
        self.threshold = threshold
        self.permute_num = permute_num
        
    def preprocess_data(self, data, neighbors=20):
        
        self.data = ad.AnnData(X=data)
        sc.pp.neighbors(self.data, n_neighbors=self.neighbors, method='umap')
        self.connectivity = self.data.obsp['connectivities']
        self.inverse_degree = np.array(1/self.connectivity.sum(axis=1)).flatten()
    
    def form_clusters(self, connectivity=None, threshold=None):
        
        if threshold is not None:
            self.threshold = threshold
            
        if connectivity is not None:
            self.connectivity = connectivity
            
        total_points = self.connectivity.shape[0]
        cluster_labels = -np.ones(total_points)
        remaining_points = total_points
        cluster_index = 0
        is_initial_seed = True

        while remaining_points > 0: 
            if is_initial_seed == True: 
                seed = np.random.randint(0, remaining_points)
                is_initial_seed = False
            else:
                seed = unclustered_index[np.argmin(connectivity_value)]

            cluster_labels[seed] = cluster_index
            unclustered_index = np.argwhere(cluster_labels == -1).flatten() 
            similar_cluster_points = (cluster_labels == cluster_index)
            threshold_value = self.inverse_degree[unclustered_index] * self.threshold
            
            is_cluster_formed = True
            while is_cluster_formed:
                connectivity_value = np.array(self.connectivity[np.ix_(similar_cluster_points, unclustered_index)].mean(axis=0)).flatten()
                cluster_formed_index = connectivity_value > threshold_value
                is_cluster_formed = np.sum(cluster_formed_index)
                similar_cluster_points[unclustered_index[cluster_formed_index]] = 1
                not_clustered_index = np.logical_not(cluster_formed_index)
                unclustered_index = unclustered_index[not_clustered_index]
                threshold_value = threshold_value[not_clustered_index]

            cluster_labels[similar_cluster_points] = cluster_index
            remaining_points -= np.sum(similar_cluster_points)
            cluster_index = cluster_index + 1 
        
        self.cluster_labels = cluster_labels
        return cluster_labels

    def validate (self, num_permute=None, seed=0, init_pos=None):
        # init_pos is landmarking the proto vectors
        np.random.seed(seed)
        
        # initialization
        cluster_labels = -np.ones((self.A.shape[0], self.num_permute)) 
        Dinv = self.Dinv
        A = self.A * self.fire_temp
        
        # Starting point for the Monte Carlo trial
        if init_pos is None:
            init_pos = np.random.randint(A.shape[0])
            
        for p in range(self.num_permute): 
            if p % 100 == 0:  
                print("MC iteration", p)
            seed = init_pos if p == 0 else np.random.randint(A.shape[0])
            label_num = self.cluster_labels[seed]
            # The rest of your code here
            
        self.cluster_labels = cluster_labels


        if num_permute is not None:
            self.num_permute = num_permute
        
        if self.num_permute > self.A.shape[0]:
            self.num_permute = self.A.shape[0]
            
        
        cluster_labels = self.cluster_labels
        if len(cluster_labels) == 0:
            print("No fitting has been run yet.")
            return -1
        
            
        self.quantification()

    def calculate_quantification(self):
        self.entropy_values = np.zeros(self.clustering_labels.shape[0])
        for i in range(self.clustering_labels.shape[0]): #iterate over every data point
            current_labels = self.clustering_labels[i, :]
            labeled_data = current_labels[current_labels >= 0].astype(int)
            distribution = np.bincount(labeled_data) / np.sum(np.bincount(labeled_data))
            node_entropy = scipy.stats.entropy(distribution)
            self.entropy_values[i] = node_entropy
        self.entropy_values = np.nan_to_num(self.entropy_values)


