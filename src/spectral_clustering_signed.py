
import scipy
from scipy import sparse
import numpy as np
import torch






class Spectral_clustering_init():
    def __init__(self,num_of_eig=None,method='Normalized_sym',device=None):
        
        self.num_of_eig=num_of_eig
        self.method=method
        self.device=device

    
    def spectral_clustering(self):
        
        sparse_i=self.sparse_i_idx.cpu().numpy()
        sparse_j=self.sparse_j_idx.cpu().numpy()
        weights=self.weights_signed.cpu().numpy()
        idx_shape=sparse_i.shape[0]
        if (sparse_i<sparse_j).sum()==idx_shape:
            sparse_i_new=np.concatenate((sparse_i,sparse_j))
            sparse_j_new=np.concatenate((sparse_j,sparse_i))
            weights_new=np.concatenate((weights,weights))

            sparse_i=sparse_i_new
            sparse_j=sparse_j_new
            weights=weights_new
            
        V=np.ones(sparse_i.shape[0])
   
        Affinity_matrix_1=sparse.coo_matrix((V,(sparse_i,sparse_j)),shape=(self.input_size,self.input_size))
        Affinity_matrix_2=sparse.coo_matrix((V*weights,(sparse_i,sparse_j)),shape=(self.input_size,self.input_size))

        
        
        
       
        
        if self.method=='Adjacency':
             eig_val, eig_vect = scipy.sparse.linalg.eigsh(Affinity_matrix_2,self.num_of_eig,which='LM')
             X = eig_vect.real
             rows_norm = np.linalg.norm(X, axis=1, ord=2)
             U_norm = (X.T / rows_norm).T
            

            
        elif self.method=='Normalized_sym':
            n, m = Affinity_matrix_2.shape
            diags = Affinity_matrix_1.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix_2
            with scipy.errstate(divide="ignore"):
                diags_sqrt = 1.0 / np.sqrt(diags)
            diags_sqrt[np.isinf(diags_sqrt)] = 0
            DH = sparse.spdiags(diags_sqrt, [0], m, n, format="csr")
            tem=DH @ (L @ DH)
            eig_val, eig_vect = scipy.sparse.linalg.eigsh(tem,self.num_of_eig+1,which='SA')
            X = eig_vect.real
            self.X=X
            rows_norm = np.linalg.norm(X, axis=1,ord=2)
            U_norm =(X.T / rows_norm).T
            U_norm=U_norm[:,0:self.num_of_eig]
            
                
        elif self.method=='Normalized':
            n, m = Affinity_matrix_1.shape
            diags = Affinity_matrix_1.sum(axis=1).flatten()
            D = sparse.spdiags(diags, [0], m, n, format="csr")
            L = D - Affinity_matrix_2
            with scipy.errstate(divide="ignore"):
                diags_inv = 1.0 /diags
            diags_inv[np.isinf(diags_inv)] = 0
            DH = sparse.spdiags(diags_inv, [0], m, n, format="csr")
            tem=DH @L
            eig_val, eig_vect = scipy.sparse.linalg.eigs(tem,self.num_of_eig+1,which='SR')
    
            X = eig_vect.real
            self.X=X
            U_norm =X
            U_norm=U_norm[:,0:self.num_of_eig]

        
        else:
            print('Invalid Spectral Clustering Method')


        
        return torch.from_numpy(U_norm).float().to(self.device)
            
        



