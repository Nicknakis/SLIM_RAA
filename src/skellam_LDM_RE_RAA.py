
# Import all the packages
import torch
import torch.nn as nn
import numpy as np
from spectral_clustering_signed import Spectral_clustering_init

# import stats
import math
from torch_sparse import spspmm

from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
from numpy.matlib import repmat





  


class SLIM_RAA_und(nn.Module,Spectral_clustering_init):
    def __init__(self,sparse_i,sparse_j,weights_signed, input_size,latent_dim,sample_size,scaling=1,device=None):
        super(SLIM_RAA_und, self).__init__()
        # initialization
        Spectral_clustering_init.__init__(self,num_of_eig=latent_dim,method='Normalized_sym',device=device)
        self.input_size=input_size
    
        self.device=device
        # self.bias1=nn.Parameter(torch.randn(1,device=device))
        # self.bias2=nn.Parameter(torch.randn(1,device=device))
        self.scaling_factor=nn.Parameter(torch.randn(1,device=device))
        self.latent_dim=latent_dim
      
        self.gamma=nn.Parameter(torch.rand(input_size,device=device))
        self.delta=nn.Parameter(torch.rand(input_size,device=device))
        # self.gamma1=nn.Parameter(torch.rand(input_size,device=device))
        # self.delta1=nn.Parameter(torch.rand(input_size,device=device))

        # self.gamma1=torch.ones(self.input_size,device=device)
        self.weights_signed=weights_signed
        #self.alpha=nn.Parameter(torch.randn(self.input_size,device=device))
        
        self.scaling=scaling
        #create indices to index properly the receiver and senders variable
        self.sparse_i_idx=sparse_i
        self.sparse_j_idx=sparse_j
        
        self.sampling_weights=torch.ones(self.input_size,device=device)
        self.sample_size=sample_size
        self.Softmax=nn.Softmax(1)

        
        if sample_size==input_size:
            self.up_i,self.up_j=torch.triu_indices(input_size,input_size,1)
        else:
            self.up_i,self.up_j=torch.triu_indices(sample_size,sample_size,1)

            
        self.softplus=nn.Softplus()
       
        
        self.spectral_data=self.spectral_clustering()
        self.elements=0.5*(input_size*(input_size-1))
        self.torch_pi=torch.tensor(math.pi)

      

        spectral_centroids_to_z=self.spectral_data
       
        self.latent_z_=spectral_centroids_to_z
        print('\nFinished spectral decomposition of the singed Laplacian...\n')
        
            

    def sample_network(self):
        # USE torch_sparse lib i.e. : from torch_sparse import spspmm
    
        # sample for undirected network
        sample_idx=torch.multinomial(self.sampling_weights, self.sample_size,replacement=False)
        # translate sampled indices w.r.t. to the full matrix, it is just a diagonal matrix
        indices_translator=torch.cat([sample_idx.unsqueeze(0),sample_idx.unsqueeze(0)],0)
        # adjacency matrix in edges format
        edges=torch.cat([self.sparse_i_idx.unsqueeze(0),self.sparse_j_idx.unsqueeze(0)],0)
        # matrix multiplication B = Adjacency x Indices translator
        # see spspmm function, it give a multiplication between two matrices
        # indexC is the indices where we have non-zero values and valueC the actual values (in this case ones)
        indexC, valueC = spspmm(edges,self.weights_signed.float(), indices_translator,torch.ones(indices_translator.shape[1]),self.input_size,self.input_size,self.input_size,coalesced=True)
        # second matrix multiplication C = Indices translator x B, indexC returns where we have edges inside the sample
        indexC, valueC=spspmm(indices_translator,torch.ones(indices_translator.shape[1]),indexC,valueC,self.input_size,self.input_size,self.input_size,coalesced=True)
        
        # edge row position
        sparse_i_sample=indexC[0,:]
        # edge column position
        sparse_j_sample=indexC[1,:]
     
        
        return sample_idx,sparse_i_sample,sparse_j_sample,valueC
    
    
    
    def furthest_sum(self, exclude=[]):
        """
        CODE taken from: https://github.com/ulfaslak/py_pcha/blob/master/py_pcha/furthest_sum.py

        
        Furthest sum algorithm, to efficiently generat initial seed/archetypes.
        Note: Commonly data is formatted to have shape (examples, dimensions).
        This function takes input and returns output of the transposed shape,
        (dimensions, examples).
        
        Parameters
        ----------
        K : numpy 2d-array
            Either a data matrix or a kernel matrix.
        noc : int
            Number of candidate archetypes to extract.
        i : int
            inital observation used for to generate the FurthestSum.
        exclude : numpy.1darray
            Entries in K that can not be used as candidates.
        Output
        ------
        i : int
            The extracted candidate archetypes
        """
        def max_ind_val(l):
            return max(zip(range(len(l)), l), key=lambda x: x[1])
        
        print('Initializing convex hull based on furthest sum... \n')
        K=self.latent_z_.transpose(0,1).cpu().numpy()
        noc=self.latent_dim
        i=[6]
        
        I, J = K.shape
        index = np.array(range(J))
        index[exclude] = 0
        index[i] = -1
        ind_t = i
        sum_dist = np.zeros((1, J), np.complex128)

        if J > noc * I:
            Kt = K
            Kt2 = np.sum(Kt**2, axis=0)
            for k in range(1, noc + 11):
                if k > noc - 1:
                    Kq = np.dot(Kt[:, i[0]], Kt)
                    sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[i[0]])
                    index[i[0]] = i[0]
                    i = i[1:]
                t = np.where(index != -1)[0]
                Kq = np.dot(Kt[:, ind_t].T, Kt)
                sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * Kq + Kt2[ind_t])
                ind, val = max_ind_val(sum_dist[:, t][0].real)
                ind_t = t[ind]
                i.append(ind_t)
                index[ind_t] = -1
        else:
            if I != J or np.sum(K - K.T) != 0:  # Generate kernel if K not one
                Kt = K
                K = np.dot(Kt.T, Kt)
                K = np.lib.scimath.sqrt(
                    repmat(np.diag(K), J, 1) - 2 * K + \
                    repmat(np.mat(np.diag(K)).T, 1, J)
                )

            Kt2 = np.diag(K)  # Horizontal
            for k in range(1, noc + 11):
                if k > noc - 1:
                    sum_dist -= np.lib.scimath.sqrt(Kt2 - 2 * K[i[0], :] + Kt2[i[0]])
                    index[i[0]] = i[0]
                    i = i[1:]
                t = np.where(index != -1)[0]
                sum_dist += np.lib.scimath.sqrt(Kt2 - 2 * K[ind_t, :] + Kt2[ind_t])
                ind, val = max_ind_val(sum_dist[:, t][0].real)
                ind_t = t[ind]
                i.append(ind_t)
                index[ind_t] = -1
        convex_set=K[:,i].transpose()
        return i,convex_set
    
    def find_convex_hull(self):
        idx,self.convex_set=self.furthest_sum()
        self.hull_idx=np.array(idx)
    
    
    def LDM_to_RAA(self):
        
        z_ldm=self.latent_z_.cpu().numpy()
        self.K_arch=self.convex_set.shape[0]
        # k x D
        self.R=nn.Parameter(torch.tensor(self.convex_set,device=self.device))
        
        
       
        # N x K
        self.G=nn.Parameter(torch.randn(self.input_size,self.K_arch,device=self.device))
        # N x K
        # initial G
        in_G=-100*torch.ones(self.input_size,self.K_arch,device=self.device)
        in_G[self.hull_idx]=-1*in_G[self.hull_idx]
        self.G=nn.Parameter(in_G)
        
        
        zetas_RAA=[]
        
        A_new=self.convex_set

        for z in z_ldm:
            m1= -(z.reshape(-1,self.latent_dim)@A_new.T).reshape(-1,1)
            
            p=matrix(m1.astype(np.double))
            m2=A_new@A_new.T
            Q=matrix(m2.astype(np.double))
            G=matrix(-1*np.eye(A_new.shape[0],A_new.shape[0]))
            h=matrix(np.zeros(A_new.shape[0]))
            A=matrix(np.ones(A_new.shape[0]).reshape(1,-1))
            b=matrix(1.0)
            sol=solvers.qp(Q, p, G, h, A, b)
            dz=torch.tensor(np.array(sol['x']))
            zetas_RAA.append(dz)
            
            
        temp_zetas=torch.cat(zetas_RAA,1).float().transpose(0,1)
        self.temp_zetas=temp_zetas.abs()
        self.latent_z1=nn.Parameter(torch.log(temp_zetas.abs()))
        #NxK
        self.latent_raa_z=self.Softmax(self.latent_z1)
  
        self.Gate=torch.sigmoid(self.G)
        self.C = (self.latent_raa_z * self.Gate) / (self.latent_raa_z * self.Gate).sum(0)
        self.A=(self.R.transpose(0,1)@(self.latent_raa_z.transpose(0,1)@self.C)).transpose(0,1)
        self.latent_z=self.latent_raa_z@self.A
    
        print('Initialization Done...\n')

    
    
    def LSM_likelihood_bias_sample(self,epoch):
        '''
        Skellam MAP ignoring constant terms
        
        '''
        self.epoch=epoch
        
        
        
        self.latent_raa_z=self.Softmax(self.latent_z1)
  
        self.Gate=torch.sigmoid(self.G)
        self.C = (self.latent_raa_z * self.Gate) / (self.latent_raa_z * self.Gate).sum(0)
        self.A=(self.R.transpose(0,1)@(self.latent_raa_z.transpose(0,1)@self.C)).transpose(0,1)
        self.latent_z=self.latent_raa_z@self.A
        
        
        sample_idx,sparse_i_sample,sparse_j_sample,self.weights_sample=self.sample_network()

        if self.scaling:
            
            
            mat=torch.exp(-((torch.cdist(self.latent_z[sample_idx],self.latent_z[sample_idx],p=2))+1e-06)).detach()
            z_pdist1_1=0.5*torch.mm(torch.exp(self.gamma[sample_idx].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma[sample_idx]).unsqueeze(-1))))
            z_pdist1_2=0.5*torch.mm(torch.exp(-self.delta[sample_idx].unsqueeze(0)),(torch.mm(((1/mat)-torch.diag(torch.diagonal((1/mat)))),torch.exp(-self.delta[sample_idx]).unsqueeze(-1))))

            z_pdist1=z_pdist1_1+z_pdist1_2

            #take the sampled matrix indices in order to index gamma_i and alpha_j correctly and in agreement with the previous
            temp_dist=(((((self.latent_z[sparse_i_sample]-self.latent_z[sparse_j_sample]+1e-06)**2).sum(-1)))**0.5).detach()
            z_pdist2=((self.weights_sample/2)*((-2*temp_dist)+(self.gamma[sparse_i_sample]+self.gamma[sparse_j_sample]+self.delta[sparse_i_sample]+self.delta[sparse_j_sample]))).sum()
            # z_pdist2_neg=(((((self.latent_z[sparse_i_neg]-self.latent_z[sparse_j_neg]+1e-06)**2).sum(-1)))**0.5-self.bias2).sum()
           
            temp_bias_0=0.5*(((self.gamma[sample_idx].unsqueeze(1)+self.gamma[sample_idx])-(self.delta[sample_idx].unsqueeze(1)+self.delta[sample_idx]))[self.up_i,self.up_j])
            temp_bias_1=0.5*(self.gamma[sparse_i_sample]+self.gamma[sparse_j_sample]-self.delta[sparse_i_sample]-self.delta[sparse_j_sample])
            

            log_bessel_0,log_bessel_1=self.bessel_calc_sample(rates_non_link=temp_bias_0,rates_link=temp_bias_1)
            
            log_likelihood_sparse=z_pdist2-z_pdist1+log_bessel_0+log_bessel_1-0.5*((self.gamma[sample_idx]**2).sum()+(self.delta[sample_idx]**2).sum())

    
            if self.epoch==1000:
                self.scaling=0
        else:
            
            
            mat=torch.exp(-((torch.cdist(self.latent_z[sample_idx],self.latent_z[sample_idx],p=2))+1e-06))+1e-06
            z_pdist1_1=0.5*torch.mm(torch.exp(self.gamma[sample_idx].unsqueeze(0)),(torch.mm((mat-torch.diag(torch.diagonal(mat))),torch.exp(self.gamma[sample_idx]).unsqueeze(-1))))
            z_pdist1_2=0.5*torch.mm(torch.exp(-self.delta[sample_idx].unsqueeze(0)),(torch.mm(((1/mat)-torch.diag(torch.diagonal((1/mat)))),torch.exp(-self.delta[sample_idx]).unsqueeze(-1))))

            z_pdist1=z_pdist1_1+z_pdist1_2

            temp_dist=(((((self.latent_z[sparse_i_sample]-self.latent_z[sparse_j_sample]+1e-06)**2).sum(-1)))**0.5)
            z_pdist2=((self.weights_sample/2)*((-2*temp_dist)+(self.gamma[sparse_i_sample]+self.gamma[sparse_j_sample]+self.delta[sparse_i_sample]+self.delta[sparse_j_sample]))).sum()
           
            temp_bias_0=0.5*(((self.gamma[sample_idx].unsqueeze(1)+self.gamma[sample_idx])-(self.delta[sample_idx].unsqueeze(1)+self.delta[sample_idx]))[self.up_i,self.up_j])
            temp_bias_1=0.5*(self.gamma[sparse_i_sample]+self.gamma[sparse_j_sample]-self.delta[sparse_i_sample]-self.delta[sparse_j_sample])
            
         

            log_bessel_0,log_bessel_1=self.bessel_calc_sample(rates_non_link=temp_bias_0,rates_link=temp_bias_1)
            
            log_likelihood_sparse=z_pdist2-z_pdist1+log_bessel_0+log_bessel_1-0.5*((self.A**2).sum()+(self.gamma[sample_idx]**2).sum()+(self.delta[sample_idx]**2).sum())
        
        
        return log_likelihood_sparse
    
    def bessel_calc_sample(self,rates_non_link,rates_link):
        
       
        nu_link=self.weights_sample.abs().float()
    
    
    
        sum_el=50
       
        order=torch.arange(sum_el)


        q=-torch.special.gammaln(order+1)-torch.special.gammaln(nu_link.unsqueeze(1)+order+1)+  rates_link.unsqueeze(-1)*(nu_link.unsqueeze(1)+2*order)
        logI=torch.logsumexp(q,1)
        
        
        nu_link_z=torch.zeros(self.up_i.shape[0]).float()
    
    
    
      

        q_z=-torch.special.gammaln(order+1)-torch.special.gammaln(nu_link_z.unsqueeze(1)+order+1)+  rates_non_link.unsqueeze(-1)*(nu_link_z.unsqueeze(1)+2*order)
        logI_z=torch.logsumexp(q_z,1)
        
        nu_link_e=torch.zeros(self.weights_sample.shape[0]).float()
    
    
    
       


        q_e=-torch.special.gammaln(order+1)-torch.special.gammaln(nu_link_e.unsqueeze(1)+order+1)+  rates_link.unsqueeze(-1)*(nu_link_e.unsqueeze(1)+2*order)
        logI_e=torch.logsumexp(q_e,1)

      
        
        log_bessel_0=logI_z.sum()-logI_e.sum()

        log_bessel_1=logI.sum()
        return log_bessel_0,log_bessel_1
    
    
    
    

    
    
    
 
           
