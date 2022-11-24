

# Import all the packages
import torch

import numpy as np

from sklearn import metrics

from sklearn.linear_model import LogisticRegression




class LP_():
    def __init__(self,latent_z,gamma,delta,dataset,sparse_i,sparse_j,weights_signed,device):
        super(LP_, self).__init__()

        self.latent_z=latent_z
        self.gamma=gamma
        self.delta=delta
        removed_edges=np.loadtxt("./datasets/undirected/"+dataset+"/removed_edges.txt")

 
        removed_edges[:,0:2].sort(1)
        mask=removed_edges[:,0]<removed_edges[:,1]
        removed_edges=removed_edges[mask]

        sparse_i_rem=torch.from_numpy(removed_edges[:,0]).long().to(device)
        # input data, link column positions with i<j
        sparse_j_rem=torch.from_numpy(removed_edges[:,1]).long().to(device)


        self.weights_signed_rem=torch.from_numpy(removed_edges[:,2]).long().to(device)
        self.weights_signed=weights_signed

        idx_plus=torch.where(self.weights_signed_rem>=1)
        
        self.sparse_i_pos=sparse_i_rem[idx_plus]
        self.sparse_j_pos=sparse_j_rem[idx_plus]
        idx_neg=torch.where(self.weights_signed_rem<=-1)
        
        self.sparse_i_neg=sparse_i_rem[idx_neg]
        self.sparse_j_neg=sparse_j_rem[idx_neg]
        self.sparse_i_zer=np.loadtxt("./datasets/undirected/"+dataset+'/sparse_i_zer.txt')
        self.sparse_j_zer=np.loadtxt("./datasets/undirected/"+dataset+'/sparse_j_zer.txt')

        self.binary_operator='con'
        ########################################################################################
        pos_idx=torch.where(self.weights_signed>0)
        neg_idx=torch.where(self.weights_signed<0)
        
        self.train_pos_i=sparse_i[pos_idx].cpu().numpy()
        self.train_pos_j=sparse_j[pos_idx].cpu().numpy()
        self.train_neg_i=sparse_i[neg_idx].cpu().numpy()
        self.train_neg_j=sparse_j[neg_idx].cpu().numpy()
        self.train_zer_i=np.loadtxt("./datasets/undirected/"+dataset+'/sparse_i_zer_train.txt').astype(int)
        self.train_zer_j=np.loadtxt("./datasets/undirected/"+dataset+'/sparse_j_zer_train.txt').astype(int)
        
        self.test_pos_i=self.sparse_i_pos.cpu().numpy()
        self.test_pos_j=self.sparse_j_pos.cpu().numpy()
        self.test_neg_i=self.sparse_i_neg.cpu().numpy()
        self.test_neg_j=self.sparse_j_neg.cpu().numpy()
        self.test_zer_i=self.sparse_i_zer.astype(int)
        self.test_zer_j=self.sparse_j_zer.astype(int)
        
    def vec1(self,i,j):
        
        d=-((((self.latent_z[i]-self.latent_z[j])**2).sum(-1))**0.5)+self.gamma[i]+self.gamma[j]
        return np.exp(d.detach().cpu().numpy())    
    
    def vec2(self,i,j):
        d=((((self.latent_z[i]-self.latent_z[j])**2).sum(-1))**0.5)-self.delta[i]-self.delta[j]
        return np.exp(d.detach().cpu().numpy())    
    
    

    def pos_neg(self):

        train_target1=np.ones(self.train_pos_i.shape[0])
        train_target2=-np.ones(self.train_neg_i.shape[0])
        train_target=np.concatenate((train_target1,train_target2))
        
        
        test_target1=np.ones(self.test_pos_i.shape[0])
        test_target2=-np.ones(self.test_neg_i.shape[0])
        test_target=np.concatenate((test_target1,test_target2))
        
        
        if self.binary_operator == "con":
            value_train1 = np.concatenate((self.vec1(self.train_pos_i,self.train_pos_j).reshape(-1,1),self.vec2(self.train_pos_i,self.train_pos_j).reshape(-1,1)),1)
            value_train2 =np.concatenate((self.vec1(self.train_neg_i,self.train_neg_j).reshape(-1,1),self.vec2(self.train_neg_i,self.train_neg_j).reshape(-1,1)),1)
            value_train1=np.concatenate((value_train1,np.log(value_train1)),1)
            value_train2=np.concatenate((value_train2,np.log(value_train2)),1)
            value_train=np.concatenate((value_train1,value_train2))
            
            value_test1 = np.concatenate((self.vec1(self.test_pos_i,self.test_pos_j).reshape(-1,1),self.vec2(self.test_pos_i,self.test_pos_j).reshape(-1,1)),1)
            value_test2 =np.concatenate((self.vec1(self.test_neg_i,self.test_neg_j).reshape(-1,1),self.vec2(self.test_neg_i,self.test_neg_j).reshape(-1,1)),1)
            value_test1=np.concatenate((value_test1,np.log(value_test1)),1)
            value_test2=np.concatenate((value_test2,np.log(value_test2)),1)
            value_test=np.concatenate((value_test1,value_test2))
            




   
        train_features =value_train #ss.fit_transform(value_train)
        test_features =value_test #ss.fit_transform(value_test)
        
        
        
        
        clf = LogisticRegression()
        clf.fit(value_train, train_target)
        
        train_preds = clf.predict_proba(train_features)[:, 1]
        test_preds = clf.predict_proba(test_features)[:, 1]
        
        
        precision, tpr, thresholds = metrics.precision_recall_curve(test_target,test_preds)
        
           
        roc=metrics.roc_auc_score(test_target,test_preds)
        pr=metrics.auc(tpr,precision)
        print(f'p@n ROC score: {roc}\n')
        print(f'p@n PR score: {pr}\n')
        return roc,pr




    def pos_zer(self):

        ####################################################################################################################
        
        
        
        train_target1=np.ones(self.train_pos_i.shape[0])
        train_target2=np.zeros(self.train_zer_i.shape[0])
        train_target=np.concatenate((train_target1,train_target2))
        
        
        test_target1=np.ones(self.test_pos_i.shape[0])
        test_target2=np.zeros(self.test_zer_i.shape[0])
        test_target=np.concatenate((test_target1,test_target2))
        
        
        if self.binary_operator == "con":
            value_train1 = np.concatenate((self.vec1(self.train_pos_i,self.train_pos_j).reshape(-1,1),self.vec2(self.train_pos_i,self.train_pos_j).reshape(-1,1)),1)
            value_train2 =np.concatenate((self.vec1(self.train_zer_i,self.train_zer_j).reshape(-1,1),self.vec2(self.train_zer_i,self.train_zer_j).reshape(-1,1)),1)
            value_train1=np.concatenate((value_train1,np.log(value_train1)),1)
            value_train2=np.concatenate((value_train2,np.log(value_train2)),1)
            value_train=np.concatenate((value_train1,value_train2))
            
            value_test1 = np.concatenate((self.vec1(self.test_pos_i,self.test_pos_j).reshape(-1,1),self.vec2(self.test_pos_i,self.test_pos_j).reshape(-1,1)),1)
            value_test2 =np.concatenate((self.vec1(self.test_zer_i,self.test_zer_j).reshape(-1,1),self.vec2(self.test_zer_i,self.test_zer_j).reshape(-1,1)),1)
            value_test1=np.concatenate((value_test1,np.log(value_test1)),1)
            value_test2=np.concatenate((value_test2,np.log(value_test2)),1)
            value_test=np.concatenate((value_test1,value_test2))
        
        
        
        
        train_features =value_train #ss.fit_transform(value_train)
        test_features =value_test #ss.fit_transform(value_test)
        
        
        clf = LogisticRegression()
        clf.fit(value_train, train_target)
        
        train_preds = clf.predict_proba(train_features)[:, 1]
        test_preds = clf.predict_proba(test_features)[:, 1]
        
        
        precision, tpr, thresholds = metrics.precision_recall_curve(test_target,test_preds)
        
           
        roc=metrics.roc_auc_score(test_target,test_preds)
        pr=metrics.auc(tpr,precision)
        print(f'p@z ROC score: {roc}\n')
        print(f'p@z PR score: {pr}\n')
        return roc,pr







####################################################################################################################

    def neg_zer(self):


        train_target1=np.ones(self.train_neg_i.shape[0])
        train_target2=np.zeros(self.train_zer_i.shape[0])
        train_target=np.concatenate((train_target1,train_target2))
        
        
        test_target1=np.ones(self.test_neg_i.shape[0])
        test_target2=np.zeros(self.test_zer_i.shape[0])
        test_target=np.concatenate((test_target1,test_target2))
        
        
        if self.binary_operator == "con":
            value_train1 = np.concatenate((self.vec1(self.train_neg_i,self.train_neg_j).reshape(-1,1),self.vec2(self.train_neg_i,self.train_neg_j).reshape(-1,1)),1)
            value_train2 =np.concatenate((self.vec1(self.train_zer_i,self.train_zer_j).reshape(-1,1),self.vec2(self.train_zer_i,self.train_zer_j).reshape(-1,1)),1)
            value_train1=np.concatenate((value_train1,np.log(value_train1)),1)
            value_train2=np.concatenate((value_train2,np.log(value_train2)),1)
            value_train=np.concatenate((value_train1,value_train2))
        
           
            value_test1 = np.concatenate((self.vec1(self.test_neg_i,self.test_neg_j).reshape(-1,1),self.vec2(self.test_neg_i,self.test_neg_j).reshape(-1,1)),1)
            value_test2 =np.concatenate((self.vec1(self.test_zer_i,self.test_zer_j).reshape(-1,1),self.vec2(self.test_zer_i,self.test_zer_j).reshape(-1,1)),1)
            value_test1=np.concatenate((value_test1,np.log(value_test1)),1)
            value_test2=np.concatenate((value_test2,np.log(value_test2)),1)
        
            value_test=np.concatenate((value_test1,value_test2))
        
        
        
           
        train_features =value_train 
        test_features =value_test 
        
        
        clf = LogisticRegression()
        clf.fit(value_train, train_target)
        
        train_preds = clf.predict_proba(train_features)[:, 1]
        test_preds = clf.predict_proba(test_features)[:, 1]
        
        
        precision, tpr, thresholds = metrics.precision_recall_curve(test_target,test_preds)
        
           
        roc=metrics.roc_auc_score(test_target,test_preds)
        pr=metrics.auc(tpr,precision)
        print(f'n@z ROC score: {roc}\n')
        print(f'n@z PR score: {pr}\n')

        return roc,pr









