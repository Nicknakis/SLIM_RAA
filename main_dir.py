# Import all the packages
import argparse
import sys
from tqdm import tqdm
import torch

import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt

#from blobs import *
from sklearn.decomposition import PCA
# import sparse 
# import stats
import math



sys.path.append('./src/')

from skellam_LDM_RE_RAA_dir import SLIM_RAA_dir
from link_prediction_dir import LP_

parser = argparse.ArgumentParser(description='Skellam Latent Distance Models')

parser.add_argument('--epochs', type=int, default=5000, metavar='N',
                    help='number of epochs for training (default: 5K)')


parser.add_argument('--cuda', type=eval, 
                      choices=[True, False],  default=True,
                    help='CUDA training')



parser.add_argument('--LP',type=eval, 
                      choices=[True, False], default=True,
                    help='performs link prediction')

parser.add_argument('--pretrained',type=eval, 
                      choices=[True, False], default=True,
                    help='Uses pretrained embeddings for link prediction (default: True)')

parser.add_argument('--D', type=int, default=8, metavar='N',
                    help='dimensionality of the embeddings (default: 8)')

parser.add_argument('--lr', type=float, default=0.05, metavar='N',
                    help='learning rate for the ADAM optimizer, for large values of delta 0.01 is more stable (default: 0.05)')

parser.add_argument('--sample_percentage', type=float, default=0.2, metavar='N',
                    help='Sample size network percentage, it should be equal or less than 1 (default: 0.3)')


parser.add_argument('--reg_strength', type=float, default=0.5, metavar='N',
                    help='Regularization strength over the model parameters (default: 0.5 equivalent to normal priors)')

parser.add_argument('--dataset', type=str, default='wiki_elec',
                    help='dataset to apply Skellam Latent Distance Modeling on')



args = parser.parse_args()

if  args.cuda:
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    device = torch.device("cpu")
    torch.set_default_tensor_type('torch.FloatTensor')





plt.style.use('ggplot')


torch.autograd.set_detect_anomaly(True)


if __name__ == "__main__":
    latent_dim=args.D
    dataset=args.dataset
   
    
   
    losses=[]
    data=np.loadtxt("./datasets/directed/"+dataset+'/edges.txt')
    mask=data[:,0]!=data[:,1]
    data=data[mask]
    
    sparse_i=torch.from_numpy(data[:,0]).long().to(device)
    # input data, link column positions with i<j
    sparse_j=torch.from_numpy(data[:,1]).long().to(device)
 

    weights_signed=torch.from_numpy(data[:,2]).long().to(device)
    
   
    
    # network size
    N=int(max(sparse_j.max(),sparse_i.max())+1)
    # sample size of blocks-> sample_size*(sample_size-1)/2 pairs
    
    sample_size=int(args.sample_percentage*N)
    model = SLIM_RAA_dir(args.reg_strength,sparse_i,sparse_j,weights_signed,N,latent_dim=latent_dim,sample_size=sample_size,device=device).to(device)         
    
    # create initial convex hull
    model.find_convex_hull()
    # initialize SLIM-RAA model
    model.LDM_to_RAA()
    
    # set-up optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)  

    if args.pretrained:
       
        
        model.gamma_1=torch.load('./pretrained_emb/directed/'+dataset+f'/gamma_1_{latent_dim}_RAA_reg_{args.reg_strength}',map_location=device)
        model.gamma_2=torch.load('./pretrained_emb/directed/'+dataset+f'/gamma_2_{latent_dim}_RAA_reg_{args.reg_strength}',map_location=device)

        model.delta_1=torch.load('./pretrained_emb/directed/'+dataset+f'/delta_1_{latent_dim}_RAA_reg_{args.reg_strength}',map_location=device)
        model.delta_2=torch.load('./pretrained_emb/directed/'+dataset+f'/delta_2_{latent_dim}_RAA_reg_{args.reg_strength}',map_location=device)

        model.latent_z1=torch.load(f"./pretrained_emb/directed/{dataset}/Z1_{latent_dim}_RAA_reg_{args.reg_strength}",map_location=device)
        model.G=torch.load(f"./pretrained_emb/directed/{dataset}/G_{latent_dim}_RAA_reg_{args.reg_strength}",map_location=device)
        model.R=torch.load(f"./pretrained_emb/directed/{dataset}/R_{latent_dim}_RAA_reg_{args.reg_strength}",map_location=device)
        
        model.latent_raa_z=model.Softmax(model.latent_z1)
        model.Gate=torch.sigmoid(model.G)
        model.C = (model.latent_raa_z * model.Gate) / (model.latent_raa_z * model.Gate).sum(0)
        model.A=(model.R.transpose(0,1)@(model.latent_raa_z.transpose(0,1)@model.C)).transpose(0,1)
                
        model.latent_z= model.latent_raa_z[0:N]@model.A
        model.latent_w= model.latent_raa_z[N:]@model.A
        
    else:
    
        for epoch in tqdm(range(args.epochs),desc="Model is Running…",ascii=False, ncols=75):                 
            
                              
            
            loss=-model.LSM_likelihood_bias_sample(epoch=epoch)/model.sample_size
    
                
            losses.append(loss.item())
            
     
         
            optimizer.zero_grad() # clear the gradients.   
            loss.backward() # backpropagate
            optimizer.step() # update the weights
            
    if args.LP:
        pred=LP_(model.latent_z,model.latent_w,model.gamma_1,model.gamma_2,model.delta_1,model.delta_2,dataset,sparse_i,sparse_j,weights_signed,device=device)
        # p@n
        p_n_roc,p_n_pr=pred.pos_neg()
        # p@z
        p_z_roc,p_z_pr=pred.pos_zer()
        # n@z
        n_z_roc,n_z_pr=pred.neg_zer()

            
    # PCA PLOTS------------------------------------------------------------------
    # Project latent space on the firt two Principal Components
    data=np.loadtxt("./datasets/directed/"+dataset+'/edges.txt')
    data[:,0:2].sort(1)
    mask=data[:,0]<data[:,1]
    data=data[mask]
    
    sparse_i=torch.from_numpy(data[:,0]).long()
    # input data, link column positions with i<j
    sparse_j=torch.from_numpy(data[:,1]).long()
    arg_max=model.Softmax(model.latent_z1).max(1)[1].detach().cpu().numpy()


    weights_signed=torch.from_numpy(data[:,2]).long()
    N=int(max(sparse_j.max(),sparse_i.max())+1)
    
    c_dict={}
    c_dict[0]='red'
    c_dict[2]='blue'
    temp_w=((weights_signed/weights_signed.abs()).cpu().numpy()+1).astype(int)


    pca = PCA(n_components=2)
    id_max=model.Softmax(model.latent_z1).max(0)[1].cpu().numpy()
    X=torch.cat((model.latent_z,model.latent_w)).detach().cpu().numpy()
    X_=pca.fit_transform(X)


    
    plt.figure(figsize=(15,15),dpi=120)

   
    plt.scatter(X_[0:N][:,0],X_[0:N][:,1],c=arg_max[0:N],s=15)
    plt.scatter(X_[N:][:,0],X_[N:][:,1],c=arg_max[N:],s=20,marker='*')
    plt.axis('off')
    plt.savefig("DIR_SLIM_RAA_PCA.png",dpi=120)    
    plt.show()

    comp=pca.components_.transpose()
    inv = np.arctan2(comp[:, 1], comp[:, 0])
    degree = np.mod(np.degrees(inv), 360)

    idxs=np.argsort(degree)

    



    step=(2*math.pi)/latent_dim
    radius=10
    points=np.zeros((latent_dim,2))
    for i in range(latent_dim): 
        points[i,0] = (radius * math.cos(i*step))
        points[i,1] = (radius * math.sin(i*step))
        
    points=points[idxs]

    latent_raa_z=model.Softmax(model.latent_z1)
       
    # plt.scatter(points[:,0],points[:,1])
    _X=latent_raa_z.detach().cpu().numpy()@points



    w_numpy=weights_signed.abs().cpu().numpy()
    # plt.scatter(_X[:,0],_X[:,1])

    plt.figure(figsize=(15,15),dpi=120)

    for i,j,w in zip(sparse_i.cpu().numpy(),sparse_j.cpu().numpy(),temp_w):
        c_i=0
        if w==0:
            plt.plot([_X[0:N][i,0], _X[N:][j,0]], [_X[0:N][i,1], _X[N:][j,1]],color=c_dict[w],lw=0.2,alpha=0.2,zorder=3)
        c_i+=c_i
    plt.scatter(_X[0:N][:,0],_X[0:N][:,1],c=arg_max[0:N],s=15)
    plt.scatter(_X[N:][:,0],_X[N:][:,1],c=arg_max[N:],s=20,marker='*')
    plt.scatter(points[:,0],points[:,1],c='black',s=100,zorder=2)
    plt.axis('off')
    plt.savefig(f"DIR_cir_{dataset}_neg.png",dpi=120)
    plt.show()





    plt.figure(figsize=(15,15),dpi=120)

    for i,j,w in zip(sparse_i.cpu().numpy(),sparse_j.cpu().numpy(),temp_w):
        c_i=0
        if w==2:
            plt.plot([_X[0:N][i,0], _X[N:][j,0]], [_X[0:N][i,1], _X[N:][j,1]],color=c_dict[w],lw=0.1,alpha=0.1,zorder=3)
        c_i+=c_i
    plt.scatter(_X[0:N][:,0],_X[0:N][:,1],c=arg_max[0:N],s=15)
    plt.scatter(_X[N:][:,0],_X[N:][:,1],c=arg_max[N:],s=20,marker='*')
    plt.scatter(points[:,0],points[:,1],c='black',s=100,zorder=2)
    plt.axis('off')
    plt.savefig(f"DIR_cir_{dataset}_pos.png",dpi=120)
    plt.show()
