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
plt.set_cmap("tab10")



sys.path.append('./src/')

from skellam_LDM_RE_RAA import SLIM_RAA_und
from link_prediction import LP_

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
                      choices=[True, False], default=False,
                    help='Uses pretrained embeddings for link prediction (default: False)')

parser.add_argument('--D', type=int, default=8, metavar='N',
                    help='dimensionality of the embeddings (default: 8)')

parser.add_argument('--lr', type=float, default=0.05, metavar='N',
                    help='learning rate for the ADAM optimizer, for large values of delta 0.01 is more stable (default: 0.05)')

parser.add_argument('--sample_percentage', type=float, default=0.3, metavar='N',
                    help='Sample size network percentage, it should be equal or less than 1 (default: 0.3)')



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
    data=np.loadtxt("./datasets/undirected/"+dataset+'/edges.txt')
    data[:,0:2].sort(1)
    mask=data[:,0]<data[:,1]
    data=data[mask]
    
    sparse_i=torch.from_numpy(data[:,0]).long().to(device)
    # input data, link column positions with i<j
    sparse_j=torch.from_numpy(data[:,1]).long().to(device)
 

    weights_signed=torch.from_numpy(data[:,2]).long().to(device)
    
   
    
    # network size
    N=int(sparse_j.max()+1)
    # sample size of blocks-> sample_size*(sample_size-1)/2 pairs
    
    sample_size=int(args.sample_percentage*N)
    model = SLIM_RAA_und(sparse_i,sparse_j,weights_signed,N,latent_dim=latent_dim,sample_size=sample_size,device=device).to(device)         
    
    # create initial convex hull
    model.find_convex_hull()
    # initialize SLIM-RAA model
    model.LDM_to_RAA()
    
    # set-up optimizer
    optimizer = optim.Adam(model.parameters(), args.lr)  

    if args.pretrained:
        model.latent_z=torch.load('./pretrained_emb/'+dataset+f'/Z_{latent_dim}_RAA',map_location=device)
        model.gamma=torch.load('./pretrained_emb/'+dataset+f'/gamma_{latent_dim}_RAA',map_location=device)
        model.delta=torch.load('./pretrained_emb/'+dataset+f'/delta_{latent_dim}_RAA',map_location=device)
        
        model.latent_z1=torch.load(f"./pretrained_emb/{dataset}/Z1_{latent_dim}_RAA",map_location=device)
        model.G=torch.load(f"./pretrained_emb/{dataset}/G_{latent_dim}_RAA",map_location=device)
        model.R=torch.load(f"./pretrained_emb/{dataset}/R_{latent_dim}_RAA",map_location=device)

        
        
        
    else:
    
        for epoch in tqdm(range(args.epochs),desc="Model is Running…",ascii=False, ncols=75):                 
            
                              
            
            loss=-model.LSM_likelihood_bias_sample(epoch=epoch)/model.sample_size
    
                
            losses.append(loss.item())
            
     
         
            optimizer.zero_grad() # clear the gradients.   
            loss.backward() # backpropagate
            optimizer.step() # update the weights
            
    if args.LP:
        pred=LP_(model.latent_z,model.gamma,model.delta,dataset,sparse_i,sparse_j,weights_signed,device=device)
        # p@n
        p_n_roc,p_n_pr=pred.pos_neg()
        # p@z
        p_z_roc,p_z_pr=pred.pos_zer()
        # n@z
        n_z_roc,n_z_pr=pred.neg_zer()

            
    # PCA PLOTS------------------------------------------------------------------
    # Project latent space on the firt two Principal Components
    latent_raa_z=model.Softmax(model.latent_z1)

    
    c_dict={}
    c_dict[0]='red'
    c_dict[2]='blue'
    temp_w=((weights_signed/weights_signed.abs()).cpu().numpy()+1).astype(int)


    pca = PCA(n_components=2)
    id_max=model.Softmax(model.latent_z1).max(0)[1].cpu().numpy()
    X=model.latent_z.detach().cpu().numpy()
    X_=pca.fit_transform(X)

    arg_max=latent_raa_z.max(1)[1].detach().cpu().numpy()

    plt.figure(figsize=(15,15),dpi=300)

    # for i,j,w in zip(sparse_i.cpu().numpy(),sparse_j.cpu().numpy(),temp_w):
    #     plt.plot([X_[i,0], X_[j,0]], [X_[i,1], X_[j,1]],color=c_dict[w],lw=0.01,alpha=0.2)
    # plt.scatter(A[:,0],A[:,1],c='black')
    plt.scatter(X_[:,0],X_[:,1],c=arg_max,s=20)
    plt.axis('off')
    plt.savefig("SLIM_RAA_PCA.png",dpi=300,bbox_inches = 'tight')    
    plt.show()

    
    
    # CIRCULAR PLOTS------------------------------------------------------------------- 
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

       
    # plt.scatter(points[:,0],points[:,1])
    _X=latent_raa_z.detach().cpu().numpy()@points


    print('CREATING and SAVING files (please wait a couple of minutes)\n')
    plt.figure(figsize=(15,15),dpi=120)

    for i,j,w in zip(sparse_i.cpu().numpy(),sparse_j.cpu().numpy(),temp_w):
        if w==0:
            plt.plot([_X[i,0], _X[j,0]], [_X[i,1], _X[j,1]],color=c_dict[w],lw=0.15,alpha=0.25,zorder=2)
    plt.scatter(_X[:,0],_X[:,1],c=arg_max,s=20)
    plt.scatter(points[:,0],points[:,1],c='black',s=100)
    plt.axis('off')
    plt.savefig(f"cir_{dataset}_neg.png",dpi=120)
    plt.show()




    plt.figure(figsize=(15,15),dpi=120)

    for i,j,w in zip(sparse_i.cpu().numpy(),sparse_j.cpu().numpy(),temp_w):
        if w==2:
            plt.plot([_X[i,0], _X[j,0]], [_X[i,1], _X[j,1]],color=c_dict[w],lw=0.1,alpha=0.25,zorder=2)
    plt.scatter(_X[:,0],_X[:,1],c=arg_max,s=20)
    plt.scatter(points[:,0],points[:,1],c='black',s=100)
    plt.axis('off')
    plt.savefig(f"cir_{dataset}_pos.png",dpi=120)
    plt.show()
    print('DONE!')
        
        
