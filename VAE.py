import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torchvision.utils import save_image
from model import VAE, VAEFC
from sklearn.manifold import TSNE 
import matplotlib.pyplot as plt
from matplotlib import cm
import numpy as np


# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a directory if not exists
sample_dir = 'samples'
tsne_dir = 'tsne'
if not os.path.exists(sample_dir):
    os.makedirs(sample_dir)
    if not os.path.exists(tsne_dir):
        os.makedirs(tsne_dir)

# Hyper-parameters
image_size = 784
h_dim = 400
z_dim = 20
num_epochs = 15
batch_size = 32
learning_rate = 1e-3

# MNIST dataset
dataset = torchvision.datasets.MNIST(root='./MNIST',
                                     train=True,
                                     transform=transforms.ToTensor(),
                                     download=True)

# Data loader
data_loader = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=batch_size, 
                                          shuffle=True)

# for tsne
draw_data = torch.utils.data.DataLoader(dataset=dataset,
                                          batch_size=2000, 
                                          shuffle=True)

# VAE model
model = VAE().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Start training
for epoch in range(num_epochs):
    #embs=[]
    #label=[]

    for i, (x, y) in enumerate(data_loader):
        # Forward pass
        #x = x.to(device).view(-1, image_size) #reshape the tensor
        x = x.to(device).view(batch_size, 1, 28, 28)
        x_reconst, mu, log_var = model(x)
        
        # Compute reconstruction loss and kl divergence
        reconst_loss = F.binary_cross_entropy(x_reconst, x, reduction='sum') #size_average=False
        kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        
        # Backprop and optimize
        loss = reconst_loss + kl_div
        optimizer.zero_grad() #clear
        loss.backward()
        optimizer.step()
        
        '''
        tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
        low_dim_embs = tsne.fit_transform(mu.detach().cpu().numpy())
        label.append(y)
        embs.append(low_dim_embs)
        '''

        if (i+1) % 100 == 0:
            #print("mu:") # [128,20]
            #print(mu,mu.shape)
            #print("x:") # [128,784]
            #print(x,x.shape)
            #print("l:") # [128]
            #print(l,l.shape)
            print ("Epoch[{}/{}], Step [{}/{}], Reconst Loss: {:.4f}, KL Div: {:.4f}" 
                   .format(epoch+1, num_epochs, i+1, len(data_loader), reconst_loss.item(), kl_div.item()))
            
            #plt.figure(figsize=(15, 15))
            #plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1], 20, labels)
            #plt.scatter(low_dim_embs[:,0], low_dim_embs[:,1])
            #plt.savefig('./tsne/tsne-{}.png'.format(epoch+1))
            #plt.show()
            #plt.ioff()
    '''
    for i in range(np.array(embs).shape[0]):
        plt.text(embs[i, 0], embs[i, 1], str(label[i]), color=plt.cm.Set1(label[i]), 
             fontdict={'weight': 'bold', 'size': 9})
    plt.scatter(embs[:,0], embs[:,1], label)
    plt.savefig('tsne-{}.png'.format(epoch+1))
    '''
    '''
    x_min, x_max = np.array(embs).min(0), np.array(embs).max(0)
    X_norm = (np.array(embs) - x_min) / (x_max - x_min)
    plt.figure(figsize=(10, 10))
    #plt.plot(X_norm[0], X_norm[1])
    for j in range(X_norm.shape[0]):
        plt.text(X_norm[j, 0], X_norm[j, 1], str(label[j]).strip('()'), color=plt.cm.Set1(label[j]), fontdict={'weight': 'bold', 'size': 9})
        plt.xticks([])
        plt.yticks([])
        # plt.show()
        plt.savefig('./tsne/epoch-{}.png'.format(epoch))
    '''
            

    
    with torch.no_grad():
        # Save the sampled images
        z = torch.randn(batch_size, z_dim).to(device) #guasion
        out = model.decode(z).view(-1, 1, 28, 28)
        save_image(out, os.path.join(sample_dir, 'sampled-{}.png'.format(epoch+1)))

        # Save the reconstructed images
        out, _, _ = model(x)
        x_concat = torch.cat([x.view(-1, 1, 28, 28), out.view(-1, 1, 28, 28)], dim=3)
        save_image(x_concat, os.path.join(sample_dir, 'reconst-{}.png'.format(epoch+1)))

        #colors = 'r', 'g', 'b', 'c', 'm', 'y', 'k', 'w', 'orange', 'purple'
        for i, (data, lab) in enumerate(draw_data):
            data = data.to(device).view(-1, 1, 28, 28)
            _, vec, _= model(data)
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=1000)
            low_dim_embs = tsne.fit_transform(vec.detach().cpu().numpy())
           
            x_min, x_max = low_dim_embs.min(0), low_dim_embs.max(0)
            X_norm = (low_dim_embs - x_min) / (x_max - x_min)
            plt.figure(figsize=(10, 10))
            #plt.plot(X_norm[0], X_norm[1])
            '''
            for i, c, label in zip(target_ids, colors, digits.target_names):
                plt.scatter(X_2d[y == i, 0], X_2d[y == i, 1], c=c, label=label)
            plt.scatter(X_norm[:0],X_norm[:1],marker="o")
            '''
            for j in range(X_norm.shape[0]):
                plt.text(X_norm[j, 0], X_norm[j, 1], str(lab[j]).strip('()'), color=plt.cm.Set1(lab[j]), fontdict={'weight': 'bold', 'size': 9})
            plt.xticks([])
            plt.yticks([])
            plt.savefig('./tsne/epoch-{}.png'.format(epoch))
            break
        