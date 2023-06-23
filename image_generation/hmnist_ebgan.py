import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils

import numpy as np
import random

import pandas as pd


from matplotlib import pyplot as plt

#################### Seed  #####################
torch.manual_seed(1235)
np.random.seed(1235)
torch.cuda.manual_seed_all(1235)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark=False
random.seed(1235)
################################################

########## Unbalanced--> Balanced ##############
from sklearn.utils import resample

def balanced_dataset(df):
    df_balanced = pd.DataFrame()
    #df = pd.DataFrame()
    unique_label=df['label'].unique()
    for l in unique_label:
        current_number=len(df[df['label'] == l])
        temp = resample(df[df['label'] == l],
                        replace=True,     # sample with replacement
                        n_samples=7000-current_number,   # to match majority class
                        random_state=123) # reproducible results
        df_balanced = pd.concat([df_balanced, temp], ignore_index=True)
        df_balanced = pd.concat([df_balanced, df[df['label'] == l]], ignore_index=True)
    return df_balanced


df =pd.read_csv('/content/hmnist_28_28_RGB.csv')
df=balanced_dataset(df)

label=df['label']
df.drop('label',axis=1,inplace=True)
X=df.values.astype(np.uint8).reshape(-1,28,28,3)

################################################


################ Preparing Dataset #############

from torch.utils.data import Dataset, DataLoader
class MyDataset(Dataset):
    def __init__(self, data,target,transform=None):
        self.data = data
        self.target= target
        self.transform = transform

    def __getitem__(self, index):
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = self.transform(x)

        return x,y

    def __len__(self):
        return len(self.data)

# image_size=32     # Change the image size

train_transform=transforms.Compose([transforms.ToPILImage(),
                              #transforms.Resize((image_size,image_size)),
                              transforms.ToTensor(),
                              transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = MyDataset(X,label,transform=train_transform)



batch_size = 200
trainloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                         shuffle=True)


################################################

############## NN structures ###################

def initialize_weights(net):
    for m in net.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.ConvTranspose2d):
            m.weight.data.normal_(0, 0.02)
            #m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.02)
            m.bias.data.zero_()
        elif isinstance(m,nn.BatchNorm2d):  # BatchNorm weight init
            m.weight.data.normal_(1.0, 0.02)
            m.bias.data.fill_(0)

class Generator_CNN(nn.Module):
    def __init__(self, latent_dim, x_shape):
        super(Generator_CNN, self).__init__()

        self.name = 'generator'
        self.latent_dim = latent_dim
        self.n_c = n_c
        self.x_shape = x_shape
        self.ngf=64

        self.main = nn.Sequential(
            nn.ConvTranspose2d(self.ngf * 4, self.ngf*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*2, 3, 4, 2, 1,bias=False),
            nn.Tanh()
        )

        self.gs = []
        for i in range(7):
            g = nn.Sequential(
                # input size is z_size
                nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0,bias=False),
                nn.BatchNorm2d(self.ngf * 8),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 3, 2, 1,bias=False),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(inplace=True),

            )
            setattr(self, 'G_{}'.format(i), g)
            self.gs.append(g)
        initialize_weights(self)


    def forward(self, zn):
        x= zn
        sp_size = (len(x) - 1) // len(self.gs) + 1
        y = []
        for _x, _g in zip(torch.split(x, sp_size, dim=0), self.gs):
            y.append(_g(_x))
        y = torch.cat(y, dim=0)

        output = self.main(y)

        return output



class Discriminator_CNN(nn.Module):
    def __init__(self):
        super(Discriminator_CNN, self).__init__()

        self.name = 'discriminator'
        self.channels =3
        self.ndf=128
        self.model = nn.Sequential(

            nn.Conv2d(self.channels, self.ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*2, self.ndf*4, 3, 2, 1,bias=False),
            nn.BatchNorm2d(self.ndf*4),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(self.ndf*4, 1, 4, 1, 0,bias=False),

        )

        self.model = nn.Sequential(self.model, torch.nn.Sigmoid())

        initialize_weights(self)

    def forward(self, img):
        # Get output
        validity = self.model(img)
        return validity.view(-1, 1).squeeze(1)

################################################
import os
current_path = os.getcwd()
directory = "HMNIST_ebgan"
result_path = os.path.join(current_path, directory)
if not os.path.isdir(result_path):
  os.mkdir(result_path)

################# EBGAN training ###############


N=len(X)
n_c=3
latent_dim=20

x_shape=(3,28,28)
generator = Generator_CNN(latent_dim, x_shape).cuda()
discriminator = Discriminator_CNN().cuda()



A=0.05
B=250
lr_g=0.001

optimizer_D=torch.optim.Adam( discriminator.parameters(),lr=2e-4,betas=(0.5,0.999))
optimizer_G=torch.optim.SGD( generator.parameters(),lr=lr_g)

bce_loss=torch.nn.BCELoss()
Tensor=torch.cuda.FloatTensor

# MSGLD setting ###################
a=1
beta_1=0.9
tau=0.001

n=0
for p in generator.parameters():
    n=n+1

M=[None]*n

n=0
for par in generator.parameters():
    M[n]=torch.zeros(list(par.size())).cuda()
    n=n+1

####################################

n_epochs=200
Gen_Dis=torch.zeros(int(N/batch_size*n_epochs/100))
Real_Dis=torch.zeros(int(N/batch_size*n_epochs/100))

generated_images=[]
js=0
iteration=0
for epoch in range(n_epochs):
    for i, ((imgs, itruth_label)) in enumerate(trainloader):
        # ---------------------
        #  Train Discriminator
        # ---------------------

        lr_d=A*(iteration+B)**(-1)

        for param_group in optimizer_D.param_groups:
          param_group['lr'] = lr_d

        batch=len(itruth_label)
        real = Variable(Tensor(batch).fill_(1.0), requires_grad=False)
        fake = Variable(Tensor(batch).fill_(0.0), requires_grad=False)

        generator.zero_grad()
        discriminator.zero_grad()

        # Configure input
        real_imgs = Variable(imgs.type(Tensor))

        optimizer_D.zero_grad()

        # Sample random latent variables
        gen_imgs=generator(torch.randn((batch,latent_dim,1,1)).cuda())


        D_gen = discriminator(gen_imgs)
        D_real = discriminator(real_imgs)

        real_loss = bce_loss(D_real,  real)
        fake_loss = bce_loss(D_gen, fake)
        d_loss = (real_loss + fake_loss)


        d_loss.backward()
        optimizer_D.step()


        # ---------------------
        #  Train Generator
        # ---------------------

        optimizer_G.zero_grad()
        gen_imgs=generator(torch.randn((batch,latent_dim,1,1)).cuda())


        D_gen = discriminator(gen_imgs)
        v_loss = bce_loss(D_gen, real)
        l2=0.0
        for p in generator.parameters():
            l2+=(p**2).sum()*40

        g_loss = v_loss +l2/N


        g_loss.backward()
        optimizer_G.step()


        n=0
        for par in generator.parameters():
          par.data.sub_(a*M[n]*lr_g/N)
          n=n+1


        with torch.no_grad():
          for param in generator.parameters():
              param.add_(torch.randn(param.size()).cuda() * np.sqrt(2*tau*lr_g/N))

        n=0
        for par in generator.parameters():
            M[n]*=beta_1
            M[n]+=(1-beta_1)*par.grad*N
            n=n+1

        iteration+=1
        if iteration%100==1:
          Gen_Dis[js]=torch.mean(discriminator(generator(torch.randn((batch,latent_dim,1,1)).cuda()))).detach().cpu().item()
          Real_Dis[js]=torch.mean(discriminator(Variable(imgs.type(Tensor)))).detach().cpu().item()
          js+=1

    if js==765:
      break
    print("epoch",epoch,"g_loss",g_loss.item(),"d_loss",d_loss.item(),"E(D(fake))",torch.mean(D_gen).item(),"E(D(real))",torch.mean(D_real).item())

    ######### Results saving ############

    if epoch>100:
        test=generator(torch.randn((500,latent_dim,1,1)).cuda()).detach().cpu().numpy()
        generated_images.extend(test)

    noise=torch.randn((7,latent_dim,1,1)).cuda().repeat(7,1,1,1)
    gen_imgs =generator(noise)
    fig, (ax1) = plt.subplots(1, 1,figsize=(5,5))
    ax1.imshow(np.transpose(vutils.make_grid(gen_imgs.detach().cpu(),padding=2,normalize=True,nrow=7),(1,2,0)))
    ax1.set_title("Generated images: EBGAN, epoch%s"%epoch,fontsize=13)
    ax1.set_xticks([])
    ax1.set_yticks([])
    plt.show()
    fig.savefig(os.path.join(result_path,'generated_hmnist_epoch%s.png'%epoch))


############# Track the generated images #########################
from pathlib import Path
import imageio

png_dir = result_path
images = []
for i, file_name in enumerate(sorted(Path(png_dir).iterdir(), key=os.path.getmtime)):
    
    if file_name.name.endswith('.png') and i%10==0:
        file_path = os.path.join(png_dir, file_name.name)
    
        images.append(imageio.imread(file_path))
        print(file_name)

# Make it pause at the end so that the viewers can ponder
for _ in range(10):
    images.append(imageio.imread(file_path))


imageio.mimsave('hmnist_ebgan_images.gif', images)



################# Convergence check ###############

plt.title("Expectation value: EBGAN")
plt.plot(np.multiply(range(int(N/batch_size*n_epochs/100)),100)+1,Real_Dis,label="Real",color='blue',linestyle='--')
plt.plot(np.multiply(range(int(N/batch_size*n_epochs/100)),100)+1,Gen_Dis,label="Fake",color='black',linestyle='-')
plt.xlabel('No. of iterations')
plt.ylabel('Expectation of Discriminator Value')
plt.legend()
plt.savefig("HMNIST_ebgan_convergence.png")

################# SSIM value check ###############
# ! pip install pytorch-msssim
number=49
test=generator(torch.randn((number,latent_dim,1,1)).cuda())

generated=(test+1)/2

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
import seaborn as sns
ssim_matrix=torch.zeros((7,7))
rmse_matrix=torch.zeros((7,7))

ssim_vector=[]
for i in range(1,49):
  for j in range(i+1,49):
    ssim_vector.append((ssim( generated[i].view(1,3,28,28), generated[j].view(1,3,28,28), data_range=1, size_average=False)).data) # return (N,)
ssim_vector=torch.stack(ssim_vector)
plt.hist(ssim_vector.detach().cpu().numpy().reshape(1128),bins=20)
plt.xlim(0,1)
plt.xlabel("SSIM values", size=14)
plt.ylabel("Count", size=14)
plt.title("Image variety by SSIM: EBGAN")
plt.legend(loc='upper right')
plt.savefig("HMNIST_ebgan_ssim.png")

