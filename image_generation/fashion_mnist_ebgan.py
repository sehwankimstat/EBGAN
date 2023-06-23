
# !pip install pot


import torch
import torch.nn as nn
from torch.autograd import Variable

import torchvision
import torchvision.transforms as transforms
import torchvision.utils as vutils


import numpy as np

import random
from matplotlib import pyplot as plt

################ For KL-prior ######################

def knn_distance(point, sample, k):
    """ Euclidean distance from `point` to it's `k`-Nearest
    Neighbour in `sample` """
    norms = torch.linalg.norm(sample-point, axis=1)
    return torch.sort(norms)[0][k]


def verify_sample_shapes(s1, s2, k):
    # Expects [N, D]
    assert(len(s1.shape) == len(s2.shape) == 2)
    # Check dimensionality of sample is identical
    assert(s1.shape[1] == s2.shape[1])


def naive_estimator(s1, s2, k=1):
    """ KL-Divergence estimator using brute-force (numpy) k-NN
        s1: (N_1,D) Sample drawn from distribution P
        s2: (N_2,D) Sample drawn from distribution Q
        k: Number of neighbours considered (default 1)
        return: estimated D(P|Q)
    """
    verify_sample_shapes(s1, s2, k)

    n, m = len(s1), len(s2)
    D = np.log(m / (n - 1))
    d = float(s1.shape[1])

    for p1 in s1:
        nu = knn_distance(p1, s2, k-1)  # -1 because 'p1' is not in 's2'
        rho = knn_distance(p1, s1, k)
        D += (d/n)*torch.log(nu/rho)
    return D

############################################################

manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)

transform = transforms.Compose(
    [#transforms.RandomHorizontalFlip(),
     transforms.ToTensor(),
     #transforms.Resize(32),
     ])


trainset = torchvision.datasets.FashionMNIST(root='./data', train=True,
                                        download=True, transform=transform)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch=300

dataloader= torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)

dataloader2= torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True, num_workers=2)

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
            nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 3, 2, 1,bias=False),
            nn.BatchNorm2d(self.ngf * 4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf * 4, self.ngf*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(self.ngf*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(self.ngf*2, 1, 4, 2, 1,bias=False),
            nn.Sigmoid()
        )

        self.gs = []
        for i in range(10):
            g = nn.Sequential(
                # input size is z_size
                nn.ConvTranspose2d(self.latent_dim, self.ngf * 8, 4, 1, 0,bias=False),
                nn.BatchNorm2d(self.ngf * 8),
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
    def __init__(self, wass_metric=False, verbose=False):
        super(Discriminator_CNN, self).__init__()

        self.name = 'discriminator'
        self.channels = 1
        self.wass = wass_metric
        self.verbose = verbose


        self.ndf=128
        self.model = nn.Sequential(

            nn.Conv2d(self.channels, self.ndf, 4,2,1,bias=False),
            nn.LeakyReLU(0.2, inplace=True),

            # ndf x 32 x 32
            nn.Conv2d(self.ndf, self.ndf*2, 4, 2, 1,bias=False),
            nn.BatchNorm2d(self.ndf*2),
            nn.LeakyReLU(0.2, inplace=True),

            # (ndf * 2) x 16 x 16
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

############## Train ################

############## EBGAN Setting ##############

latent_dim=10
n_c = 1

x_shape=(1,28,28)

generator = Generator_CNN(latent_dim, x_shape).to(device)
discriminator = Discriminator_CNN().to(device)

A=1         ### Discriminator learning rate
B=500
initial_lr_d=0.002

lr_g=0.01   ### Generator learning rate



optimizer_D=torch.optim.Adam( discriminator.parameters(),lr=initial_lr_d,betas=(0.5,0.999))
optimizer_G=torch.optim.SGD( generator.parameters(),lr=lr_g)

n=0
for p in generator.parameters():
    n=n+1

M=[None]*n
n=0
for par in generator.parameters():
    M[n]=torch.zeros(list(par.size())).cuda()
    n=n+1


N=60000
a=1
beta_1=0.9
n_epochs=200
tau=0.001

bce_loss=torch.nn.BCELoss()

iteration=0

Tensor=torch.cuda.FloatTensor
real = Variable(Tensor(batch).fill_(1.0), requires_grad=False)
fake = Variable(Tensor(batch).fill_(0.0), requires_grad=False)

Gen_Dis=torch.zeros(int(N/batch*n_epochs/100))
Real_Dis=torch.zeros(int(N/batch*n_epochs/100))
js=0
###########################################

for epoch in range(n_epochs):

  for i, ((imgs, itruth_label),(imgs2,itruth_label2)) in enumerate(zip(dataloader,dataloader2)):

      # ---------------------
      #  Train Discriminator
      # ---------------------
      lr_d=A*(iteration+B)**(-1)

      for param_group in optimizer_D.param_groups:
          param_group['lr'] = lr_d

      generator.zero_grad()
      discriminator.zero_grad()

      real_imgs = Variable(imgs.type(Tensor))

      optimizer_D.zero_grad()

      # Sample random latent variables
      gen_imgs=generator(torch.randn((batch,latent_dim,1,1)).to(device))
      D_gen = discriminator(gen_imgs)
      D_real = discriminator(real_imgs)

      real_loss = bce_loss(D_real, real)
      fake_loss = bce_loss(D_gen, fake)
      d_loss = (real_loss + fake_loss)

      d_loss.backward()
      optimizer_D.step()


      # ---------------------
      #  Train Generator
      # ---------------------

      optimizer_G.zero_grad()
      gen_imgs=generator(torch.randn((batch,latent_dim,1,1)).to(device))

      D_gen = discriminator(gen_imgs)
      v_loss = bce_loss(D_gen, real)

      l2=0.0
      for p in generator.parameters():
          l2+=(p**2).sum()*20
      g_loss=v_loss+l2/N
      #g_loss = v_loss +naive_estimator(real_imgs.view((batch,28*28)),gen_imgs.view((batch,28*28)),k=1)/N*100

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

      ### Nash equilibrium check
      if iteration%100==1:
        Gen_Dis[js]=torch.mean(discriminator(generator(torch.randn((batch,latent_dim,1,1)).cuda()))).detach().cpu().item()
        Real_Dis[js]=torch.mean(discriminator(Variable(imgs2.type(Tensor)))).detach().cpu().item()
        js+=1

  print("epoch",epoch,"g_loss",g_loss.item(),"d_loss",d_loss.item(),"E(D(fake))",torch.mean(D_gen).item(),"E(D(real))",torch.mean(D_real).item())

#### Check the generated images

number=100
x_batch = real_imgs[0:number]
test=generator(torch.randn((number,latent_dim,1,1)).cuda())

fig, (ax1) = plt.subplots(1, 1,figsize=(8,8))
ax1.imshow(np.transpose(vutils.make_grid(test.detach().cpu()[0:number],padding=2,normalize=True,nrow=10),(1,2,0)))
ax1.set_title("Generated images: EBGAN",fontsize=13)
ax1.set_xticks([])
ax1.set_yticks([])
#plt.show()
fig.savefig('fashion_mnist_generated.png')

fig, (ax1) = plt.subplots(1, 1,figsize=(8,5))
ax1.set_title("Expectation value: EBGAN")
ax1.plot(np.multiply(range(int(N/batch*n_epochs/100)),100)+1,Real_Dis,label="Real",color='blue',linestyle='--')
ax1.plot(np.multiply(range(int(N/batch*n_epochs/100)),100)+1,Gen_Dis,label="Fake",color='black',linestyle='-')
ax1.set_xlabel('No. of iterations')
ax1.set_ylabel('Expectation of Discriminator Value')
ax1.legend()
#plt.show()
fig.savefig('fashion_mnist_convergence.png')

