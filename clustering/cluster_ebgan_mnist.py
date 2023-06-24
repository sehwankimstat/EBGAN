from __future__ import print_function

try:
    import argparse
    import os
    import numpy as np

    import matplotlib
    import matplotlib.pyplot as plt

    import pandas as pd
    
    from torch.autograd import Variable
    from torch.autograd import grad as torch_grad
    from sklearn.metrics.cluster import adjusted_rand_score

    import torch
    import torchvision
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from torchvision import datasets
    import torchvision.transforms as transforms
    from torchvision.utils import save_image
    
    from itertools import chain as ichain

    from clusgan.definitions import DATASETS_DIR, RUNS_DIR
    from clusgan.utils import save_model, calc_gradient_penalty, sample_z, cross_entropy
    from clusgan.datasets import get_dataloader, dataset_list
    from clusgan.plots import plot_train_loss

except ImportError as e:
    print(e)
    raise ImportError

def main():
    global args
    parser = argparse.ArgumentParser(description="Convolutional NN Training Script")
    parser.add_argument("-r", "--run_name", dest="run_name", default='clusebgan', help="Name of training run")
    parser.add_argument("-n", "--n_epochs", dest="n_epochs", default=200, type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", dest="batch_size", default=100, type=int, help="Batch size")
    parser.add_argument("-s", "--dataset_name", dest="dataset_name", default='mnist', choices=dataset_list,  help="Dataset name")
    parser.add_argument("-w", "--wass_metric", dest="wass_metric", action='store_true', help="Flag for Wasserstein metric")
    parser.add_argument("-g", "-–gpu", dest="gpu", default=0, type=int, help="GPU id to use")
    parser.add_argument("-k", "-–num_workers", dest="num_workers", default=1, type=int, help="Number of dataset workers")
    parser.add_argument("-se", "-–seed", dest="seed", default=1, type=int, help="seed")

    args = parser.parse_args()

    run_name = args.run_name
    dataset_name = args.dataset_name
    device_id = args.gpu
    num_workers = args.num_workers

    # Training details
    n_epochs = args.n_epochs
    batch_size = args.batch_size
    seed=args.seed

    img_size = 28
    channels = 1
   
    # Latent space info
    latent_dim = 5
    n_c = 10
    betan = 10
    betac = 10
   

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    A=40
    B=10000
    tau=1
    sep_und = '_'
    run_name_comps = ['%iepoch'%n_epochs, 'z%s'%str(latent_dim), 'bs%i'%batch_size, run_name,'seed%i'%seed,'A%i'%A,'B%i'%B, 'tau%i'%tau]
    run_name = sep_und.join(run_name_comps)

    run_dir = os.path.join(RUNS_DIR, dataset_name, run_name)
    data_dir = os.path.join(DATASETS_DIR, dataset_name)
    imgs_dir = os.path.join(run_dir, 'images')
    models_dir = os.path.join(run_dir, 'models')

    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(run_dir, exist_ok=True)
    os.makedirs(imgs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    print('\nResults to be saved in directory %s\n'%(run_dir))

    x_shape = (channels, img_size, img_size)

    cuda = True if torch.cuda.is_available() else False
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.cuda.set_device(device_id)

    def initialize_weights(net):
        for m in net.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.ConvTranspose2d):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.02)
                m.bias.data.zero_()

    
    class Reshape(nn.Module):
        """
        Class for performing a reshape as a layer in a sequential model.
        """
        def __init__(self, shape=[]):
            super(Reshape, self).__init__()
            self.shape = shape
    
        def forward(self, x):
            return x.view(x.size(0), *self.shape)
        
        def extra_repr(self):
                # (Optional)Set the extra information about this module. You can test
                # it by printing an object of this class.
                return 'shape={}'.format(
                    self.shape
                )
    
    class Generator_CNN(nn.Module):
        def __init__(self, latent_dim, n_c, x_shape):
            super(Generator_CNN, self).__init__()
    
            self.name = 'generator'
            self.latent_dim = latent_dim
            self.n_c = n_c
            self.x_shape = x_shape
            self.ngf=64

            
            self.main = nn.Sequential(

                nn.ConvTranspose2d(self.ngf * 8, self.ngf * 4, 3, 2, 1, bias=True),
                nn.BatchNorm2d(self.ngf * 4),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(self.ngf * 4, self.ngf, 4, 2, 1, bias=True),
                nn.BatchNorm2d(self.ngf),
                nn.ReLU(inplace=True),

                nn.ConvTranspose2d(self.ngf, 1, 4, 2, 1, bias=True),
                nn.Sigmoid()
            )
            
            self.gs = []
            for i in range(10):
                g = nn.Sequential(
                    # input size is z_size
                    nn.ConvTranspose2d(self.latent_dim+self.n_c, self.ngf * 8, 4, 1, 0, bias=False),
                    nn.BatchNorm2d(self.ngf * 8),
                    nn.ReLU(inplace=True),
                )
                setattr(self, 'G_{}'.format(i), g)
                self.gs.append(g)
    
    
        def forward(self, zn,zc):
            x= torch.cat((zn,zc),1)
            sp_size = (len(x) - 1) // len(self.gs) + 1
            y = []
            for _x, _g in zip(torch.split(x, sp_size, dim=0), self.gs):
                y.append(_g(_x))
            y = torch.cat(y, dim=0)
    
            output = self.main(y)
    
            return output



    class Encoder_CNN(nn.Module):
        def __init__(self, latent_dim, n_c, verbose=False):
            super(Encoder_CNN, self).__init__()
    
            self.name = 'encoder'
            self.channels = 1
            self.latent_dim = latent_dim
            self.n_c = n_c
            self.cshape = (256, 1, 1)
            self.iels = int(np.prod(self.cshape))
            self.lshape = (self.iels,)
            self.verbose = verbose
            
            self.model = nn.Sequential(
   
                nn.Conv2d(self.channels, 64, 4,2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, 2,bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(128, 256, 4, 2,bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Flatten
                Reshape(self.lshape),
                
                # Fully connected layers
                torch.nn.Linear(self.iels, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Linear(1024, latent_dim + n_c)
            )
    
            initialize_weights(self)
            
            if self.verbose:
                print("Setting up {}...\n".format(self.name))
                print(self.model)
    
        def forward(self, in_feat):
            z_img = self.model(in_feat)
            # Reshape for output
            z = z_img.view(z_img.shape[0], -1)
            # Separate continuous and one-hot components
            zn = z[:, 0:self.latent_dim]
            zc_logits = z[:, self.latent_dim:]
            zc = nn.Softmax(dim=1)(zc_logits)
            return zn, zc, zc_logits

    class Discriminator_CNN(nn.Module):
        def __init__(self):
            super(Discriminator_CNN, self).__init__()
            
            self.name = 'discriminator'
            self.channels = 1
            self.cshape = (128, 5, 5)
            self.iels = int(np.prod(self.cshape))
            self.lshape = (self.iels,)
            
            self.model = nn.Sequential(
                nn.Conv2d(self.channels, 64, 4, stride=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(64, 128, 4, stride=2, bias=True),
                nn.LeakyReLU(0.2, inplace=True),
                
                # Flatten
                Reshape(self.lshape),
                
                # Fully connected layers
                torch.nn.Linear(self.iels, 1024),
                nn.LeakyReLU(0.2, inplace=True),
                torch.nn.Linear(1024, 1),
            )
    
            self.model = nn.Sequential(self.model, torch.nn.Sigmoid())
    
            initialize_weights(self)
  
    
        def forward(self, img):
            # Get output
            validity = self.model(img)
            return validity

    bce_loss = torch.nn.BCELoss()
    xe_loss = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    
    generator = Generator_CNN(latent_dim, n_c, x_shape)
    encoder = Encoder_CNN(latent_dim, n_c)
    discriminator = Discriminator_CNN()
    
    if cuda:
        generator.cuda()
        encoder.cuda()
        discriminator.cuda()
        bce_loss.cuda()
        xe_loss.cuda()
        mse_loss.cuda()

    Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
    
    # Configure training data loader
    dataloader = get_dataloader(dataset_name=dataset_name,
                                data_dir=data_dir,
                                batch_size=batch_size,
                                num_workers=num_workers)
    
    lr_g=0.005

    ge_chain = ichain(generator.parameters(),
                      encoder.parameters())
    optimizer_D=torch.optim.SGD( discriminator.parameters(),lr=2e-4)
    #optimizer_D=torch.optim.Adam( de_chain,lr=0.0002,betas=(0.5,0.9))
    
    optimizer_GE=torch.optim.SGD( ge_chain,lr=lr_g)

    n=0
    for p in ge_chain:
        n=n+1
    
    M=[None]*n

    n=0
    for par in ge_chain:
        M[n]=torch.zeros(list(par.size())).cuda()
        n=n+1
    
    beta_1=0.9
    a=1
    N=60000

    
    
    # ----------
    #  Training
    # ----------
    g_l = []
    d_l = []
    e_l = []
    c_zn = []
    c_zc = []
    c_i = []
    
    # Training loop 

    
    import torchvision.utils as vutils
    
    #plt.imshow(np.transpose(vutils.make_grid(gen_imgs.detach().cpu(), padding=2, normalize=True,nrow=10),(1,2,0)))
    
    J_g=10
    import torchvision.datasets as dset
    
    train_dataset = dset.MNIST(root='./data',train=True,
                              transform=transforms.Compose([
                                  transforms.ToTensor(),
                              ]),download=True)
    
    x_train=torch.zeros((60000,1,28,28))
    y_true=(torch.rand(60000, 1) * J_g).type(torch.LongTensor).squeeze()
    y_train= (torch.rand(60000, 1) * J_g).type(torch.LongTensor).squeeze()
    
    
    for i in range(60000):
      x_train[i,0,:,:]=train_dataset[i][0]
      y_true[i]=train_dataset[i][1]
    
    index = [0] * 10
    data = [0] * 10
    label = [0] * 10
    class_start = [0] * 10
    
    x = x_train
    for i in range(10):
        index[i] = ((y_true == i).nonzero())
        data[i] = (x[torch.squeeze(index[i])])
        label[i] = torch.FloatTensor(np.tile(np.array(([i])), (data[i].size()[0], 1)))  # 1
        class_start[i] = label[i].size()[0]
    
    
    y_label = torch.cat(
        [label[0], label[1], label[2], label[3], label[4], label[5], label[6], label[7], label[8], label[9]], dim=0)
    x = (torch.cat([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7], data[8], data[9]]))
    
    
    print('\nBegin training session with %i epochs...\n'%(n_epochs))
    
    
    iter=0
    x_train=x_train.cuda()

    valid = Variable(Tensor(batch_size, 1).fill_(1.0), requires_grad=False)
    fake = Variable(Tensor(batch_size, 1).fill_(0.0), requires_grad=False)
    for epoch in range(n_epochs):
        for i, (imgs, itruth_label) in enumerate(dataloader):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            lr_d=A*(iter+B)**(-3/4)
    
            for param_group in optimizer_D.param_groups:
              param_group['lr'] = lr_d
    
    
            # Ensure generator/encoder are trainable
            generator.train()
            encoder.train()
            # Zero gradients for models
            generator.zero_grad()
            encoder.zero_grad()
            discriminator.zero_grad()
            
            # Configure input
            real_imgs = Variable(imgs.type(Tensor))
    
            

            optimizer_D.zero_grad()
            
            # Sample random latent variables
            stack_zn = []
            stack_zc = []
            stack_zc_idx =[]
            for idx in range(n_c):
              zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                      latent_dim=latent_dim,
                                                      n_c=n_c,
                                                      fix_class=idx)
    
              if (len(stack_zn) == 0):
                stack_zn = zn_samp
                stack_zc = zc_samp
                stack_zc_idx= zc_samp_idx
              else:
                stack_zn = torch.cat((stack_zn,zn_samp),0)
                stack_zc =  torch.cat((stack_zc,zc_samp),0)
                stack_zc_idx= torch.cat((stack_zc_idx,zc_samp_idx),0)
    
            stack_zn=stack_zn.reshape((batch_size,latent_dim,1,1))
            stack_zc=stack_zc.reshape((batch_size,n_c,1,1))
            gen_imgs=generator(stack_zn,stack_zc)
                
          
            # Discriminator output from real and generated samples
            D_gen = discriminator(gen_imgs)        
            D_real = discriminator(real_imgs)
    
            real_loss = bce_loss(D_real, valid)
            fake_loss = bce_loss(D_gen, fake)
            d_loss = real_loss + fake_loss
    
            d_loss.backward(retain_graph=True)
            optimizer_D.step()
          
    
            # ---------------------
            #  Train Generator
            # ---------------------
    
            optimizer_GE.zero_grad()
            
            # Encode the generated images
            enc_gen_zn, enc_gen_zc, enc_gen_zc_logits = encoder(gen_imgs)

            zn_loss = mse_loss(enc_gen_zn, stack_zn.squeeze())
            zc_loss = xe_loss(enc_gen_zc_logits, stack_zc_idx)    
            v_loss = bce_loss(D_gen, valid)


            l2=0.0
            for p in ge_chain:
                l2+=(p**2).sum()/20

            g_loss = v_loss + betan * zn_loss + betac * zc_loss+l2/N
    
    
            g_loss.backward()
            optimizer_GE.step()
    
    
            n=0
            for par in ge_chain:
              par.data.sub_(a*M[n]*lr_g/N)
              n=n+1
        
    
            with torch.no_grad():
              for param in ge_chain:
                  param.add_(torch.randn(param.size()).cuda() * np.sqrt(2*tau*lr_g/N))
          
            n=0
            for par in ge_chain:
                M[n]*=beta_1
                M[n]+=(1-beta_1)*par.grad*N
                n=n+1
            
            iter+=1
    
        # Save training losses
        d_l.append(d_loss.item())
        g_l.append(g_loss.item())
    
        # Generator in eval mode
        generator.eval()
        encoder.eval()
    
    
        
        ## Generate samples for specified classes
        stack_zn = []
        stack_zc = []
        stack_zc_idx =[]
        for idx in range(n_c):
          zn_samp, zc_samp, zc_samp_idx = sample_z(shape=n_c,
                                                  latent_dim=latent_dim,
                                                  n_c=n_c,
                                                  fix_class=idx)
    
          if (len(stack_zn) == 0):
            stack_zn = zn_samp
            stack_zc = zc_samp
            stack_zc_idx= zc_samp_idx
          else:
            stack_zn = torch.cat((stack_zn,zn_samp),0)
            stack_zc =  torch.cat((stack_zc,zc_samp),0)
            stack_zc_idx= torch.cat((stack_zc_idx,zc_samp_idx),0)
    
        stack_zn=stack_zn.reshape((batch_size,latent_dim,1,1))
        stack_zc=stack_zc.reshape((batch_size,n_c,1,1))
        gen_imgs=generator(stack_zn,stack_zc)
    
    
    
        # Save class-specified generated examples!
        save_image(gen_imgs,
                    '%s/gen_classes_%06i.png' %(imgs_dir, epoch), 
                    nrow=n_c, normalize=True)
      
    
        print ("[Epoch %d/%d] \n"\
                "\tModel Losses: [D: %f] [G: %f]" % (epoch, 
                                                      n_epochs, 
                                                      d_loss.item(),
                                                      g_loss.item())
              )

    # Save current state of trained models
        if epoch>194:   
            m=epoch-194
            models_dirr = os.path.join(models_dir, '%i'%(m))
            if not os.path.isdir(models_dirr):
              os.mkdir(models_dirr)
                
            filename = generator.name + '.pth.tar'
            outfile = os.path.join(models_dirr, filename)
            torch.save(generator.state_dict(), outfile)
            filename = encoder.name + '.pth.tar'
            outfile = os.path.join(models_dirr, filename)
            torch.save(encoder.state_dict(), outfile)
            
if __name__ == "__main__":
    main()
