
import os

import matplotlib.pyplot as plt
import torch
from torch.distributions import Normal, Laplace
import torchdiffeq
import torchsde
from torchdyn.core import NeuralODE
from torchvision import datasets, transforms
from torchvision.transforms import ToPILImage
from torchvision.utils import make_grid
from tqdm import tqdm
import wandb 

from torchcfm.conditional_flow_matching import *
from torchcfm.models.unet import UNetModel, UNetModelGuided,EncoderUNetModel
#from torchcfm.models.unet.enc import *

savedir = "models/cond_mnist"
os.makedirs(savedir, exist_ok=True)

# %%
use_cuda = torch.cuda.is_available()
device = torch.device("cuda:0" if use_cuda else "cpu")
batch_size = 128
n_epochs = 10
latent_dim = 32

trainset = datasets.MNIST(
    "../data",
    train=True,
    download=True,
    transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))]),
)

train_loader = torch.utils.data.DataLoader(
    trainset, batch_size=batch_size, shuffle=True, drop_last=True
)

# %%
#################################
#    Class Conditional CFM
#################################

sigma = 0.0
model = UNetModel(
    dim=(1, 28, 28), 
    num_channels=32, 
    num_res_blocks=1, 
    num_classes=10, 
    class_cond=True
).to(device)

encoder_model = EncoderUNetModel(
    image_size=28,
    in_channels=1,
    model_channels=32,
    out_channels=2*latent_dim,
    num_res_blocks=1,
    attention_resolutions=(28 // 16,),
    channel_mult=(1, 2, 2)).to(device)

decoder_model = UNetModelGuided(
    dim=(1, 28, 28), 
    num_channels=32, 
    num_res_blocks=1, 
    zdim=latent_dim, 
    class_cond=True).to(device)

#optimizer = torch.optim.Adam( list(decoder_model.parameters()))
optimizer = torch.optim.Adam(
    list(encoder_model.parameters()) + list(decoder_model.parameters())
    )

FM = ConditionalFlowMatcher(sigma=sigma)
# Users can try target FM by changing the above line by
# FM = TargetConditionalFlowMatcher(sigma=sigma)
node = NeuralODE(model, solver="dopri5", sensitivity="adjoint", atol=1e-4, rtol=1e-4)

# %%
wandb.init(

)

for epoch in range(n_epochs):
    for i, data in enumerate(train_loader):
        optimizer.zero_grad()
        x1 = data[0].to(device)
        y = data[1].to(device)
        x0 = torch.randn_like(x1)
        t, xt, ut = FM.sample_location_and_conditional_flow(x0, x1)
        #z = torch.randn((xt.shape[0],latent_dim),device=xt.device,dtype=xt.dtype)
        
        latent_dist = encoder_model(x1,t)
        z_mu, z_logstd = latent_dist[:,:latent_dim], latent_dist[:,latent_dim:]
        m = Normal(loc=z_mu,scale=z_logstd.exp())
        z = m.rsample()
        
        # KL divergence
        prior = Normal(loc=torch.zeros_like(z_mu), scale=torch.ones_like(z_logstd))
        kl_div = torch.distributions.kl.kl_divergence(m, prior).mean()
        vt = decoder_model(t, xt, z)
        cfmloss = torch.mean((vt - ut) ** 2)
        
        loss = cfmloss + kl_div
        loss.backward()
        
        torch.nn.utils.clip_grad_norm_(decoder_model.parameters(), max_norm=1.0)
        torch.nn.utils.clip_grad_norm_(encoder_model.parameters(), max_norm=1.0)
        optimizer.step()
        #print(f"epoch: {epoch}, steps: {i}, loss: {loss.item():.4}", end="\r")
        wandb.log(
            {
                "epoch":epoch,
                "steps":i,
                "loss":loss.item(),
                "kl":kl_div,
                "cfmloss":cfmloss
            }
        )

# %%
USE_TORCH_DIFFEQ = True
generated_class_list = torch.arange(10, device=device).repeat(10)
with torch.no_grad():
    if USE_TORCH_DIFFEQ:
        traj = torchdiffeq.odeint(
            lambda t, x: model.forward(t, x, generated_class_list),
            torch.randn(100, 1, 28, 28, device=device),
            torch.linspace(0, 1, 2, device=device),
            atol=1e-4,
            rtol=1e-4,
            method="dopri5",
        )
    else:
        traj = node.trajectory(
            torch.randn(100, 1, 28, 28, device=device),
            t_span=torch.linspace(0, 1, 2, device=device),
        )
grid = make_grid(
    traj[-1, :100].view([-1, 1, 28, 28]).clip(-1, 1), value_range=(-1, 1), padding=0, nrow=10
)
img = ToPILImage()(grid)
plt.imshow(img)
plt.show()

