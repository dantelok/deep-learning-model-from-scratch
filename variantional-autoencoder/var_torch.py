import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np


class VAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc21 = nn.Linear(hidden_dim, latent_dim)  # Mean
        self.fc22 = nn.Linear(hidden_dim, latent_dim)  # Log-variance
        self.fc3 = nn.Linear(latent_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        x = x.view(-1, 28 * 28)
        h1 = torch.relu(self.fc1(x))
        return self.fc21(h1), self.fc22(h1)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h3 = torch.relu(self.fc3(z))
        return torch.sigmoid(self.fc4(h3))

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar


def vae_loss(recon_x, x, mu, logvar):
    x = x.view(-1, 28 * 28).to(recon_x.dtype)
    BCE = F.binary_cross_entropy(recon_x, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return BCE + KLD


# Datasets
transform = transforms.Compose([
    transforms.ToTensor(),  # Converts to [0,1]
])
mnist = torchvision.datasets.MNIST(root='./data', train=True, download=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=100, shuffle=True)

# Training Loop
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VAE().to(device)
# optimizer = torch.optim.SGD(params=model.parameters(), momentum=0.9, lr=1e-3)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

epochs = 50  # Original 500

for epoch in range(epochs):
    model.train()
    total_loss = 0
    for batch_idx, (data, _) in enumerate(dataloader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = vae_loss(recon_batch, data, mu, logvar)
        loss.backward()
        total_loss += loss.item()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {total_loss / len(mnist):.4f}")

# Sampling
model.eval()
with torch.no_grad():
    z = torch.randn(64, 20).to(device)
    sample = model.decode(z).cpu().view(-1, 1, 28, 28)
    grid = torchvision.utils.make_grid(sample, nrow=8)
    plt.imshow(grid.permute(1, 2, 0).squeeze())
    plt.axis('off')
    plt.show()


# Evaluation: Marginal Likelihood (Importance Sampling)
def estimate_log_likelihood(model, data, k=100):
    model.eval()
    with torch.no_grad():
        x = data.view(-1, 784).to(device)
        mu, logvar = model.encode(x)
        std = torch.exp(0.5 * logvar)
        log_qz_x = []
        log_px_z = []
        log_pz = []

        for _ in range(k):
            eps = torch.randn_like(std)
            z = mu + eps * std
            recon_x = model.decode(z)
            log_px = -nn.functional.binary_cross_entropy(recon_x, x, reduction='none').sum(dim=1)
            log_pz_k = -0.5 * torch.sum(z ** 2 + np.log(2 * np.pi), dim=1)
            log_qz_x_k = -0.5 * torch.sum(((z - mu) / std) ** 2 + logvar + np.log(2 * np.pi), dim=1)
            log_px_z.append(log_px)
            log_pz.append(log_pz_k)
            log_qz_x.append(log_qz_x_k)

        log_px_z = torch.stack(log_px_z, dim=1)
        log_pz = torch.stack(log_pz, dim=1)
        log_qz_x = torch.stack(log_qz_x, dim=1)

        log_w = log_px_z + log_pz - log_qz_x
        log_mean_w = torch.logsumexp(log_w, dim=1) - np.log(k)
        return log_mean_w.mean().item()


# Test on small batch for quick estimate
batch = next(iter(dataloader))[0]
print("Estimated log-likelihood:", estimate_log_likelihood(model, batch[:32]))

# Latent Interpolation
with torch.no_grad():
    data_iter = iter(dataloader)
    imgs, _ = next(data_iter)
    img1, img2 = imgs[0].view(-1, 784).to(device), imgs[1].view(-1, 784).to(device)
    mu1, _ = model.encode(img1)
    mu2, _ = model.encode(img2)

    steps = 10
    interpolations = []
    for alpha in torch.linspace(0, 1, steps):
        z = mu1 * (1 - alpha) + mu2 * alpha
        recon = model.decode(z).view(1, 28, 28).cpu()
        interpolations.append(recon)

    interp_grid = torch.cat(interpolations, dim=2)
    plt.imshow(interp_grid.squeeze().numpy(), cmap='gray')
    plt.axis('off')
    plt.title('Latent Space Interpolation')
    plt.show()


