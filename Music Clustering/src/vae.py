import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split


def beta_schedule(epoch, warmup=10, beta_max=4.0):
    return beta_max * min(1.0, epoch / float(warmup))


class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, latent_dim=10):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.mu = nn.Linear(hidden_dim, latent_dim)
        self.logvar = nn.Linear(hidden_dim, latent_dim)
        self.dec = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, input_dim),
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu = self.mu(h)
        logvar = self.logvar(h)
        z = self.reparameterize(mu, logvar)
        x_hat = self.dec(z)
        return x_hat, mu, logvar


def vae_loss(x, x_hat, mu, logvar, beta=1.0):
    recon = ((x - x_hat) ** 2).sum(dim=1).mean()
    kld = (-0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean())
    return recon + beta * kld, recon.detach(), kld.detach()


def train_vae(
    Xz,
    hidden_dim=256,
    latent_dim=10,
    batch_size=512,
    epochs=35,
    lr=3e-3,
    weight_decay=1e-4,
    seed=42,
    beta_max=4.0,
    warmup=10,
    device=None,
):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(seed)
    np.random.seed(seed)

    X_train, X_val = train_test_split(Xz, test_size=0.2, random_state=seed)
    train_loader = DataLoader(
        TensorDataset(torch.tensor(X_train, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=True,
    )
    val_loader = DataLoader(
        TensorDataset(torch.tensor(X_val, dtype=torch.float32)),
        batch_size=batch_size,
        shuffle=False,
    )

    vae = VAE(input_dim=Xz.shape[1], hidden_dim=hidden_dim, latent_dim=latent_dim).to(
        device
    )
    opt = torch.optim.AdamW(vae.parameters(), lr=lr, weight_decay=weight_decay)

    history = []
    best_val = float("inf")
    best_state = None

    for epoch in range(1, epochs + 1):
        vae.train()
        tr = tr_rec = tr_kld = 0.0
        ntr = 0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            x_hat, mu, logvar = vae(xb)
            beta = beta_schedule(epoch, warmup=warmup, beta_max=beta_max)
            loss, recon, kld = vae_loss(xb, x_hat, mu, logvar, beta=beta)
            loss.backward()
            nn.utils.clip_grad_norm_(vae.parameters(), 5.0)
            opt.step()
            bs = xb.size(0)
            tr += loss.item() * bs
            tr_rec += recon.item() * bs
            tr_kld += kld.item() * bs
            ntr += bs

        vae.eval()
        va = va_rec = va_kld = 0.0
        nva = 0
        with torch.no_grad():
            for (xb,) in val_loader:
                xb = xb.to(device)
                x_hat, mu, logvar = vae(xb)
                beta = beta_schedule(epoch, warmup=warmup, beta_max=beta_max)
                loss, recon, kld = vae_loss(xb, x_hat, mu, logvar, beta=beta)
                bs = xb.size(0)
                va += loss.item() * bs
                va_rec += recon.item() * bs
                va_kld += kld.item() * bs
                nva += bs

        row = dict(
            epoch=epoch,
            train_loss=tr / ntr,
            val_loss=va / nva,
            train_recon=tr_rec / ntr,
            val_recon=va_rec / nva,
            train_kld=tr_kld / ntr,
            val_kld=va_kld / nva,
        )
        history.append(row)
        if row["val_loss"] < best_val:
            best_val = row["val_loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in vae.state_dict().items()}

    hist = pd.DataFrame(history)
    return vae, best_state, hist


def encode_mu(vae, Xz, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = vae.to(device)
    vae.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(Xz, dtype=torch.float32).to(device)
        _, mu_all, _ = vae(X_tensor)
    return mu_all.detach().cpu().numpy()


class ConvVAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.latent_dim = latent_dim
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc_mu = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_logvar = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 8, 8)
        return self.dec(h)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        xhat = self.decode(z)
        return xhat, mu, logvar


def conv_vae_loss(x, xhat, mu, logvar, beta=1.0):
    recon = F.mse_loss(xhat, x, reduction="mean")
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return recon + beta * kl, recon.detach(), kl.detach()


def train_convvae(model, dl, epochs=15, beta_start=0.0, beta_end=1.0, device=None, lr=1e-3):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []

    for ep in range(1, epochs + 1):
        beta = beta_start + (beta_end - beta_start) * (ep - 1) / max(1, (epochs - 1))
        model.train()
        total = recon_t = kl_t = 0.0
        n = 0
        for batch in dl:
            if batch is None:
                continue
            x, _ = batch
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            xhat, mu, logvar = model(x)
            loss, recon, kl = conv_vae_loss(x, xhat, mu, logvar, beta=beta)
            loss.backward()
            opt.step()
            bs = x.size(0)
            total += loss.item() * bs
            recon_t += recon.item() * bs
            kl_t += kl.item() * bs
            n += bs
        if n == 0:
            continue
        history.append(
            {"epoch": ep, "beta": beta, "loss": total / n, "recon": recon_t / n, "kl": kl_t / n}
        )
    return pd.DataFrame(history)


class ConvAE(nn.Module):
    def __init__(self, latent_dim=32):
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.ReLU(),
        )
        self.fc = nn.Linear(256 * 8 * 8, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, 256 * 8 * 8)
        self.dec = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )

    def encode(self, x):
        h = self.enc(x).view(x.size(0), -1)
        return self.fc(h)

    def decode(self, z):
        h = self.fc_dec(z).view(z.size(0), 256, 8, 8)
        return self.dec(h)

    def forward(self, x):
        z = self.encode(x)
        xhat = self.decode(z)
        return xhat, z


def train_ae(model, dl, epochs=10, device=None, lr=1e-3):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    history = []
    for ep in range(1, epochs + 1):
        model.train()
        tot = 0.0
        n = 0
        for batch in dl:
            if batch is None:
                continue
            x, _ = batch
            x = x.to(device)
            opt.zero_grad(set_to_none=True)
            xhat, _ = model(x)
            loss = F.mse_loss(xhat, x, reduction="mean")
            loss.backward()
            opt.step()
            bs = x.size(0)
            tot += loss.item() * bs
            n += bs
        if n == 0:
            continue
        history.append({"epoch": ep, "loss": tot / n})
    return pd.DataFrame(history)


def extract_latents(model, dl, device=None):
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()
    Z, Y = [], []
    with torch.no_grad():
        for batch in dl:
            if batch is None:
                continue
            x, y = batch
            x = x.to(device)
            if isinstance(model, ConvVAE):
                _, mu, _ = model(x)
                z = mu
            else:
                _, z = model(x)
            Z.append(z.cpu().numpy())
            Y.append(y.numpy())
    return np.concatenate(Z, 0), np.concatenate(Y, 0)
