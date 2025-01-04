import torch
from torch import autograd
import torch.nn.functional as F


def d_vanilla(d_logit_real, d_logit_fake):
    d_loss = torch.mean(F.softplus(-d_logit_real)) + torch.mean(F.softplus(d_logit_fake))
    return d_loss


def g_vanilla(d_logit_fake):
    return torch.mean(F.softplus(-d_logit_fake))

def g_non_saturated(d_logit_fake):
    return torch.mean(-F.softplus(d_logit_fake))


def d_ls(d_logit_real, d_logit_fake):
    d_loss = 0.5 * (d_logit_real - torch.ones_like(d_logit_real))**2 + 0.5 * (d_logit_fake)**2
    return d_loss.mean()


def g_ls(d_logit_fake):
    gen_loss = 0.5 * (d_logit_fake - torch.ones_like(d_logit_fake))**2
    return gen_loss.mean()

def d_wasserstein(d_logit_real, d_logit_fake):
    return torch.mean(d_logit_fake - d_logit_real)


def g_wasserstein(d_logit_fake):
    return -torch.mean(d_logit_fake)


def feature_matching_loss(real_embed, fake_embed):
    fm_loss = torch.mean(torch.abs(torch.mean(fake_embed, 0) - torch.mean(real_embed, 0)))
    return fm_loss


def cal_grad_penalty(ground_truth, history_window, forecast, discriminator, device):
    batch_size, length, feature_size = ground_truth.shape
    alpha = torch.rand(batch_size, 1)
    alpha = alpha.expand(batch_size, ground_truth.nelement() // batch_size).contiguous().view(batch_size, length, feature_size)
    alpha = alpha.to(device)

    ground_truth = ground_truth.to(device)
    interpolates = alpha * ground_truth + ((1 - alpha) * forecast)
    interpolates = interpolates.to(device)
    interpolates = autograd.Variable(interpolates, requires_grad=True)
    with torch.backends.cudnn.flags(enabled=False):
        d_logit_fake = discriminator(history_window, interpolates)
    grads = autograd.grad(outputs=d_logit_fake,
                          inputs=interpolates,
                          grad_outputs=torch.ones(d_logit_fake.size()).to(device),
                          create_graph=True,
                          retain_graph=True,
                          only_inputs=True)[0]
    grads = grads.view(grads.size(0), -1)

    grad_penalty = ((grads.norm(2, dim=1) - 1)**2).mean()
    return grad_penalty