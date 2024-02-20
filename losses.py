import torch


def dsm_score_estimation_ref(scorenet, density, beta , samples, labels, sigmas):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    D1 = density(samples)
    ratio2 = torch.tensor((torch.exp(-D1))**(beta/(1+beta))).unsqueeze(-1)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss2 = torch.mean(1 / 2. * ((((scores - target) ** 2 ).sum(dim=-1))*ratio2)* used_sigmas.squeeze())
    return loss2.mean(dim=0)


def dsm_score_estimation_obj(scorenet, density, beta , samples, labels, sigmas):
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:])))
    perturbed_samples = samples + torch.randn_like(samples) * used_sigmas
    target = - 1 / (used_sigmas ** 2) * (perturbed_samples - samples)
    scores = scorenet(perturbed_samples, labels)
    D1 = density(samples)
    ratio1 = torch.tensor((torch.exp(D1))**(1/(1+beta))).unsqueeze(-1)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss1 = torch.mean(1 / 2. * ((((scores - target) ** 2 ).sum(dim=-1))*ratio1)* used_sigmas.squeeze())
    return loss1.mean(dim=0)
