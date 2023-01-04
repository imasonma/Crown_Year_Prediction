import torch


def build_distribution_model(data):
    """build distribution model.

    Args:
        data: the model features
    Returns:
        mu(torch.Tensor): the mean of model feature
        cov(torch.Tensor): the cov matraix of the model feature
    """
    b, c, h, w = data.shape
    data = data.view(b, c * h * w)
    tmp = torch.ones((1, b), device=data.device) @ data
    cov = (data.t() @ data - (tmp.t() @ tmp) / b) / (b - 1)

    mu = data.mean(dim=0)
    return mu, cov
