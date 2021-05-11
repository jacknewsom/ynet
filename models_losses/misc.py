import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from ynet.models_losses.lovasz_utils import StableBCELoss, lovasz_hinge, lovasz_softmax_flat
from torch_scatter import scatter_mean, scatter_add

# Collection of Miscellaneous Loss Functions not yet implemented in Pytorch.

def logit_fn(input, eps=1e-6):
    x = torch.clamp(input, min=eps, max=1-eps)
    return torch.log(x / (1 - x))


def unique_label_torch(label):
    _, label2, cts = torch.unique(label, return_inverse=True, return_counts=True)
    return label2, cts


def iou_batch(pred: torch.BoolTensor, labels: torch.BoolTensor, eps=0.0):
    '''
    pred: N x C
    labels: N x C (one-hot)
    '''
    intersection = (pred & labels).float().sum(0)
    union = (pred | labels).float().sum(0)
    iou = (intersection + eps) / (union + eps)  # We smooth our devision to avoid 0/0

    return iou.mean()


class LovaszHingeLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction='none'):
        super(LovaszHingeLoss, self).__init__(reduction=reduction)

    def forward(self, logits, targets):
        num_clusters = targets.shape[1]
        return lovasz_hinge(logits.T.view(num_clusters, 1, -1),
                            targets.T.view(num_clusters, 1, -1))

class LovaszSoftmaxWithLogitsLoss(torch.nn.modules.loss._Loss):

    def __init__(self, reduction='none'):
        super(LovaszHingeLoss, self).__init__(reduction=reduction)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, logits, targets):
        probs = self.softmax(logits)
        return lovasz_softmax_flat(probs, targets)


def find_cluster_means(features, labels):
    '''
    For a given image, compute the centroids mu_c for each
    cluster label in the embedding space.
    Inputs:
        features (torch.Tensor): the pixel embeddings, shape=(N, d) where
        N is the number of pixels and d is the embedding space dimension.
        labels (torch.Tensor): ground-truth group labels, shape=(N, )
    Returns:
        cluster_means (torch.Tensor): (n_c, d) tensor where n_c is the number of
        distinct instances. Each row is a (1,d) vector corresponding to
        the coordinates of the i-th centroid.
    '''
    device = features.device
    bincount = torch.bincount(labels)
    zero_bins = bincount > 0
    bincount[bincount == 0] = 1.0
    numerator = torch.zeros(bincount.shape[0], features.shape[1]).to(device)
    numerator = numerator.index_add(0, labels, features)
    centroids = numerator / bincount.view(-1, 1)
    centroids = centroids[zero_bins]
    return centroids


def intra_cluster_loss(features, cluster_means, labels, margin=1.0):
    x = features[:, None, :]
    mu = cluster_means[None, :, :]
    l = torch.clamp(torch.norm(x - mu, dim=-1) - margin, min=0)**2
    l = torch.gather(l, 1, labels.view(-1, 1)).squeeze()
    intra_loss = torch.mean(scatter_mean(l, labels))
    return intra_loss


def inter_cluster_loss(cluster_means, margin=0.2):
    inter_loss = 0.0
    n_clusters = len(cluster_means)
    if n_clusters < 2:
        # Inter-cluster loss is zero if there only one instance exists for
        # a semantic label.
        return 0.0
    else:
        indices = torch.triu_indices(cluster_means.shape[0], cluster_means.shape[0], 1)
        dist = distances(cluster_means, cluster_means)
        return torch.pow(torch.clamp(2.0 * margin - dist[indices[0, :], \
            indices[1, :]], min=0), 2).mean()

def regularization_loss(cluster_means):
    return torch.mean(torch.norm(cluster_means, dim=1))


def margin_smoothing_loss(sigma, sigma_means, labels, margin=0):
    x = sigma[:, None]
    mu = sigma_means[None, :]
    l = torch.sqrt(torch.clamp(torch.abs(x-mu) - margin, min=0)**2 + 1e-6)
    l = torch.gather(l, 1, labels.view(-1, 1)).squeeze()
    loss = torch.mean(scatter_mean(l, labels))
    return loss


def get_probs(embeddings, margins, labels, eps=1e-6):

    device = embeddings.device
    n = labels.shape[0]
    centroids = find_cluster_means(embeddings, labels)
    sigma = scatter_mean(margins.squeeze(), labels)
    num_clusters = labels.unique().shape[0]

    # Compute spatial term
    em = embeddings[:, None, :]
    centroids = centroids[None, :, :]
    sqdists = ((em - centroids)**2).sum(-1)

    p = sqdists / (2.0 * sigma.view(1, -1)**2)
    p = torch.clamp(torch.exp(-p), min=eps, max=1-eps)
    logits = logit_fn(p, eps=eps)
    eye = torch.eye(len(labels.unique()), dtype=torch.float32, device=device)
    targets = eye[labels]
    loss_tensor = nn.BCEWithLogitsLoss(reduction='none')(logits, targets)
    loss = loss_tensor.mean(dim=0).mean()
    with torch.no_grad():
        acc = iou_batch(logits > 0, targets.bool())
    smoothing_loss = margin_smoothing_loss(margins.squeeze(), sigma.detach(), labels, margin=0)
    p = torch.gather(p, 1, labels.view(-1, 1))
    return loss, smoothing_loss, p.squeeze(), acc


def multivariate_kernel(centroid, log_sigma, Lprime, eps=1e-8):
    def f(x):
        N = x.shape[0]
        L = torch.zeros(3, 3)
        tril_indices = torch.tril_indices(row=3, col=3, offset=-1)
        L[tril_indices[0], tril_indices[1]] = Lprime
        sigma = torch.exp(log_sigma) + eps
        L += torch.diag(sigma)
        cov = torch.matmul(L, L.T)
        dist = torch.matmul((x - centroid), torch.inverse(cov))
        dist = torch.bmm(dist.view(N, 1, -1), (x-centroid).view(N, -1, 1)).squeeze()
        probs = torch.exp(-dist)
        return probs
    return f


def distances(v1, v2, eps=1e-6):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.sqrt(torch.clamp(torch.pow(v2_2 - v1_2, 2).sum(2), min=eps))


def squared_distances(v1, v2):
    v1_2 = v1.unsqueeze(1).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    v2_2 = v2.unsqueeze(0).expand(v1.size(0), v2.size(0), v1.size(1)).double()
    return torch.pow(v2_2 - v1_2, 2).sum(2)


def bhattacharyya_distance_matrix(v1, v2, eps=1e-8):
    x1, s1 = v1[:, :3], v1[:, 3].view(-1)
    x2, s2 = v2[:, :3], v1[:, 3].view(-1)
    g1 = torch.ger(s1**2, 1.0 / (s2**2 + eps))
    g2 = g1.t()
    dist = squared_distances(x1.contiguous(), x2.contiguous())
    denom = 1.0 / (eps + s1.unsqueeze(1)**2 + s2**2)
    out = 0.25 * torch.log(0.25 * (g1 + g2 + 2)) + 0.25 * dist / denom
    return out


def bhattacharyya_coeff_matrix(v1, v2, eps=1e-6):
    x1, s1 = v1[:, :3], v1[:, 3].view(-1)
    x2, s2 = v2[:, :3], v1[:, 3].view(-1)
    g1 = torch.ger(s1**2, 1.0 / (s2**2 + eps))
    g2 = g1.t()
    dist = squared_distances(x1.contiguous(), x2.contiguous())
    denom = 1.0 / (eps + s1.unsqueeze(1)**2 + s2**2)
    out = 0.25 * torch.log(0.25 * (g1 + g2 + 2)) + 0.25 * dist / denom
    out = torch.exp(-out)
    return out


class FocalLoss(nn.Module):
    '''
    Original Paper: https://arxiv.org/abs/1708.02002
    Implementation: https://www.kaggle.com/c/tgs-salt-identification-challenge/discussion/65938
    '''
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce
        self.stable_bce = StableBCELoss()

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = self.stable_bce(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class WeightedFocalLoss(FocalLoss):

    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(WeightedFocalLoss, self).__init__(alpha=alpha, gamma=gamma, logits=logits, reduce=reduce)

    def forward(self, inputs, targets):
        with torch.no_grad():
            pos_weight = torch.sum(targets == 0) / (1.0 + torch.sum(targets == 1))
            weight = torch.ones(inputs.shape[0]).cuda()
            weight[targets == 1] = pos_weight
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        else:
            BCE_loss = self.stable_bce(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
        F_loss = torch.mul(F_loss, weight)

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss

def gaussian_kernel_torch(centroid, sigma, eps=1e-8):
    def f(x):
        dists = torch.sum(torch.pow(x - centroid, 2), dim=1)
        probs = torch.exp(-dists / (2.0 * (sigma)**2 + eps))
        return probs
    return f

def gaussian_kernel_np(centroid, sigma, eps=1e-8):
    def f(x):
        dists = np.sum(np.power(x - centroid, 2), axis=1)
        probs = np.exp(-dists / (2.0 * (sigma)**2 + eps))
        return probs
    return f

def fit_predict_np(embeddings, seediness, margins, fitfunc,
                 s_threshold=0.0, p_threshold=0.5):
    pred_labels = -np.ones(embeddings.shape[0])
    probs = []
    spheres = []
    seediness_copy = seediness.copy()
    count = 0
    #if seediness_copy.shape[0] == 1:
    #    return np.argmax(seediness_copy)
    while count < int(seediness.shape[0]):
        i = np.argsort(seediness_copy.squeeze())[-1]
        seedScore = seediness[i]
        if seedScore < s_threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        spheres.append((centroid, sigma))
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.reshape(-1, 1))
        cluster_index = (pValues > p_threshold).reshape(-1) & (seediness_copy > 0).reshape(-1)
        seediness_copy[cluster_index] = -1
        count += np.sum(cluster_index)
    if len(probs) == 0:
        return pred_labels, 1
    probs = np.hstack(probs)
    pred_labels = np.argmax(probs, axis=1)
    return pred_labels, probs.shape[1]

def fit_predict_torch(embeddings, seediness, margins, fitfunc,
                 s_threshold=0.0, p_threshold=0.5):
    pred_labels = -torch.ones(embeddings.shape[0])
    probs = []
    spheres = []
    seediness_copy = seediness.detach().clone()
    count = 0
    #if seediness_copy.shape[0] == 1:
    #    return np.argmax(seediness_copy)
    while count < int(seediness.shape[0]):
        i = torch.argsort(seediness_copy.squeeze())[-1]
        seedScore = seediness[i]
        if seedScore < s_threshold:
            break
        centroid = embeddings[i]
        sigma = margins[i]
        spheres.append((centroid, sigma))
        f = fitfunc(centroid, sigma)
        pValues = f(embeddings)
        probs.append(pValues.reshape(-1, 1))
        cluster_index = (pValues > p_threshold).reshape(-1) & (seediness_copy > 0).reshape(-1)
        seediness_copy[cluster_index] = -1
        count += torch.sum(cluster_index)
    if len(probs) == 0:
        return pred_labels, 1
    probs = torch.hstack(probs)
    pred_labels = torch.argmax(probs, dim=1)
    return pred_labels, probs.shape[1]