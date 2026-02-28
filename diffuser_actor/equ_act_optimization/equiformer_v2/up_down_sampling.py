import einops
import torch
from torch_cluster import knn

from diffuser_actor.equ_act_optimization.equiformer_v2.connectivity import KnnGraph


class CoefficientMaxPool(torch.nn.Module):
    """
    Fourier coefficient degree(l)-channel wise maxpool
    """

    def __init__(self, lmax):
        super().__init__()
        self.lmax = lmax
        self.register_buffer('i2l', torch.zeros(((self.lmax + 1) ** 2, self.lmax + 1)))
        count = 0
        for l in range(self.lmax + 1):
            self.i2l[count:count + 2 * l + 1, l] = 1
            count += 2 * l + 1

    def forward(self, x):
        """
        x in shape [batch n_neighbor irrep channel]
        """
        assert x.dim() == 4
        dot = x * x
        ll2 = torch.einsum('bnic, il -> bnlc', dot, self.i2l)
        idx = torch.argmax(ll2, dim=1, keepdim=True)
        idx = torch.einsum('belc, il-> beic', idx.float(), self.i2l).long()
        out = torch.gather(x, 1, idx).squeeze(1)
        return out


def interpolation(xyz, new_xyz, feat, batch_src, batch_dst, k=3):
    """
    trilinear interpolation
    input: xyz: (m, 3), new_xyz: (n, 3), feat: (m, i, c), offset: (b), new_offset: (b)
    output: (n, c)
    """
    assert xyz.is_contiguous() and new_xyz.is_contiguous() and feat.is_contiguous()
    edge_dst, edge_src = knn(xyz, new_xyz, k, batch_x=batch_src, batch_y=batch_dst)
    dist = (xyz[edge_src] - new_xyz[edge_dst]).norm(p=1, dim=-1)  # (n x k)
    dist_recip = 1.0 / (dist + 1e-8)
    dist_recip = dist_recip.reshape(-1, 3)
    norm = dist_recip.sum(-1, keepdim=True)
    weight = dist_recip / norm
    new_feat = feat[edge_src] * weight.reshape(-1, 1, 1)
    new_feat = einops.rearrange(new_feat, '(b k) i c -> b k i c', k=k)
    new_feat = new_feat.sum(1)
    return new_feat


if __name__ == '__main__':
    device = 'cuda:0'
    A = torch.rand(2, 2, 4, 2).to(device)
    print(A)
    pool = CoefficientMaxPool(lmax=1).to(device)
    B = pool(A)
    print(B)

    ori = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0],
                        [1, 0, 0], [1, 0, 1], [1, 1, 0]]).to(device).float()
    bo = torch.arange(2).repeat_interleave(3).to(device)
    tar = torch.tensor([[0, 0, 0], [0, 0, 1], [0, 1, 0], [0, 0.5, 0.5],
                        [1, 0, 0], [1, 0, 1], [1, 1, 0], [1, 0.5, 0.5]]).to(device).float()
    bt = torch.arange(2).repeat_interleave(4).to(device)
    emb = torch.tensor([[0, 0], [1, 2], [5, 10], [0, 0], [3, 6], [15, 30]]).to(device).float()
    up = interpolation(ori, tar, emb, bo, bt)
    print(up)
