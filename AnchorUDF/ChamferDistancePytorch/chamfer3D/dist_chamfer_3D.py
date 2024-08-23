import torch
from torch import nn
from torch.autograd import Function

# Chamfer's distance module
class Chamfer3DFunction(Function):
    @staticmethod
    def forward(ctx, xyz1, xyz2):
        batch_size, n, dim = xyz1.size()
        assert dim == 3, "Input points should have 3 dimensions."
        _, m, _ = xyz2.size()

        dist1 = torch.cdist(xyz1, xyz2).min(dim=2)
        dist2 = torch.cdist(xyz2, xyz1).min(dim=2)

        idx1 = dist1.indices
        idx2 = dist2.indices

        dist1 = dist1.values
        dist2 = dist2.values

        ctx.save_for_backward(xyz1, xyz2, idx1, idx2)
        return dist1, dist2, idx1, idx2

    @staticmethod
    def backward(ctx, graddist1, graddist2, gradidx1, gradidx2):
        xyz1, xyz2, idx1, idx2 = ctx.saved_tensors

        gradxyz1 = torch.zeros_like(xyz1)
        gradxyz2 = torch.zeros_like(xyz2)

        for i in range(xyz1.size(0)):
            gradxyz1[i] = 2 * (xyz1[i] - xyz2[i, idx1[i]])
            gradxyz2[i] = 2 * (xyz2[i] - xyz1[i, idx2[i]])

        return gradxyz1, gradxyz2

class chamfer_3DDist(nn.Module):
    def __init__(self):
        super(chamfer_3DDist, self).__init__()

    def forward(self, input1, input2):
        input1 = input1.contiguous()
        input2 = input2.contiguous()
        return Chamfer3DFunction.apply(input1, input2)
