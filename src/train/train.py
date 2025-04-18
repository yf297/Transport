import torch
from torch.optim import Adam

from fields.scalar_field import ScalarField
from fields.vector_field import VectorField
from models.neural_flow import NeuralFlow
from models.gp import GP
from train.optim import mle
from train.scale import NormalizeTime, NormalizeLocation, NormalizeLocationInverse, NormalizeScalar, ScaleFlow


def train_vector_field(
    sf: ScalarField
) -> VectorField:
    nt = NormalizeTime(sf.times)
    nl = NormalizeLocation(sf.locations.reshape(-1, 2))
    nli = NormalizeLocationInverse(sf.locations.reshape(-1, 2))
    ns = NormalizeScalar(sf.scalar.reshape(sf.times.size(0), -1))
    
    T = nt(sf.times)
    H, W, _ = sf.locations.shape
    XY = nl(sf.locations.reshape(-1, 2))
    Z = ns(sf.scalar.reshape(T.size(0), -1))

    flow = NeuralFlow()
    gp = GP(flow)

    mle(T, XY, Z, gp, 50, 2000, 1)

    T = sf.times
    XY = sf.locations.reshape(-1, 2)
    TXY = torch.cat([
        T.repeat_interleave(XY.size(0)).unsqueeze(1),
        XY.repeat(T.size(0), 1)
    ], dim=-1)

    Jacobians = torch.vmap(torch.func.jacrev(ScaleFlow(gp.flow,nt,nl,nli)))(TXY)
    Dt = Jacobians[..., 0:1]
    Dx = Jacobians[..., 1:]
    vector = torch.linalg.solve(Dx, -Dt).squeeze(-1)
    vector = vector.view(T.size(0), H, W, 2)
    
    vf = VectorField(sf.times, sf.locations, vector)
    vf.map = sf.map
    return vf
