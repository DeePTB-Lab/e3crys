import torch
from e3nn.o3 import Irreps

from src.model import FinalMLP, MiddleMLP, Model, ReadoutLayer
from src.train_test.train import _create_shared_components


def test_current_high_order_model_forward_smoke():
    torch.manual_seed(0)

    (
        embedding_layer,
        invariant_layers,
        equivariant_layers,
        irreps_list,
        final_irreps_hidden,
        final_irreps_out,
        _,
    ) = _create_shared_components(
        dist_emb_func="gaussian",
        embed_dim=8,
        max_atom_type=10,
        cutoff=5.0,
        inv_update_method="comformer",
        num_inv_layers=1,
        num_equi_layers=1,
        equi_update_method="tpconv_with_edge",
        tp_method="so2",
        scalar_dim=2,
        vec_dim=1,
        num_final_hidden_layers=1,
        final_scalar_hidden_dim=4,
        final_vec_hidden_dim=1,
        final_scalar_out_dim=2,
        final_vec_out_dim=1,
    )

    model = Model(
        embedding_layer=embedding_layer,
        invariant_layers=invariant_layers,
        middle_mlp=MiddleMLP(
            scalar_dim_in=8,
            scalar_dim_hidden=16,
            scalar_dim_out=2,
            num_hidden_layers=1,
        ),
        equivariant_layers=equivariant_layers,
        final_mlp=FinalMLP(
            irreps_in=Irreps(irreps_list[-1]),
            irreps_hidden=final_irreps_hidden,
            irreps_out=final_irreps_out,
            num_hidden_layers=1,
        ),
        readout_layer=ReadoutLayer(
            l_max=0,
            symmetry=None,
            irreps_out=final_irreps_out,
        ),
        final_pooling=True,
    )
    model.eval()

    atom_type = torch.tensor([6, 8, 6, 3, 3, 8], dtype=torch.long)
    edge_index = torch.tensor(
        [
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 0, 2, 1, 4, 3, 5, 4],
        ],
        dtype=torch.long,
    )
    edge_vec = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.5, 0.5, 0.0],
            [-0.5, -0.5, 0.0],
            [0.0, 0.5, 0.5],
            [0.0, -0.5, -0.5],
        ],
        dtype=torch.float32,
    )
    batch_index = torch.tensor([0, 0, 0, 1, 1, 1], dtype=torch.long)

    with torch.no_grad():
        output = model(atom_type, edge_vec, edge_index, batch_index)

    assert output.shape == (2,)
    assert torch.isfinite(output).all()
