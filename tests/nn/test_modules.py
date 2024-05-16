import torch
from af3.nn import common
from af3.nn import modules as M
from af3.nn import diffusion_modules as DM

bshape = (3, 1)
n_msa = 15
seq_len = 31
d_msa = 29
d_pair = 13
n_diff = 7
d_token = 17
d_cond = 19

m = torch.randn(*bshape, n_msa, seq_len, d_msa)
z = torch.randn(*bshape, seq_len, seq_len, d_pair)

a = torch.randn(n_diff, *bshape, seq_len, d_token)
s = torch.randn(*bshape, seq_len, d_cond)

print("alg  3 relpos")
relpos_module = M.RelativePositionEncoding(32, 2)
input_size = (*bshape, 127)     # larger seqlen for better test
features = {
    "asym_id": torch.randint(0, 5, input_size),
    "sym_id": torch.randint(0, 2, input_size),
    "entity_id": torch.randint(0, 3, input_size),
    "token_index": torch.randint(1, 127, input_size),
    "residue_index": torch.randint(1, 127, input_size),
}
rpe = relpos_module(features)
assert rpe.shape == (*input_size, input_size[-1], relpos_module.output_dim), rpe.shape
del relpos_module, rpe, features

print("alg  4 onehot")
x = torch.randint(0, 128, input_size)
x_1h = common.one_hot(x, 128)
assert x_1h.shape == (*x.shape, 128)
x_cont = torch.randn(*input_size)
x_cont_1h = common.one_hot_continuous(
    x, bins=torch.linspace(-2, 2, 100)
)
assert x_cont_1h.shape == (*x_cont.shape, 101), x_cont_1h.shape
del x, x_1h, x_cont, x_cont_1h

print("alg  9 outer prod mean")
opm = M.OuterProductMean(d_msa, d_pair, d_hid=7)
_z = opm(m)
assert z.shape == _z.shape, _z.shape
_z.sum().backward()
del opm, _z

print("alg 10 pair avg weighting ...")
paw_layer = M.PairAverageWeighting(d_msa, d_pair, d_hid=7, num_heads=3)
_m = paw_layer(m, z)
assert _m.shape == m.shape, _m.shape
_m.sum().backward()
del paw_layer, _m

print("alg 11 transition ...")
transition = M.Transition(d_msa, 4)
_m = transition(m)
assert _m.shape == m.shape, _m.shape
_m.sum().backward()
del transition, _m


print("alg 12 & 13 triangle updates")
triangle_update_out = M.TriangleMultiplicationOutgoing(d_pair, d_hid=11)
triangle_update_in = M.TriangleMultiplicationIncoming(d_pair, d_hid=11)
_z = triangle_update_out(z)
assert _z.shape == z.shape, _z.shape
_z.sum().backward()
_z = triangle_update_in(z)
assert _z.shape == z.shape, _z.shape
_z.sum().backward()
del triangle_update_out, triangle_update_in, _z

print("alg 14 & 15 triangle attention")
triangle_att_start = M.TriangleAttentionStarting(d_pair, d_hid=11, num_heads=3)
triangle_att_end = M.TriangleAttentionEnding(d_pair, d_hid=11, num_heads=3)
_z = triangle_att_start(z)
assert _z.shape == z.shape, _z.shape
_z.sum().backward()
_z = triangle_att_end(z)
assert _z.shape == z.shape, _z.shape
_z.sum().backward()
del triangle_att_start, triangle_att_end, _z

print("alg 22 fourier emb")
fourier_emb = DM.FourierEmbedding(257)
t = torch.rand(*bshape, 47)     # simulate diffusion batch size
sigma_data = torch.rand(*t.shape)
t_emb = fourier_emb(t, sigma_data)
assert t_emb.shape == (*t.shape, 257), t_emb
t_emb.sum().backward()
del fourier_emb, t_emb

print("alg 23 diffusion tfm")
diff_tfm = DM.DiffusionTransformer(3, d_token, d_pair, d_cond, num_heads=3, n_transition=1)
beta = torch.rand(*bshape, seq_len, seq_len)
_a = diff_tfm(
    a, s, z, beta
)
assert _a.shape == a.shape, _a.shape
_a.sum().backward()
del diff_tfm, _a

print("alg 24 att pair bias")
att_pair_bias = DM.AttentionPairBias(d_token, d_pair, d_cond, num_heads=3)
_a = att_pair_bias(a, s, z, beta)
assert _a.shape == a.shape, _a.shape
_a.sum().backward()
del att_pair_bias, _a

print("alg 25 cond trans blk")
cond_trans_blk = DM.ConditionalTransitionBlock(d_token, d_cond)
_a = cond_trans_blk(a, s[None])
assert _a.shape == a.shape, _a.shape
_a.sum().backward()
del cond_trans_blk, _a

print("alg 26 adaln")
adaln = DM.AdaptiveLayerNorm(d_token, d_cond)
_a = adaln(a, s[None])
assert _a.shape == a.shape, _a.shape
_a.sum().backward()
del adaln, _a