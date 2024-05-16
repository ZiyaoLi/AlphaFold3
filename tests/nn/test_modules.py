import torch
from af3.nn import modules as M
from af3.nn import diffusion_modules as DM

bshape = (3, 1)
n_msa = 15
seq_len = 31
d_msa = 29
d_pair = 13

m = torch.randn(*bshape, n_msa, seq_len, d_msa)
z = torch.randn(*bshape, seq_len, seq_len, d_pair)

print("alg10 pair avg weighting ...")
paw_layer = M.PairAverageWeighting(d_msa, d_pair, d_hid=7, num_heads=3)
_m = paw_layer(m, z)
assert _m.shape == m.shape, _m.shape
_m.sum().backward()
del paw_layer, _m

print("alg11 transition ...")
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
t_emb = fourier_emb(t)
assert t_emb.shape == (*t.shape, 257), t_emb
t_emb.sum().backward()
del fourier_emb, t_emb

