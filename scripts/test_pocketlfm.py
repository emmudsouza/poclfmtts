import torch

from pocketlfm import LFMBackbone, PocketLFM, PocketLFMConfig

torch.manual_seed(0)


def small_cfg(**kw) -> PocketLFMConfig:
    base = dict(d_model=64, n_layers=4, attn_layer_indices=(1, 3), n_heads=4, n_kv_heads=2,
                ff_dim=128, mtp_horizon=3, head_dim=32, head_depth=2,
                text_vocab=100, text_dim=32, text_heads=4)
    base.update(kw)
    return PocketLFMConfig(**base)


def _batch(cfg, lengths):
    b = len(lengths)
    lt_max = max(lt for lt, _ in lengths)
    t_max = max(t for _, t in lengths)
    tt = torch.zeros(b, lt_max, dtype=torch.long)
    lat = torch.zeros(b, t_max, cfg.latent_dim)
    for i, (lt, t) in enumerate(lengths):
        tt[i, :lt] = torch.randint(0, cfg.text_vocab, (lt,))
        lat[i, :t] = torch.randn(t, cfg.latent_dim)
    tl = torch.tensor([lt for lt, _ in lengths])
    ll = torch.tensor([t for _, t in lengths])
    return tt, tl, lat, ll


def test_backbone_param_budget():
    bd = PocketLFM(PocketLFMConfig()).param_breakdown()
    assert 70e6 < bd["backbone"] < 80e6, f"backbone {bd['backbone']/1e6:.1f}M"


def test_backbone_streaming_equivalence():
    cfg = small_cfg()
    bb = LFMBackbone(cfg).eval()
    x = torch.randn(1, 9, cfg.d_model)
    with torch.no_grad():
        full = bb(x)
        state = bb.init_state()
        outs = []
        for chunk in [x[:, :4], x[:, 4:5], x[:, 5:6], x[:, 6:9]]:
            outs.append(bb(chunk, state))
        stream = torch.cat(outs, dim=1)
    assert torch.allclose(full, stream, atol=1e-5), (full - stream).abs().max().item()


def test_attention_sliding_window_bounds_memory():
    cfg = small_cfg(attn_window=8, attn_sink=4)
    bb = LFMBackbone(cfg).eval()
    state = bb.init_state()
    with torch.no_grad():
        for _ in range(40):
            bb(torch.randn(1, 1, cfg.d_model), state)
    cap = cfg.attn_sink + cfg.attn_window
    for i in cfg.attn_layer_indices:
        kv = state["layers"][i]["mix"]
        assert kv["k"].shape[2] == cap and kv["pos"].shape[0] == cap


def test_compute_losses_runs_and_backprops():
    cfg = small_cfg()
    model = PocketLFM(cfg)
    tt, tl, lat, ll = _batch(cfg, [(6, 10), (4, 7), (5, 3)])
    out = model.compute_losses(tt, tl, lat, ll)
    assert set(out) == {"mtp", "eos", "total"}
    assert all(torch.isfinite(v) for v in out.values())
    out["total"].backward()
    assert any(p.grad is not None for p in model.parameters())


def test_prefix_text_conditions_output():
    cfg = small_cfg()
    model = PocketLFM(cfg).eval()
    tt_a = torch.randint(0, cfg.text_vocab, (1, 6))
    tt_b = torch.randint(0, cfg.text_vocab, (1, 6))
    with torch.no_grad():
        a = model.generate_latents(tt_a, max_frames=8, flow_steps=1, temp=0.0, eos_threshold=1e9)
        b = model.generate_latents(tt_b, max_frames=8, flow_steps=1, temp=0.0, eos_threshold=1e9)
    assert (a - b).abs().mean() > 1e-4, "text prefix should change the output"


def test_compute_losses_overfits_single_batch():
    cfg = small_cfg()
    model = PocketLFM(cfg)
    tt, tl, lat, ll = _batch(cfg, [(6, 12), (6, 12)])
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3)
    first = None
    for _ in range(200):
        out = model.compute_losses(tt, tl, lat, ll)
        opt.zero_grad()
        out["total"].backward()
        opt.step()
        if first is None:
            first = float(out["mtp"].detach())
    assert float(out["mtp"].detach()) < 0.7 * first, (first, float(out["mtp"].detach()))


def test_generate_latents_shape_and_sequential_mtp():
    cfg = small_cfg()
    model = PocketLFM(cfg).eval()
    tt = torch.randint(0, cfg.text_vocab, (1, 5))
    out = model.generate_latents(tt, max_frames=10, flow_steps=1, eos_threshold=1e9)
    assert out.shape == (1, 10, cfg.latent_dim)


def test_collate_right_pads():
    from data_ljspeech import collate

    batch = [
        (torch.tensor([1, 2, 3]), torch.randn(5, 32)),
        (torch.tensor([4, 5]), torch.randn(2, 32)),
    ]
    out = collate(batch)
    assert out["text_tokens"].shape == (2, 3)
    assert out["latents"].shape == (2, 5, 32)
    assert out["lat_lens"].tolist() == [5, 2]
