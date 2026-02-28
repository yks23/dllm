"""
Info-Gain Sampler for Dream.

Dream-specific: left-padded canvas, right-shifted logits, 3-arg forward.
Cache modes: ``None`` (baseline) · ``"prefix"`` · ``"dual"``.
"""

from dataclasses import dataclass

import torch

from dllm.core.samplers.base import BaseSampler, BaseSamplerConfig, BaseSamplerOutput
from dllm.core.samplers.utils import get_num_transfer_tokens
from dllm.pipelines.dream.models.generation_utils import top_k_logits, top_p_logits
from dllm.pipelines.info_gain.core import (
    compute_entropy,
    expand_kv,
    pad_logits,
    score_candidates,
)


def _sample(logits, temperature=0.0, top_p=None, top_k=None):
    if temperature > 0:
        logits = logits / temperature
    if top_p is not None and top_p < 1:
        logits = top_p_logits(logits, top_p)
    if top_k is not None:
        logits = top_k_logits(logits, top_k)
    return torch.softmax(logits, -1).max(-1)  # (confidence, x0)


# ── Info-Gain select (Dream forward convention) ─────────────────────────────


def _info_gain_select(
    model,
    x,
    logits,
    mask_index,
    k,
    n_cand,
    pos_temp,
    mask_token_id,
    *,
    block_start=0,
    block_end=0,
    block_size=0,
    past_key_values=None,
    dual_cache=False,
    replace_position=None,
    attention_mask=None,
    tok_idx=None,
    right_shift_logits=True,
    temperature=0.0,
    top_p=None,
    top_k=None,
    variant="info_gain",
):
    device, T = x.device, x.shape[1]
    # Use dtype-appropriate minimum value to avoid overflow when converting to bfloat16
    neg = torch.finfo(logits.dtype).min

    # Block-restricted mask & base sample
    vbm = mask_index.clone()
    vbm[:, :block_start] = False
    vbm[:, block_end:] = False
    valid = torch.where(vbm[0])[0]
    nv = valid.shape[0]
    if nv == 0:
        return x.clone(), 0.0, None

    cb, x0b = _sample(logits[mask_index], temperature, top_p, top_k)
    fc = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
    fc[mask_index] = cb
    fc[:, :block_start] = neg
    fc[:, block_end:] = neg
    x0c = torch.full_like(x, mask_token_id)
    x0c[mask_index] = x0b.clone()

    def _make(sel):
        xo = x.clone()
        xo[0, sel] = x0c[0, sel]
        return xo, compute_entropy(logits)[0, sel].sum().item(), None

    if nv <= k:
        return _make(valid)
    if pos_temp <= 0 or n_cand <= 1:
        _, ti = torch.topk(fc[0], k)
        return _make(ti)

    # Diverse candidates (position sampling; token diversity via confidence perturbation)
    actions, x0cs, seen = [], [], set()
    for c in range(n_cand):
        if c == 0:
            fc_c, x0c_c = fc, x0c
        else:
            cf, x0f = _sample(logits[mask_index].clone(), temperature, top_p, top_k)
            if temperature > 0:
                cf = cf + 0.1 * (
                    -torch.log(-torch.log(torch.rand_like(cf) + 1e-10) + 1e-10)
                )
            fc_c = torch.full(mask_index.shape, neg, device=device, dtype=logits.dtype)
            fc_c[mask_index] = cf
            fc_c[:, :block_start] = neg
            fc_c[:, block_end:] = neg
            x0c_c = torch.full_like(x, mask_token_id)
            x0c_c[mask_index] = x0f.clone()
        vc = fc_c[0, valid]
        if c == 0:
            _, tk = torch.topk(vc, min(k, nv))
        else:
            g = -torch.log(-torch.log(torch.rand(nv, device=device) + 1e-10) + 1e-10)
            _, tk = torch.topk(vc / pos_temp + g, min(k, nv))
        act = valid[tk]
        key = tuple(sorted(act.tolist()))
        if key not in seen:
            seen.add(key)
            actions.append(act)
            x0cs.append(x0c_c)

    if len(actions) <= 1:
        return _make(actions[0])

    # Batch next-states & lookahead
    nc = len(actions)
    xb = x.expand(nc, -1).clone()
    for i in range(nc):
        xb[i, actions[i]] = x0cs[i][0, actions[i]]

    def _shift(lg):
        return torch.cat([lg[:, :1], lg[:, :-1]], dim=1) if right_shift_logits else lg

    def _expand_attn(am, n, seq_len=None):
        if am is None or am == "full":
            return am
        # Handle sequence length mismatch
        if seq_len is not None and am.shape[-1] != seq_len:
            # If sequence length doesn't match, slice or pad attention_mask
            if am.shape[-1] > seq_len:
                am = am[..., :seq_len]
            else:
                # Pad with False (invalid positions)
                pad_len = seq_len - am.shape[-1]
                if am.dim() == 2:
                    am = torch.cat([am, torch.zeros(am.shape[0], pad_len, dtype=am.dtype, device=am.device)], dim=-1)
                elif am.dim() == 4:
                    am = torch.cat([am, torch.zeros(*am.shape[:-1], pad_len, dtype=am.dtype, device=am.device)], dim=-1)
        # Expand batch dimension
        if am.dim() == 2:
            # [B, T] -> [n, T]
            # Dream model expects 2D attention_mask, it will convert to 4D internally
            return am.expand(n, -1)
        elif am.dim() == 4:
            # [B, H, T, T] -> [n, H, T, T]
            return am.expand(n, -1, -1, -1)
        else:
            # Fallback: expand first dimension
            return am.expand(n, *(-1,) * (am.dim() - 1))

    with torch.no_grad():
        if dual_cache and past_key_values is not None:
            ep = expand_kv(past_key_values, nc)
            rp = (
                replace_position.expand(nc, -1)
                if replace_position is not None
                else None
            )
            rtk = (
                tok_idx[:, block_start:block_end].expand(nc, -1)
                if tok_idx is not None
                else None
            )
            xb_slice = xb[:, block_start:block_end]
            am_slice = _expand_attn(attention_mask, nc, xb_slice.shape[1]) if attention_mask is not None else None
            nl = _shift(
                model(
                    xb_slice,
                    am_slice,
                    rtk,
                    past_key_values=ep,
                    use_cache=False,
                    dual_cache=True,
                    replace_position=rp,
                ).logits
            )
            next_logits = pad_logits(nl, T, block_start, device)
        elif past_key_values is not None:
            ep = expand_kv(past_key_values, nc)
            rtk = (
                tok_idx[:, block_start:].expand(nc, -1) if tok_idx is not None else None
            )
            xb_slice = xb[:, block_start:]
            am_slice = _expand_attn(attention_mask, nc, xb_slice.shape[1]) if attention_mask is not None else None
            nl = _shift(
                model(
                    xb_slice,
                    am_slice,
                    rtk,
                    past_key_values=ep,
                    use_cache=False,
                ).logits
            )
            next_logits = nl.new_zeros(nc, T, nl.shape[-1])
            next_logits[:, block_start : block_start + nl.shape[1]] = nl
        else:
            bt = tok_idx.expand(nc, -1) if tok_idx is not None else None
            # For Dream model with batch size > 1, we need to handle attention_mask carefully
            # If attention_mask is 2D and batch size > 1, Dream's scaled_dot_product_attention
            # may need 4D mask. For now, pass None or "full" when batch size > 1
            if attention_mask is not None and nc > 1:
                # When batch size > 1, Dream model may need 4D attention mask
                # Convert 2D to 4D: [B, T] -> [B, 1, T, T]
                if attention_mask.dim() == 2:
                    # Create 4D mask: [nc, 1, T, T]
                    am_2d = _expand_attn(attention_mask, nc, xb.shape[1])
                    # Convert to 4D: [nc, T] -> [nc, 1, T, T]
                    am_4d = torch.logical_and(
                        am_2d.unsqueeze(1).unsqueeze(-2),
                        am_2d.unsqueeze(1).unsqueeze(-1),
                    )
                    am_expanded = am_4d
                else:
                    am_expanded = _expand_attn(attention_mask, nc, xb.shape[1])
            else:
                am_expanded = _expand_attn(attention_mask, nc, xb.shape[1]) if attention_mask is not None else None
            nl = _shift(model(xb, am_expanded, bt).logits)
            next_logits = nl

    _, _, scores = score_candidates(
        logits, next_logits, xb, actions, mask_token_id, device, variant=variant
    )
    best = scores.argmax().item()
    xo = x.clone()
    xo[0, actions[best]] = x0cs[best][0, actions[best]]
    return xo, scores[best].item(), next_logits[best : best + 1]


# ── Config / Sampler ────────────────────────────────────────────────────────


@dataclass
class InfoGainDreamSamplerConfig(BaseSamplerConfig):
    max_new_tokens: int = 20
    max_length: int = None
    steps: int = 512
    eps: float = 1e-3
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 50
    right_shift_logits: bool = True
    use_cache: str | None = None  # None / "prefix" / "dual"
    block_size: int = 32
    threshold: float | None = None
    candidate_number: int = 8
    position_temperature: float = 0.1
    variant: str = "info_gain"  # "info_gain" or "lookum"


@dataclass
class InfoGainDreamSampler(BaseSampler):
    @torch.no_grad()
    def sample(self, inputs, config=None, **kwargs):
        config = config or InfoGainDreamSamplerConfig()
        C = {
            f: kwargs.get(f, getattr(config, f))
            for f in (
                "max_new_tokens",
                "max_length",
                "steps",
                "eps",
                "temperature",
                "top_p",
                "top_k",
                "threshold",
                "use_cache",
                "block_size",
                "return_dict",
                "right_shift_logits",
                "candidate_number",
                "position_temperature",
                "variant",
            )
        }
        uc = C["use_cache"]
        if uc == "none":
            uc = None
        assert uc in (None, "prefix", "dual"), f"bad use_cache={uc!r}"
        mtid = self.tokenizer.mask_token_id
        eos = self.tokenizer.eos_token_id
        pad_token_id = getattr(self.tokenizer, 'pad_token_id', None)

        # ── canvas (left-padded) ────────────────────────────────────────
        if isinstance(inputs[0], list):
            inputs = [
                torch.as_tensor(p, dtype=torch.long, device=self.model.device)
                for p in inputs
            ]
        pls = [p.shape[0] for p in inputs]
        B = len(inputs)
        mnt = C["max_new_tokens"]
        ml = C["max_length"]
        if ml is None:
            ml = mnt + max(pls)
        elif mnt is None:
            mnt = ml - max(pls)
        T = ml

        x = torch.full((B, T), eos, dtype=torch.long, device=self.model.device)
        sls = []
        for i, p in enumerate(inputs):
            tl = pls[i] + mnt
            sls.append(tl)
            st = T - tl
            x[i, st : st + pls[i]] = p
            x[i, st + pls[i] : T] = mtid
        am = torch.zeros(B, T, dtype=torch.bool, device=self.model.device)
        for j, L in enumerate(sls):
            if L > 0:
                am[j, -L:] = True
        pid = None
        if torch.any(~am):  # Check if any False values
            am_long = am.long()
            pid = am_long.cumsum(-1) - 1
            pid.masked_fill_(~am, 1)

        def shift(lg):
            # Suppress EOS and PAD tokens by subtracting 4 from logits
            if eos is not None:
                lg[:, :, eos] -= 4.0
            if pad_token_id is not None:
                lg[:, :, pad_token_id] -= 4.0
            return (
                torch.cat([lg[:, :1], lg[:, :-1]], dim=1)
                if C["right_shift_logits"]
                else lg
            )
        
        def _check_and_fill_eos():
            """Check if EOS token appears, and fill remaining mask positions with EOS if found."""
            if eos is None:
                return
            # Dream uses left-padded canvas, generation region is from gs to T
            gs = T - mnt
            # Check if EOS appears in the generation region
            for i in range(B):
                # Generation region is from gs to T (left-padded)
                gen_region = x[i, gs:]
                eos_positions = (gen_region == eos).nonzero(as_tuple=True)[0]
                if len(eos_positions) > 0:
                    # Found EOS, fill all remaining mask positions with EOS
                    first_eos_pos = gs + eos_positions[0].item()
                    # Fill all mask positions after the first EOS with EOS
                    if first_eos_pos + 1 < T:
                        remaining_mask = (x[i, first_eos_pos + 1:] == mtid)
                        if remaining_mask.any():
                            x[i, first_eos_pos + 1:][remaining_mask] = eos

        ig = dict(
            n_cand=C["candidate_number"],
            pos_temp=C["position_temperature"],
            mask_token_id=mtid,
            right_shift_logits=C["right_shift_logits"],
            temperature=C["temperature"],
            top_p=C["top_p"],
            top_k=C["top_k"],
            variant=C["variant"],
        )

        def _fill(logits, mi):
            ml_ = logits[mi]
            if ml_.numel() == 0:
                return x.clone()
            _, x0f = _sample(ml_, C["temperature"], C["top_p"], C["top_k"])
            xo = x.clone()
            canvas = torch.full_like(x, mtid)
            canvas[mi] = x0f
            xo[mi] = canvas[mi]
            return xo

        # ── no-cache mode ───────────────────────────────────────────────
        if uc is None:
            mi = x == mtid
            ntt = get_num_transfer_tokens(mi, C["steps"], self.scheduler)
            es = ntt.size(1)
            hist = [x.clone()] if C["return_dict"] else None
            cl = None

            for i in range(es):
                mi = x == mtid
                if not mi.any():
                    break
                ki = ntt[0, i].item()
                if ki <= 0:
                    continue
                rem = int(mi.sum().item())
                is_last = ki >= rem
                if cl is not None:
                    logits = cl
                    cl = None
                else:
                    logits = shift(self.model(x, am, pid).logits)
                if is_last:
                    x = _fill(logits, mi)
                    _check_and_fill_eos()
                    hist and hist.append(x.clone())
                    break
                bp = self._bypass(logits, x, mi, mtid, ki, C["threshold"])
                if bp is not None:
                    x = bp
                    _check_and_fill_eos()
                    cl = None
                    hist and hist.append(x.clone())
                    continue
                gs = T - mnt
                x, _, cl = _info_gain_select(
                    self.model,
                    x,
                    logits,
                    mi,
                    ki,
                    block_start=gs,
                    block_end=T,
                    block_size=mnt,
                    attention_mask=am,
                    tok_idx=pid,
                    **ig,
                )
                _check_and_fill_eos()
                hist and hist.append(x.clone())
            return (
                x
                if not C["return_dict"]
                else BaseSamplerOutput(sequences=x, histories=hist)
            )

        # ── cache modes (prefix / dual) ─────────────────────────────────
        bs = C["block_size"] or mnt
        assert mnt % bs == 0
        nb = mnt // bs
        assert C["steps"] % nb == 0
        spb = C["steps"] // nb
        is_dual = uc == "dual"
        eps_val = C["eps"]

        if torch.any(~am):  # Check if any False values
            cam = torch.logical_and(
                am.unsqueeze(1).unsqueeze(-2),
                am.unsqueeze(1).unsqueeze(-1),
            )
            tok_idx = pid
        else:
            cam = "full"
            tok_idx = None

        hist = [x.clone()] if C["return_dict"] else None
        gs = T - mnt
        pkv = None

        for nb_i in range(nb):
            cbs, cbe = gs + nb_i * bs, gs + (nb_i + 1) * bs

            # Block entry: full forward → cache update
            mo = self.model(x, cam, tok_idx, use_cache=True)
            pkv = mo.past_key_values
            logits = shift(mo.logits)
            _, x0f = _sample(logits, C["temperature"], C["top_p"], C["top_k"])
            x[:, cbs] = x0f[:, cbs]
            _check_and_fill_eos()
            hist and hist.append(x.clone())

            rp = None
            if not is_dual:
                pkv = [
                    tuple(pkv[li][kj][:, :cbs, :] for kj in range(len(pkv[li])))
                    for li in range(len(pkv))
                ]
            else:
                rp = torch.zeros_like(x, dtype=torch.bool)
                rp[:, cbs:cbe] = True

            ts = torch.linspace(1, eps_val, spb + 1, device=x.device)
            ins = 1
            crl = None

            while True:
                region = x[:, cbs:cbe] if is_dual else x[:, cbs:]
                mir = region == mtid
                mir[:, bs:] = False
                if not mir.any():
                    break
                nmt = mir.sum() / mir.shape[0]
                ks = (
                    (
                        int(nmt * (1 - ts[ins + 1] / ts[ins]))
                        if ins < spb - 1
                        else int(nmt)
                    )
                    if ins < spb
                    else int(mir.sum().item())
                )
                if ks <= 0:
                    ins += 1
                    continue
                rem = int((x[:, cbs:cbe] == mtid).sum().item())
                is_last = ks >= rem

                if crl is not None:
                    lf = crl
                    crl = None
                else:
                    ca_r = cam[:, :, :, cbs:] if cam != "full" else cam
                    rtk = (
                        (tok_idx[:, cbs:cbe] if is_dual else tok_idx[:, cbs:])
                        if tok_idx is not None
                        else None
                    )
                    fkw = dict(past_key_values=pkv, use_cache=is_dual)
                    if is_dual:
                        fkw.update(dual_cache=True, replace_position=rp)
                    lf = pad_logits(
                        shift(self.model(region, ca_r, rtk, **fkw).logits),
                        T,
                        cbs,
                        x.device,
                    )

                fm = torch.zeros_like(x, dtype=torch.bool)
                fm[:, cbs:cbe] = mir[:, :bs]

                if is_last:
                    x = _fill(lf, fm)
                    _check_and_fill_eos()
                    crl = None
                    hist and hist.append(x.clone())
                    break
                bp = self._bypass(lf, x, fm, mtid, ks, C["threshold"])
                if bp is not None:
                    x = bp
                    _check_and_fill_eos()
                    crl = None
                    hist and hist.append(x.clone())
                    ins += 1
                    if not (x[:, cbs:cbe] == mtid).any():
                        break
                    continue

                ca_r2 = cam[:, :, :, cbs:] if cam != "full" else cam
                x, _, crl = _info_gain_select(
                    self.model,
                    x,
                    lf,
                    fm,
                    ks,
                    block_start=cbs,
                    block_end=cbe,
                    block_size=bs,
                    past_key_values=pkv,
                    dual_cache=is_dual,
                    replace_position=rp,
                    attention_mask=ca_r2,
                    tok_idx=tok_idx,
                    **ig,
                )
                _check_and_fill_eos()
                hist and hist.append(x.clone())
                ins += 1
                if not (x[:, cbs:cbe] == mtid).any():
                    break

        return (
            x
            if not C["return_dict"]
            else BaseSamplerOutput(sequences=x, histories=hist)
        )

    @staticmethod
    def _bypass(logits, x, mi, mtid, k, threshold):
        if threshold is None:
            return None
        ml = logits[mi]
        if ml.numel() == 0:
            return None
        conf, x0f = _sample(ml)
        fc = torch.full(mi.shape, -torch.inf, device=x.device, dtype=logits.dtype)
        fc[mi] = conf
        x0c = torch.full_like(x, mtid)
        x0c[mi] = x0f
        _, sel = torch.topk(fc[0], min(k, int(mi.sum().item())))
        if fc[0, sel[0]] < threshold:
            return None
        tr = mi.new_zeros(mi.shape)
        tr[0, sel] = True
        for i in range(1, len(sel)):
            if fc[0, sel[i]] < threshold:
                tr[0, sel[i]] = False
        tr &= mi
        if not tr.any():
            return None
        xo = x.clone()
        xo[tr] = x0c[tr]
        return xo

    @torch.no_grad()
    def infill(self, inputs, config=None, **kwargs):
        raise NotImplementedError
