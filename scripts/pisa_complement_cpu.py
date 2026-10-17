"""Bounded CPU comparison of known PISA block covariance approximations.

This is an equal-cell implementation comparison, not a full PISA reproduction.
Every approximate control uses Equation 3's centroid denominator. No bound
from the paper's exact-denominator analysis is imported.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import torch


METHODS = ('zero_order', 'global_mean', 'complement_mean', 'weighted_cells')
SEED = 20261017


def compare(q, k, v, selected, exact_k=None, exact_v=None):
    """Q[H,Q,D], K[H,C,B,D], V[H,C,B,Dv]; whole selected cells.

    H_j = sum_n v_jn (k_jn - mean_k_j)^T is an unnormalized
    cross-moment. Optional exact_k/exact_v[H,T,D/Dv] retain text tokens once.
    Using covariance H_j/B without restoring B is incorrect.
    A common per-query logit shift stabilizes all shared numerators/masses.
    """
    if not (q.ndim == 3 and k.ndim == v.ndim == 4
            and q.shape[0] == k.shape[0] == v.shape[0]
            and k.shape[:3] == v.shape[:3] and q.shape[-1] == k.shape[-1]
            and min(*q.shape, *k.shape, *v.shape) > 0):
        raise ValueError('Expected nonempty compatible Q[H,Q,D], K/V[H,C,B,D/Dv]')
    if not all(x.device.type == 'cpu' and x.dtype == torch.float64
               and bool(torch.isfinite(x).all()) for x in (q, k, v)):
        raise ValueError('Finite CPU float64 tensors required')
    if any(type(j) is not int or not 0 <= j < k.shape[1] for j in selected):
        raise ValueError('Invalid selected cell ID')
    if len(set(selected)) != len(selected):
        raise ValueError('Selected cell IDs must be unique')
    if (exact_k is None) != (exact_v is None):
        raise ValueError('Both exact key and value tensors are required')
    if exact_k is None:
        exact_k, exact_v = k.new_empty(k.shape[0], 0, k.shape[-1]), v.new_empty(v.shape[0], 0, v.shape[-1])
    if not (exact_k.ndim == exact_v.ndim == 3
            and exact_k.shape[:2] == exact_v.shape[:2]
            and exact_k.shape[0] == k.shape[0]
            and exact_k.shape[-1] == k.shape[-1] and exact_v.shape[-1] == v.shape[-1]
            and all(x.device.type == 'cpu' and x.dtype == torch.float64
                    and bool(torch.isfinite(x).all()) for x in (exact_k, exact_v))):
        raise ValueError('Invalid exact token shapes, dtype or values')
    selected = sorted(selected)
    remaining = [j for j in range(k.shape[1]) if j not in selected]
    t = q / math.sqrt(k.shape[-1])
    mean_k = k.mean(2)
    sum_v = v.sum(2)
    cross = torch.einsum('hcbi,hcbj->hcij', v, k - mean_k[:, :, None, :])
    logits = torch.einsum('hqd,hcbd->hqcb', t, k)
    extra_logits = t @ exact_k.transpose(1, 2)
    full_logits = torch.cat((logits.flatten(2), extra_logits), -1)
    full_values = torch.cat((v.flatten(1, 2), exact_v), 1)
    dense = full_logits.softmax(-1) @ full_values
    dense_lse = full_logits.logsumexp(-1)
    # Independent grouped exact oracle: conditional values and exact cell LSE.
    cell_lse = logits.logsumexp(-1)
    cell_values = torch.einsum('hqcb,hcbv->hqcv', logits.softmax(-1), v)
    group_lse = torch.cat((cell_lse, extra_logits), -1)
    group_values = torch.cat((cell_values, exact_v[:, None].expand(-1, q.shape[1], -1, -1)), 2)
    exact_oracle = (group_lse.softmax(-1)[..., None] * group_values).sum(2)
    kept_logits = torch.cat((logits[:, :, selected].flatten(2), extra_logits), -1)
    kept_values = torch.cat((v[:, selected].flatten(1, 2), exact_v), 1)
    alpha_logits = torch.einsum('hqd,hcd->hqc', t, mean_k[:, remaining])
    shift = torch.cat((kept_logits, alpha_logits), -1).amax(-1)
    kept_weights = (kept_logits - shift[..., None]).exp()
    exact_numerator = kept_weights @ kept_values
    alpha = (alpha_logits - shift[..., None]).exp()
    denominator = kept_weights.sum(-1) + k.shape[2] * alpha.sum(-1)
    numerator0 = exact_numerator + torch.einsum('hqc,hcv->hqv', alpha, sum_v[:, remaining])
    global_h = cross.mean(1)
    conditional_h = cross[:, remaining].mean(1) if remaining else torch.zeros_like(global_h)
    global_term = alpha.sum(-1)[..., None] * torch.einsum('hvd,hqd->hqv', global_h, t)
    conditional_term = alpha.sum(-1)[..., None] * torch.einsum('hvd,hqd->hqv', conditional_h, t)
    weighted_term = torch.einsum('hqc,hcvd,hqd->hqv', alpha, cross[:, remaining], t)
    terms = (torch.zeros_like(numerator0), global_term, conditional_term, weighted_term)
    outputs = {name: (numerator0 + term) / denominator[..., None]
               for name, term in zip(METHODS, terms)}
    sparse = kept_logits.softmax(-1) @ kept_values if kept_logits.shape[-1] else None
    sparse_lse = kept_logits.logsumexp(-1) if kept_logits.shape[-1] else None
    return dict(outputs=outputs, dense=dense, dense_lse=dense_lse, sparse=sparse, sparse_lse=sparse_lse,
                exact_oracle=exact_oracle, approximate_lse=shift + denominator.log(),
                approximate_exact_mass=kept_weights.sum(-1)/denominator,
                approximate_residual_cell_mass=k.shape[2]*alpha/denominator[..., None],
                denominator_scaled=denominator, logit_shift=shift,
                numerator0_scaled=numerator0, alpha_scaled=alpha,
                cross_moments=cross, global_h=global_h, conditional_h=conditional_h,
                selected=selected, remaining=remaining)


def declared_cases():
    """Cases fixed before results; no fitted coefficients or selected seeds."""
    generator = torch.Generator().manual_seed(SEED)
    h, c, b, d, nq = 2, 4, 4, 16, 3
    q = torch.zeros(h, nq, d, dtype=torch.float64)
    q[:, :, 0] = math.sqrt(d) * torch.tensor([.25, .5, 1.], dtype=torch.float64)
    q[1, :, 0] *= -.75
    q[:, :, 1] = .1
    pattern = torch.tensor([-1., -.5, .5, 1.], dtype=torch.float64)
    k = torch.zeros(h, c, b, d, dtype=torch.float64)
    k[..., 0] = .2 * pattern
    k[..., 1] = .05 * pattern.flip(0)
    # Other value channels keep vector error meaningful, without changing H.
    v = .1 * torch.randn(h, c, 1, d, generator=generator, dtype=torch.float64).expand(h, c, b, d).clone()
    v[..., 0] += .5

    def case(name, scales, means=None, selected=(0,), identical=False, common_shift=0.):
        kk, vv = k.clone(), v.clone()
        vv[..., 0] += torch.tensor(scales, dtype=torch.float64)[None, :, None] * pattern
        if means is not None:
            kk[..., 0] += torch.tensor(means, dtype=torch.float64)[None, :, None]
        if identical:
            kk.zero_()
            kk[..., 0] = .4
        # Every Q has the same second component, so this is a common logit offset.
        kk[..., 1] += common_shift
        return name, q.clone(), kk, vv, list(selected)

    return [
        case('selected_large_H_tail_small_H', [20., .2, .2, .2]),
        case('selected_small_H_tail_large_H', [.2, 20., 20., 20.]),
        case('homogeneous_H', [1., 1., 1., 1.]),
        case('heterogeneous_alphas', [.2, 3., .2, 1.], [-1., 2., -2., .5]),
        case('stable_large_common_logit', [.2, 3., .2, 1.], [-1., 2., -2., .5], common_shift=80000.),
        case('empty_complement', [1., 2., 3., 4.], selected=(0, 1, 2, 3)),
        case('no_selected_cells', [1., 2., 3., 4.], selected=()),
        case('homogeneous_keys', [1., 2., 3., 4.], identical=True),
    ]


def error(actual, reference):
    delta = actual - reference
    norm = reference.norm().item()
    return dict(max_abs=delta.abs().max().item(), l2=delta.norm().item(),
                relative_l2=delta.norm().item()/norm if norm > 1e-15 else None,
                per_head_query_l2=delta.norm(dim=-1).tolist())


def report():
    records = []
    for name, q, k, v, selected in declared_cases():
        result = compare(q, k, v, selected)
        record = dict(name=name, q=q.tolist(), k=k.tolist(), v=v.tolist(), selected=selected,
                      remaining=result['remaining'], dense=result['dense'].tolist(),
                      oracle_error=error(result['exact_oracle'], result['dense']),
                      approximate_lse=result['approximate_lse'].tolist(),
                      approximate_exact_mass=result['approximate_exact_mass'].tolist(),
                      approximate_residual_cell_mass=result['approximate_residual_cell_mass'].tolist(),
                      sparse=None if result['sparse'] is None else dict(output=result['sparse'].tolist(), error=error(result['sparse'], result['dense'])),
                      denominator_error=error(result['approximate_lse'], result['dense_lse']),
                      conditional_h=result['conditional_h'].tolist(),
                      global_h=result['global_h'].tolist(),
                      controls={method: dict(output=output.tolist(), error=error(output, result['dense']))
                                for method, output in result['outputs'].items()})
        records.append(record)
    return dict(scope='Synthetic FP64 CPU implementation comparison of known equal-cell PISA approximations; no model, efficacy, novelty or runtime claim',
                source='https://arxiv.org/html/2602.01077v2', seed=SEED,
                torch=torch.__version__, threads=torch.get_num_threads(),
                script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                equations=dict(t='q/sqrt(D)', H_j='sum_n v_jn (k_jn-mean_k_j)^T',
                               D='Z_exact + B sum_unselected alpha_j',
                               N0='N_exact + sum_unselected alpha_j sum_n v_jn',
                               global_mean='(N0 + sum(alpha) Hbar_all t)/D',
                               complement_mean='(N0 + sum(alpha) Hbar_unselected t)/D',
                               weighted_cells='(N0 + sum(alpha_j H_j t))/D'),
                limits=['All controls share approximate centroid denominator, exact support, N0 and post-scaling Q/K/V.',
                        'No exact-denominator theorem bound is imported into this implementation comparison.',
                        'Equal complete cells only; no low rank, geometry prefix, partial cells, routing or full PISA implementation.',
                        'No tuning or clipping. First-order correction need not preserve the value convex hull.',
                        'All prespecified cases reported; no selected aggregate or quality claim.'], cases=records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    payload = report()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
    print(json.dumps({case['name']: {name: control['error']['relative_l2']
                                   for name, control in case['controls'].items()}
                      for case in payload['cases']}, indent=2))


if __name__ == '__main__':
    main()
