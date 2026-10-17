"""Synthetic CPU checks of known mass and first-order attention summaries.

This implements unique token accounting and full raw moments. The centered
first-order exponential expansion is MomentKV-like; complement-mean correction
also appears in PISA. This is not a new attention algorithm or efficacy result.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import torch


@dataclass
class Moments:
    count: int
    sum_k: torch.Tensor
    sum_v: torch.Tensor
    sum_vk: torch.Tensor

    def record(self):
        return {'count': self.count, 'sum_k': self.sum_k.tolist(),
                'sum_v': self.sum_v.tolist(), 'sum_vk': self.sum_vk.tolist()}


def check_tensors(q, k, v):
    if not (q.ndim == k.ndim == v.ndim == 2 and k.shape[0] == v.shape[0]
            and q.shape[1] == k.shape[1] and min(q.shape[0], k.shape[0], k.shape[1], v.shape[1]) > 0):
        raise ValueError('Expected nonempty Q[M,D], K[N,D], V[N,Dv]')
    if not all(x.device.type == 'cpu' and x.dtype == torch.float64 and bool(torch.isfinite(x).all())
               for x in (q, k, v)):
        raise ValueError('This bounded audit requires finite CPU float64 tensors')


def token_partition(count, groups):
    """Union explicit token IDs exactly once, including overlap within a group."""
    if not isinstance(count, int) or count <= 0:
        raise ValueError('Positive token count required')
    flattened = []
    for group in groups.values():
        for index in group:
            if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index >= count:
                raise ValueError('Token IDs must be valid integers')
            flattened.append(index)
    retained = torch.tensor(sorted(set(flattened)), dtype=torch.long)
    complement = torch.tensor(sorted(set(range(count))-set(flattened)), dtype=torch.long)
    return retained, complement, torch.tensor(flattened, dtype=torch.long)


def moments(keys, values):
    if keys.ndim != 2 or values.ndim != 2 or keys.shape[0] != values.shape[0]:
        raise ValueError('Moment shape mismatch')
    return Moments(keys.shape[0], keys.sum(0), values.sum(0), values.T @ keys)


def subtract_moments(total, retained):
    if retained.count > total.count:
        raise ValueError('Retained count exceeds total; union accounting is required')
    return Moments(total.count-retained.count, total.sum_k-retained.sum_k,
                   total.sum_v-retained.sum_v, total.sum_vk-retained.sum_vk)


def dense_attention(q, k, v):
    check_tensors(q, k, v)
    logits = (q / math.sqrt(k.shape[1])) @ k.T
    if not bool(torch.isfinite(logits).all()):
        raise ValueError('Nonfinite logits')
    return logits.softmax(-1) @ v, logits.logsumexp(-1)


def summarized_attention(q, retained_k, retained_v, summaries, order, shared_covariance=None):
    """Count-weighted mean keys, with optional centered K/V covariance term.

    For a block C, t=q/sqrt(D), mu_k=sum_k/n, mu_v=sum_v/n:
      Z_C ~= n exp(t.mu_k)
      N_C ~= Z_C [mu_v + Cov(v,k)t]  (order 1)
    The first-order denominator correction is zero because keys are centered.
    The correction can leave the value convex hull. It is not clipped here.
    """
    if order not in (0, 1):
        raise ValueError('Only zeroth and first order are implemented')
    if shared_covariance is not None and order != 1:
        raise ValueError('A covariance shortcut requires first order')
    scaled_q = q / math.sqrt(q.shape[1])
    logits = [scaled_q @ retained_k.T]
    block_values = []
    for summary in summaries:
        if summary.count < 0:
            raise ValueError('Negative complement mass')
        if summary.count == 0:
            continue
        mean_k, mean_v = summary.sum_k/summary.count, summary.sum_v/summary.count
        logits.append((scaled_q @ mean_k + math.log(summary.count))[:, None])
        block_value = mean_v.expand(q.shape[0], -1)
        if order == 1:
            covariance = (summary.sum_vk - torch.outer(summary.sum_v, mean_k)) / summary.count
            if shared_covariance is not None:
                if shared_covariance.shape != covariance.shape:
                    raise ValueError('Shared covariance shape mismatch')
                covariance = shared_covariance
            block_value = block_value + scaled_q @ covariance.T
        block_values.append(block_value)
    all_logits = torch.cat(logits, dim=1)
    if all_logits.shape[1] == 0 or not bool(torch.isfinite(all_logits).all()):
        raise ValueError('At least one finite mass contribution required')
    weights = all_logits.softmax(-1)
    retained_count = retained_k.shape[0]
    output = weights[:, :retained_count] @ retained_v
    for index, block_value in enumerate(block_values):
        output = output + weights[:, retained_count+index, None] * block_value
    return output, all_logits.logsumexp(-1)


def complement_oracle(q, k, v, retained, complement):
    """Exact grouped numerator/denominator, using every complement token."""
    scaled_q = q / math.sqrt(k.shape[1])
    logits = scaled_q @ k.T
    anchor = logits.max(-1, keepdim=True).values
    kept_weights = (logits[:, retained]-anchor).exp()
    other_weights = (logits[:, complement]-anchor).exp()
    numerator = kept_weights @ v[retained] + other_weights @ v[complement]
    mass = kept_weights.sum(-1) + other_weights.sum(-1)
    return numerator/mass[:, None], anchor[:, 0]+mass.log()


def error(value, reference):
    difference = value-reference
    denominator = reference.norm().item()
    return {'max_abs': difference.abs().max().item(), 'l2': difference.norm().item(),
            'relative_l2': difference.norm().item()/denominator if denominator else None,
            'reference_l2': denominator}


def evaluate_case(name, q, k, v, groups, compare_global_covariance=False):
    check_tensors(q, k, v)
    kept, complement, multiset = token_partition(k.shape[0], groups)
    total = moments(k, v)
    retained = moments(k[kept], v[kept])
    remainder = subtract_moments(total, retained)
    direct = moments(k[complement], v[complement])
    baseline, true_log_mass = dense_attention(q, k, v)
    approximations = {}
    methods = [('count_weighted_mean', 0, None), ('centered_first_order', 1, None)]
    global_covariance = total.sum_vk/total.count - torch.outer(total.sum_v/total.count, total.sum_k/total.count)
    if compare_global_covariance:
        methods.append(('global_covariance_shortcut', 1, global_covariance))
    for label, order, override in methods:
        output, log_mass = summarized_attention(q, k[kept], v[kept], [remainder], order, override)
        approximations[label] = {'output': output.tolist(), 'output_error': error(output, baseline),
            'log_partition_error': error(log_mass, true_log_mass),
            'value_coordinate_hull_violation_max': torch.maximum(
                (output-v.max(0).values).clamp_min(0), (v.min(0).values-output).clamp_min(0)).max().item()}
    oracle, oracle_mass = complement_oracle(q, k, v, kept, complement)
    # A deliberate bad control duplicates overlapped explicit tokens, while the
    # true complement is still included once. This changes the reference measure.
    naive_ids = torch.cat((multiset, complement))
    naive, _ = dense_attention(q, k[naive_ids], v[naive_ids])
    return {'name': name, 'q': q.tolist(), 'k': k.tolist(), 'v': v.tolist(), 'groups': groups,
            'retained_unique': kept.tolist(), 'complement': complement.tolist(),
            'original_count': k.shape[0], 'explicit_multiset_count': len(multiset),
            'naive_total_count': len(naive_ids), 'accounted_unique_count': len(kept)+remainder.count,
            'complement_statistics': remainder.record(),
            'covariance_comparison': {'global_covariance': global_covariance.tolist(),
                'complement_covariance': (remainder.sum_vk/remainder.count - torch.outer(
                    remainder.sum_v/remainder.count, remainder.sum_k/remainder.count)).tolist(),
                'scope': 'Only the covariance is substituted; complement mean and mass remain unchanged. PISA-inspired global shortcut, not literal PISA reproduction.'}
                if compare_global_covariance and remainder.count else None,
            'complement_statistics_vs_direct': {'count_error': remainder.count-direct.count,
                'sum_k': error(remainder.sum_k, direct.sum_k), 'sum_v': error(remainder.sum_v, direct.sum_v),
                'sum_vk': error(remainder.sum_vk, direct.sum_vk)},
            'dense_output': baseline.tolist(), 'dense_log_partition': true_log_mass.tolist(),
            'approximations': approximations,
            'exact_complement_oracle_error': error(oracle, baseline),
            'exact_complement_oracle_log_partition_error': error(oracle_mass, true_log_mass),
            'naive_duplicate_concatenation_error': error(naive, baseline)}


def declared_cases():
    """Fixed cases/seeds, with no selection based on computed errors."""
    generator = torch.Generator(device='cpu').manual_seed(20261017)
    q = torch.randn(3, 4, generator=generator, dtype=torch.float64)
    k = torch.randn(12, 4, generator=generator, dtype=torch.float64)
    v = torch.randn(12, 3, generator=generator, dtype=torch.float64)
    groups = {'local': [0, 1, 2], 'halo': [2, 3], 'register': [3, 6]}
    scalar_q = torch.tensor([[4.]], dtype=torch.float64)
    return [
        ('zero_complement', q, k, v, {'local': list(range(12)), 'halo': [1, 2], 'register': [2]}),
        ('identical_keys', q, k[:1].expand_as(k).clone(), v, groups),
        ('overlap_bias', torch.zeros(1, 1, dtype=torch.float64),
         torch.zeros(4, 1, dtype=torch.float64), torch.tensor([[0.], [0.], [0.], [4.]], dtype=torch.float64),
         {'local': [0, 3], 'halo': [3], 'register': [3]}),
        ('small_logits_gaussian', .05*q, k, v, groups),
        ('heterogeneous_gaussian', 2*q, k, v, groups),
        ('heterogeneous_groups_covariance_comparison',
         torch.tensor([[.5, .2], [-.3, .4], [1., -.7]], dtype=torch.float64),
         torch.tensor([[3., 0.], [3.5, 1.], [2.5, -1.], [4., .5],
                       [-2., 0.], [-2.5, 1.], [-1.5, -1.], [-3., -.5]], dtype=torch.float64),
         torch.tensor([[4., -1.], [5., 2.], [3., 0.], [6., 1.],
                       [0., 2.], [1., 3.], [-1., 1.], [.5, 2.5]], dtype=torch.float64),
         {'local': [0, 1, 2], 'halo': [2, 3], 'register': [3]}, True),
        ('stable_large_common_logit', torch.ones(1, 1, dtype=torch.float64),
         torch.tensor([[1000.], [1001.], [999.], [1002.]], dtype=torch.float64),
         torch.tensor([[1.], [-2.], [3.], [.5]], dtype=torch.float64),
         {'local': [0], 'halo': [0], 'register': [1]}),
        ('bimodal_non_gaussian_first_order_failure', scalar_q,
         torch.tensor([[0.], [-2.], [2.]], dtype=torch.float64),
         torch.tensor([[0.], [-1.], [1.]], dtype=torch.float64),
         {'local': [0], 'halo': [], 'register': []}),
    ]


def partition_dependence():
    q = torch.ones(1, 1, dtype=torch.float64)
    k = torch.tensor([[-2.], [0.], [2.]], dtype=torch.float64)
    v = torch.tensor([[0.], [1.], [0.]], dtype=torch.float64)
    variants = {'no_explicit_tokens': [], 'positive_key_explicit': [2],
                'negative_key_explicit': [0], 'all_tokens_explicit': [0, 1, 2]}
    cases = [evaluate_case(name, q, k, v, {'local': indices, 'halo': [], 'register': []})
             for name, indices in variants.items()]
    singleton_summaries = [moments(k[i:i+1], v[i:i+1]) for i in range(3)]
    singleton, _ = summarized_attention(q, k[:0], v[:0], singleton_summaries, 1)
    dense, _ = dense_attention(q, k, v)
    return {'description': 'Fixed dense attention target, changed exact/summary partition or complement grouping.',
            'cases': cases, 'singleton_summary_output': singleton.tolist(),
            'singleton_summary_error': error(singleton, dense),
            'interpretation': 'Unique mass accounting preserves the declared target, not approximation partition invariance.'}


def run_audit():
    started = time.perf_counter()
    cases = [evaluate_case(*case) for case in declared_cases()]
    return {'scope': 'Fixed synthetic float64 CPU algebra falsifier, not model/quality/performance evidence',
            'seed': 20261017, 'dtype': 'torch.float64', 'torch': torch.__version__,
            'threads': torch.get_num_threads(), 'cases_declared_before_evaluation': [c[0] for c in declared_cases()],
            'equations': {'scaled_query': 't=q/sqrt(D)',
                'union': 'U=unique(local union halo union register); C=all_token_ids minus U',
                'raw_complement': 'stats(C)=stats(all)-stats(U), after deduplication',
                'zeroth_mass': 'Z_C ~= n_C exp(t dot mean_k_C)',
                'first_order_numerator': 'N_C ~= Z_C [mean_v_C + ((sum_vk_C-sum_v_C mean_k_C^T)/n_C)t]',
                'first_order_denominator': 'Centered first-order denominator term vanishes because sum(k-mean_k)=0'},
            'known_algorithm_boundaries': [
                'Count-weighted centroid mass is known proportional-attention accounting.',
                'Centered first-order K/V covariance correction matches the stated MomentKV-like formula; PISA also treats complement-mean correction.',
                'One fixed heterogeneous case substitutes global covariance into otherwise identical complement means/mass. This is PISA-inspired, not a literal PISA reproduction.',
                'No low-rank projection, compressed KV system, kernel optimization or performance benchmark is implemented.',
                'Register means explicit original token IDs in this test, not extra learned tokens or learned virtual memory.',
                'Raw moments are sufficient for the first-order approximation, not for exact heterogeneous softmax.',
                'No output-error case is selected or tuned after observing results.'],
            'cases': cases, 'partition_dependence': partition_dependence(),
            'novelty_established': False, 'efficacy_evaluated': False,
            'source_sha256': {str(Path(__file__).resolve()): hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
            'wall_seconds_excluding_imports': time.perf_counter()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(2)
    report = run_audit()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2, allow_nan=False)
        handle.write('\n')
    print(json.dumps({case['name']: {name: result['output_error']['relative_l2']
                     for name, result in case['approximations'].items()} for case in report['cases']}, indent=2))


if __name__ == '__main__':
    main()
