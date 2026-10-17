"""Independent equations and numerical limits for equal-cell CPU comparisons."""
import math

import pytest
import torch

from scripts.pisa_complement_cpu import METHODS, compare, declared_cases


@pytest.fixture(scope='module')
def cases():
    torch.set_num_threads(2)
    return {name: (q, k, v, selected, compare(q, k, v, selected))
            for name, q, k, v, selected in declared_cases()}


def test_exact_cell_oracle_matches_dense(cases):
    for *_, result in cases.values():
        torch.testing.assert_close(result['exact_oracle'], result['dense'], atol=1e-12, rtol=1e-12)


def test_independent_cell_means_and_numerators(cases):
    # Direct scalar loops, without the implementation's batched contraction.
    q, k, v, selected, result = cases['heterogeneous_alphas']
    for head in range(q.shape[0]):
        moments = []
        for cell in range(k.shape[1]):
            mean = sum(k[head, cell]) / k.shape[2]
            moments.append(sum(torch.outer(vv, kk-mean) for kk, vv in zip(k[head, cell], v[head, cell])))
        residual = result['remaining']
        conditional = sum(moments[j] for j in residual)/len(residual)
        global_mean = sum(moments)/len(moments)
        torch.testing.assert_close(result['conditional_h'][head], conditional, atol=1e-13, rtol=1e-13)
        for query in range(q.shape[1]):
            t = q[head, query]/math.sqrt(k.shape[-1])
            alpha = {j: torch.exp(t @ k[head, j].mean(0)) for j in residual}
            denominator = torch.zeros((), dtype=torch.float64)
            numerator = torch.zeros(v.shape[-1], dtype=torch.float64)
            for j in selected:
                for kk, vv in zip(k[head, j], v[head, j]):
                    weight = torch.exp(t @ kk)
                    denominator += weight
                    numerator += weight*vv
            for j in residual:
                denominator += k.shape[2]*alpha[j]
                numerator += alpha[j]*v[head, j].sum(0)
            terms = {'zero_order': torch.zeros_like(numerator),
                     'global_mean': sum(alpha.values())*(global_mean@t),
                     'complement_mean': sum(alpha.values())*(conditional@t),
                     'weighted_cells': sum(alpha[j]*(moments[j]@t) for j in residual)}
            for name in METHODS:
                torch.testing.assert_close(result['outputs'][name][head, query],
                                           (numerator+terms[name])/denominator, atol=1e-13, rtol=1e-13)
            assert result['approximate_lse'][head, query].item() == pytest.approx(denominator.log().item())


@pytest.mark.parametrize('name', ['empty_complement', 'homogeneous_keys'])
def test_exact_degenerate_limits(cases, name):
    result = cases[name][-1]
    for output in result['outputs'].values():
        assert torch.isfinite(output).all()
        torch.testing.assert_close(output, result['dense'], atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(result['approximate_lse'], result['dense_lse'], atol=1e-13, rtol=1e-13)


def test_large_common_logit_preserves_outputs(cases):
    base, shifted = (cases[name][-1] for name in ('heterogeneous_alphas', 'stable_large_common_logit'))
    assert shifted['approximate_lse'].min() > 1000
    for name in METHODS:
        assert torch.isfinite(shifted['outputs'][name]).all()
        torch.testing.assert_close(shifted['outputs'][name], base['outputs'][name], atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(shifted['dense'], base['dense'], atol=2e-10, rtol=2e-10)
    torch.testing.assert_close(shifted['approximate_lse']-base['approximate_lse'],
                               torch.full_like(base['approximate_lse'], 2000.), atol=1e-10, rtol=0)


def test_homogeneous_H_all_first_order_controls_agree(cases):
    result = cases['homogeneous_H'][-1]
    for name in ('global_mean', 'weighted_cells'):
        torch.testing.assert_close(result['outputs'][name], result['outputs']['complement_mean'], atol=1e-13, rtol=1e-13)


def test_selection_changes_conditional_mean_not_global_mean(cases):
    q, k, v, _, first = cases['selected_large_H_tail_small_H']
    second = compare(q, k, v, [1])
    torch.testing.assert_close(first['global_h'], second['global_h'], atol=0, rtol=0)
    assert not torch.allclose(first['conditional_h'], second['conditional_h'])
    assert first['global_h'].norm() > first['conditional_h'].norm()*10


def test_heterogeneous_alpha_weighting_is_distinct(cases):
    result = cases['heterogeneous_alphas'][-1]
    assert not torch.allclose(result['outputs']['weighted_cells'], result['outputs']['complement_mean'])
    # The denominator is common even though centroid approximation differs from dense.
    assert (result['approximate_lse']-result['dense_lse']).abs().max() > 1e-5


def test_empty_exact_support_is_well_defined(cases):
    result = cases['no_selected_cells'][-1]
    for output in result['outputs'].values():
        assert torch.isfinite(output).all()
    torch.testing.assert_close(result['global_h'], result['conditional_h'], atol=0, rtol=0)


@pytest.mark.parametrize('selected', [[0, 0], [-1], [4], [True], [.5]])
def test_invalid_cell_support_rejected(cases, selected):
    q, k, v, _, _ = cases['homogeneous_H']
    with pytest.raises(ValueError):
        compare(q, k, v, selected)


def test_partition_reordering_does_not_change_operator(cases):
    q, k, v, _, _ = cases['heterogeneous_alphas']
    order = torch.tensor([2, 0, 3, 1])
    original = compare(q, k, v, [0, 3])
    reordered = compare(q, k[:, order], v[:, order], [1, 2])
    for name in METHODS:
        torch.testing.assert_close(original['outputs'][name], reordered['outputs'][name], atol=1e-13, rtol=1e-13)


def test_optional_exact_text_matches_explicit_dense_and_sparse(cases):
    q, k, v, selected, _ = cases['heterogeneous_alphas']
    text_k, text_v = k[:, 2, :2].clone()+.7, v[:, 2, :2].clone()-1.2
    result = compare(q, k, v, selected, exact_k=text_k, exact_v=text_v)
    all_k = torch.cat((k.flatten(1, 2), text_k), 1)
    all_v = torch.cat((v.flatten(1, 2), text_v), 1)
    direct = ((q/math.sqrt(k.shape[-1])) @ all_k.transpose(1, 2)).softmax(-1) @ all_v
    torch.testing.assert_close(result['dense'], direct, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(result['exact_oracle'], direct, atol=1e-13, rtol=1e-13)
    kept_k, kept_v = torch.cat((k[:, selected].flatten(1, 2), text_k), 1), torch.cat((v[:, selected].flatten(1, 2), text_v), 1)
    sparse = ((q/math.sqrt(k.shape[-1])) @ kept_k.transpose(1, 2)).softmax(-1) @ kept_v
    torch.testing.assert_close(result['sparse'], sparse, atol=1e-13, rtol=1e-13)
    torch.testing.assert_close(result['approximate_exact_mass']+result['approximate_residual_cell_mass'].sum(-1),
                               torch.ones_like(result['approximate_exact_mass']), atol=1e-13, rtol=1e-13)
    q, k, v, selected, _ = cases['homogeneous_keys']
    homogeneous = compare(q, k, v, selected, exact_k=text_k, exact_v=text_v)
    for output in homogeneous['outputs'].values():
        torch.testing.assert_close(output, homogeneous['dense'], atol=1e-13, rtol=1e-13)
