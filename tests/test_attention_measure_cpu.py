"""Synthetic identities and failure cases for known attention summaries."""
import math

import pytest
import torch

from scripts.attention_measure_cpu import (
    declared_cases, dense_attention, evaluate_case, moments, partition_dependence,
    subtract_moments, summarized_attention, token_partition,
)


@pytest.fixture(scope='module')
def cases():
    torch.set_num_threads(2)
    return {case[0]: evaluate_case(*case) for case in declared_cases()}


def test_unique_union_and_complement_moment_subtraction(cases):
    case = cases['heterogeneous_gaussian']
    assert case['retained_unique'] == [0, 1, 2, 3, 6]
    assert case['complement'] == [4, 5, 7, 8, 9, 10, 11]
    assert case['accounted_unique_count'] == case['original_count'] == 12
    assert case['explicit_multiset_count'] == 7
    assert case['complement_statistics_vs_direct']['count_error'] == 0
    for key in ('sum_k', 'sum_v', 'sum_vk'):
        assert case['complement_statistics_vs_direct'][key]['max_abs'] < 1e-13


@pytest.mark.parametrize('groups', [{'local': [-1]}, {'halo': [3]}, {'register': [True]}, {'local': [1.5]}])
def test_invalid_token_ids_are_rejected(groups):
    with pytest.raises(ValueError):
        token_partition(3, groups)


def test_zero_complement_is_exact(cases):
    case = cases['zero_complement']
    assert case['complement_statistics']['count'] == 0
    for approx in case['approximations'].values():
        assert approx['output_error']['max_abs'] < 1e-13


def test_identical_keys_count_weighted_exact(cases):
    case = cases['identical_keys']
    for approx in case['approximations'].values():
        assert approx['output_error']['max_abs'] < 1e-13
    q = torch.zeros(1, 1, dtype=torch.float64)
    k = torch.zeros(3, 1, dtype=torch.float64)
    v = torch.tensor([[0.], [3.], [3.]], dtype=torch.float64)
    exact, _ = dense_attention(q, k, v)
    correct, _ = summarized_attention(q, k[:1], v[:1], [moments(k[1:], v[1:])], 0)
    unweighted_centroids = torch.tensor([[0.], [3.]], dtype=torch.float64).mean(0)
    assert exact.item() == correct.item() == 2.
    assert unweighted_centroids.item() == 1.5


def test_duplicate_concatenation_changes_attention_measure(cases):
    case = cases['overlap_bias']
    assert case['accounted_unique_count'] == 4
    assert case['naive_total_count'] == 6
    assert case['dense_output'] == [[1.]]
    assert case['naive_duplicate_concatenation_error']['max_abs'] == pytest.approx(1.)
    assert case['exact_complement_oracle_error']['max_abs'] < 1e-13


def test_exact_complement_oracle_matches_dense_for_every_declared_case(cases):
    for case in cases.values():
        assert case['exact_complement_oracle_error']['max_abs'] < 1e-12
        assert case['exact_complement_oracle_log_partition_error']['max_abs'] < 1e-12


def test_large_logits_remain_finite(cases):
    case = cases['stable_large_common_logit']
    assert case['dense_log_partition'][0] > 1000
    for approx in case['approximations'].values():
        assert math.isfinite(approx['output'][0][0])
        assert math.isfinite(approx['log_partition_error']['max_abs'])


def test_first_order_is_not_reliable_for_bimodal_large_logits(cases):
    case = cases['bimodal_non_gaussian_first_order_failure']
    first = case['approximations']['centered_first_order']
    assert first['output'][0][0] == pytest.approx(16/3)
    assert first['value_coordinate_hull_violation_max'] > 4
    assert first['output_error']['l2'] > case['approximations']['count_weighted_mean']['output_error']['l2']


def test_centered_first_order_has_correct_derivative_at_zero_query():
    k = torch.tensor([[-1., 2.], [2., 0.], [1., -2.]], dtype=torch.float64)
    v = torch.tensor([[1.], [-2.], [3.]], dtype=torch.float64)
    direction = torch.tensor([[.7, -.2]], dtype=torch.float64)
    epsilon = 1e-5
    full = moments(k, v)
    plus, _ = dense_attention(epsilon*direction, k, v)
    minus, _ = dense_attention(-epsilon*direction, k, v)
    approx_plus, _ = summarized_attention(epsilon*direction, k[:0], v[:0], [full], 1)
    approx_minus, _ = summarized_attention(-epsilon*direction, k[:0], v[:0], [full], 1)
    torch.testing.assert_close((plus-minus)/(2*epsilon), (approx_plus-approx_minus)/(2*epsilon), rtol=1e-9, atol=1e-9)


def test_partition_changes_approximation_but_not_dense_target():
    result = partition_dependence()
    cases = {case['name']: case for case in result['cases']}
    dense = cases['no_explicit_tokens']['dense_output'][0][0]
    assert dense == pytest.approx(1/(math.exp(-2)+1+math.exp(2)))
    assert all(case['dense_output'][0][0] == dense for case in cases.values())
    first = lambda name: cases[name]['approximations']['centered_first_order']['output'][0][0]
    assert first('no_explicit_tokens') == pytest.approx(1/3)
    assert first('positive_key_explicit') == pytest.approx(2*math.exp(-1)/(math.exp(2)+2*math.exp(-1)))
    assert first('negative_key_explicit') == pytest.approx(0.)
    assert first('all_tokens_explicit') == pytest.approx(dense)
    assert result['singleton_summary_error']['max_abs'] < 1e-13


def test_negative_mass_from_naive_repeated_subtraction_is_rejected():
    k = torch.ones(2, 1, dtype=torch.float64)
    with pytest.raises(ValueError, match='union accounting'):
        subtract_moments(moments(k, k), moments(k.repeat(2, 1), k.repeat(2, 1)))


def test_global_covariance_shortcut_changes_only_covariance(cases):
    case = cases['heterogeneous_groups_covariance_comparison']
    q, k, v = [torch.tensor(case[name], dtype=torch.float64) for name in ('q', 'k', 'v')]
    kept, complement, _ = token_partition(len(k), case['groups'])
    stats = moments(k[complement], v[complement])
    covariance = torch.tensor(case['covariance_comparison']['global_covariance'], dtype=torch.float64)
    scaled = q/math.sqrt(k.shape[1])
    exact_weights = (scaled @ k[kept].T).exp()
    complement_mass = stats.count*(scaled @ (stats.sum_k/stats.count)).exp()
    corrected = stats.sum_v/stats.count + scaled @ covariance.T
    expected = (exact_weights @ v[kept] + complement_mass[:, None]*corrected) / (
        exact_weights.sum(-1)+complement_mass)[:, None]
    shortcut = case['approximations']['global_covariance_shortcut']
    torch.testing.assert_close(torch.tensor(shortcut['output'], dtype=torch.float64), expected, rtol=1e-12, atol=1e-12)
    assert shortcut['log_partition_error'] == case['approximations']['centered_first_order']['log_partition_error']
    assert shortcut['output'] != case['approximations']['centered_first_order']['output']
