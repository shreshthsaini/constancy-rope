import pytest
import torch
from crope.deterministic_allocation import DeterministicAllocationSEGA
from crope.sega_official import OfficialSEGA


@pytest.mark.parametrize('enabled,warn', [(False, False), (True, True), (True, False)])
def test_allocation_policy_restored_even_on_failure(monkeypatch, enabled, warn):
    previous = (torch.are_deterministic_algorithms_enabled(),
                torch.is_deterministic_algorithms_warn_only_enabled())
    def fail(self, features, sigma):
        assert torch.are_deterministic_algorithms_enabled()
        assert not torch.is_deterministic_algorithms_warn_only_enabled()
        raise ValueError('injected allocation failure')
    monkeypatch.setattr(OfficialSEGA, 'update', fail)
    try:
        torch.use_deterministic_algorithms(enabled, warn_only=warn)
        with pytest.raises(ValueError, match='injected'):
            DeterministicAllocationSEGA((128, 128)).update(None, 1.)
        assert torch.are_deterministic_algorithms_enabled() == enabled
        assert torch.is_deterministic_algorithms_warn_only_enabled() == warn
    finally:
        torch.use_deterministic_algorithms(previous[0], warn_only=previous[1])


def test_cpu_policy_preserves_formula_and_repeatability():
    x = torch.randn(1, 128 * 128, 4)
    ordinary = OfficialSEGA((128, 128))
    deterministic = DeterministicAllocationSEGA((128, 128))
    a = ordinary.update(x, .8)
    b = deterministic.update(x, .8)
    assert torch.equal(a, b)
    assert ordinary.diagnostics == deterministic.diagnostics
    assert torch.equal(b, deterministic.update(x, .8))
