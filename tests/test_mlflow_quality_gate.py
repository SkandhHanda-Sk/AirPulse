
from src.models.evaluate_residual_hgb import validation_quality_gate_passed


def test_quality_gate_requires_improvement_over_persistence_mae():
    assert validation_quality_gate_passed(9.5, 10.0)
    assert not validation_quality_gate_passed(10.0, 10.0)
    assert not validation_quality_gate_passed(10.5, 10.0)
