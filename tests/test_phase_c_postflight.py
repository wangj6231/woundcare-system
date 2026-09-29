import numpy as np
import pytest

from experiments.phase_c_postflight import verify_mask_representation
from experiments.phase_c_execution import PhaseCError


def test_historical_empty_materialization_is_valid_without_pixels():
    # The unchanged historical np.asarray([]).reshape(-1, 512, 512) is float64.
    empty = np.asarray([]).reshape(-1, 512, 512)
    assert verify_mask_representation(empty, 0) == "EMPTY_HISTORICAL_ARRAY"


def test_nonempty_float_masks_are_not_silently_accepted():
    with pytest.raises(PhaseCError, match="FAIL_RESULT_INTEGRITY"):
        verify_mask_representation(np.zeros((1, 512, 512)), 1)


def test_nonempty_boolean_masks_match_prediction_count():
    assert verify_mask_representation(np.zeros((2, 512, 512), dtype=bool), 2) == "BOOLEAN_MASKS"
    with pytest.raises(PhaseCError, match="FAIL_RESULT_INTEGRITY"):
        verify_mask_representation(np.zeros((2, 512, 512), dtype=bool), 3)


def test_empty_mask_with_wrong_spatial_shape_rejected():
    with pytest.raises(PhaseCError, match="FAIL_RESULT_INTEGRITY"):
        verify_mask_representation(np.zeros((0, 768, 768)), 0)
