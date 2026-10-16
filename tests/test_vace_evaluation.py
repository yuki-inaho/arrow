"""Independent mask/provenance checks; media IO is tested by the reused helper."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


SPEC = importlib.util.spec_from_file_location(
    "evaluate_vace_freeze_test", Path(__file__).parents[1] / "scripts/evaluate_vace_freeze.py")
assert SPEC is not None and SPEC.loader is not None
evaluation = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(evaluation)


def three_pixels():
    source = np.array([[[[10, 10, 10], [20, 20, 20], [30, 30, 30]]]], dtype=np.uint8)
    draft = np.array([[[[10, 10, 10], [40, 40, 40], [127, 127, 127]]]], dtype=np.uint8)
    raw = np.array([[[[80, 80, 80], [90, 90, 90], [100, 100, 100]]]], dtype=np.uint8)
    # Literal expected result, independently specified (not np.where).
    composite = np.array([[[[10, 10, 10], [40, 40, 40], [100, 100, 100]]]], dtype=np.uint8)
    edit = np.array([[[False, False, True]]])
    provenance = {"source_preserved_mask": np.array([[[True, False, False]]]),
                  "warped_known_mask": np.array([[[False, True, False]]])}
    return source, draft, raw, composite, edit, provenance


def test_three_distinct_provenance_pixels():
    report, _ = evaluation.analyze_pixels(*three_pixels())
    assert report["integrity_failures"] == []
    regions = report["regions"]
    assert regions["source_preserved"]["source_to_composite_mae_255"] == 0
    assert regions["warped_known"]["source_to_composite_mae_255"] == 20
    assert regions["unknown_generation"]["source_to_composite_mae_255"] == 70
    assert regions["unknown_generation"]["raw_to_composite_mae_255"] == 0
    assert regions["known_unclassified"]["pixel_count"] == 0
    assert regions["known_unclassified"]["source_to_composite_mae_255"] is None


@pytest.mark.parametrize("pixel,failed", [(0, "composite_known_equals_draft"),
                                         (1, "composite_known_equals_draft"),
                                         (2, "composite_unknown_equals_raw")])
def test_detects_corruption_in_each_region(pixel, failed):
    source, draft, raw, composite, edit, provenance = three_pixels()
    composite[0, 0, pixel, 1] += 1
    report, _ = evaluation.analyze_pixels(source, draft, raw, composite, edit, provenance)
    assert failed in report["integrity_failures"]
    assert report["integrity"][failed]["mismatched_pixels"] == 1


def test_equal_pixels_do_not_imply_source_provenance():
    source, draft, raw, composite, edit, _ = three_pixels()
    report, _ = evaluation.analyze_pixels(source, draft, raw, composite, edit)
    assert report["regions"]["source_preserved"]["status"] == "not_available"
    assert report["regions"]["warped_known"]["status"] == "not_available"
    assert report["regions"]["known_unclassified"]["pixel_count"] == 2


def test_overlapping_provenance_rejected():
    source, draft, raw, composite, edit, provenance = three_pixels()
    provenance["warped_known_mask"][0, 0, 0] = True
    with pytest.raises(ValueError, match="overlaps"):
        evaluation.analyze_pixels(source, draft, raw, composite, edit, provenance)


def test_empty_unknown_not_claimed_verified_and_frame0_source_audited():
    source, draft, raw, _, edit, _ = three_pixels()
    edit[:] = False
    report, _ = evaluation.analyze_pixels(source, draft, raw, draft.copy(), edit)
    assert report["integrity"]["composite_unknown_equals_raw"]["status"] == "not_applicable"
    assert "frame0_no_edit_equals_source" in report["integrity_failures"]


def test_provenance_claim_is_checked_against_source():
    source, draft, raw, composite, edit, provenance = three_pixels()
    provenance["source_preserved_mask"][0, 0, 1] = True
    provenance["warped_known_mask"][0, 0, 1] = False
    report, _ = evaluation.analyze_pixels(source, draft, raw, composite, edit, provenance)
    assert "explicit_source_preserved_equals_source" in report["integrity_failures"]
    assert "explicit_source_draft_equals_source" in report["integrity_failures"]


def test_reference_projected_is_distinct_from_current_observation():
    source, draft, raw, composite, edit, provenance = three_pixels()
    provenance["reference_projected_mask"] = provenance["warped_known_mask"].copy()
    provenance["warped_known_mask"][:] = False
    report, _ = evaluation.analyze_pixels(source, draft, raw, composite, edit, provenance)
    assert report["regions"]["reference_projected"]["pixel_count"] == 1
    assert report["regions"]["reference_projected"]["source_to_composite_mae_255"] == 20
    assert report["regions"]["warped_known"]["pixel_count"] == 0


def test_complete_explicit_provenance_must_partition_all_pixels():
    source, draft, raw, composite, edit, provenance = three_pixels()
    provenance["reference_projected_mask"] = np.zeros(edit.shape, bool)
    provenance["warped_known_mask"][:] = False
    with pytest.raises(ValueError, match="full partition"):
        evaluation.analyze_pixels(source, draft, raw, composite, edit, provenance)
