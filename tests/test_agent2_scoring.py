from shapely.geometry import Polygon
import pytest

from agent2.hindcast.scoring import HypothesisScorer, ScoringConfig


def test_centroid_scoring_decay():
    scorer = HypothesisScorer(ScoringConfig(centroid_sigma_km=3.0))

    # Identical coordinates
    score_0, dist_0 = scorer.score_centroid((80.0, 15.0), (80.0, 15.0))
    assert dist_0 == 0.0
    assert score_0 == 1.0

    # Near point (~3 km)
    score_near, dist_near = scorer.score_centroid((80.0, 15.0), (80.025, 15.0))
    # Far point (~30 km)
    score_far, dist_far = scorer.score_centroid((80.0, 15.0), (80.25, 15.0))

    assert dist_near < dist_far
    assert score_near > score_far
    assert score_far < 0.01


def test_iou_scoring():
    scorer = HypothesisScorer()

    poly1 = Polygon([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])
    # Identical polygon
    assert scorer.score_iou(poly1, poly1) == 1.0

    # Half overlap
    poly_half = Polygon([[0.5, 0], [1.5, 0], [1.5, 1], [0.5, 1], [0.5, 0]])
    iou_half = scorer.score_iou(poly1, poly_half)
    # intersection = 0.5 * 1.0 = 0.5, union = 1.5 * 1.0 = 1.5 -> 0.5 / 1.5 = 0.333
    assert abs(iou_half - 1.0 / 3.0) < 0.01

    # Disjoint polygon
    poly_disjoint = Polygon([[2, 2], [3, 2], [3, 3], [2, 3], [2, 2]])
    assert scorer.score_iou(poly1, poly_disjoint) == 0.0


def test_composite_scoring():
    scorer = HypothesisScorer(ScoringConfig(
        weight_centroid=0.4, weight_iou=0.4, weight_area=0.1, weight_shape=0.1
    ))
    poly = Polygon([[80.0, 15.0], [80.02, 15.0], [80.02, 15.02], [80.0, 15.02], [80.0, 15.0]])
    centroid = (80.01, 15.01)

    eval_perfect = scorer.evaluate(
        obs_geom=poly,
        obs_centroid=centroid,
        obs_area_km2=5.0,
        sim_geom=poly,
        sim_centroid=centroid,
        sim_area_km2=5.0
    )
    assert eval_perfect["composite_score"] == 1.0
    assert eval_perfect["iou_score"] == 1.0
