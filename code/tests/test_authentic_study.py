from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pytest
import torch

import rsqaoa
from hqml_observable_subspaces import authentic_study as study
from rsqaoa.circuits import MaxCutProblem, RDTYPE
from rsqaoa.operator import QAOASensitivity, adjointness_gap


CODE = Path(__file__).resolve().parents[1]


def _small_problem() -> MaxCutProblem:
    return MaxCutProblem(
        6,
        [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (0, 5), (1, 4)],
        p=1,
    )


def _theta(problem: MaxCutProblem) -> torch.Tensor:
    return torch.linspace(0.17, 0.71, problem.dim, dtype=RDTYPE)


def test_version_and_registered_data_contract() -> None:
    assert rsqaoa.__version__ == "0.5.0"
    assert len(study.GSET_HASHES) == 8
    assert all(len(value) == 64 for value in study.GSET_HASHES.values())
    assert study.GSET_ROOT.startswith("https://web.stanford.edu/")


def test_revised_task_banks_are_disjoint_normalized_and_out_of_span() -> None:
    problem = _small_problem()
    weights = np.ones(problem.m)
    for protocol in study.PROTOCOLS:
        train, test = study.protocol_tasks(problem, weights, protocol)
        assert train.shape == test.shape == (3, problem.m)
        np.testing.assert_allclose(np.linalg.norm(train, axis=1), 1.0)
        np.testing.assert_allclose(np.linalg.norm(test, axis=1), 1.0)
        assert not any(np.array_equal(left, right) for left in train for right in test)
        assert study.relative_span_residual(train, test) > 1e-3


def test_legacy_disjoint_vectors_can_still_be_in_span() -> None:
    problem = _small_problem()
    train, test = study.legacy_tasks(problem, np.ones(problem.m))
    assert not any(np.array_equal(left, right) for left in train for right in test)
    assert study.relative_span_residual(train, test) < 1e-12


def test_leave_one_out_radius_detects_redundancy_and_innovation() -> None:
    redundant = np.asarray([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    independent = np.eye(3)
    assert study.leave_one_out_radius(redundant) == pytest.approx(1.0 / 3.0)
    assert study.leave_one_out_radius(independent) == pytest.approx(1.0)
    repeated = np.asarray([[1.0, 0.0], [1.0, 0.0]])
    assert study.leave_one_out_radius(repeated) < 1e-25


def test_action_factor_matches_dense_robust_objective() -> None:
    problem = _small_problem()
    train, _ = study.protocol_tasks(problem, np.ones(problem.m), "full_objective")
    delta = study.leave_one_out_radius(train)
    operator = QAOASensitivity(problem, _theta(problem))
    task, robust, source_rank, action_count = study.task_and_robust_action_bases(
        operator, train, delta
    )
    jacobian = QAOASensitivity(problem, _theta(problem)).dense_jacobian().numpy()
    dense = study.dense_bases(jacobian, train, delta)
    np.testing.assert_allclose(
        study.relative_discarded_energy(jacobian, train, task[:, :2]),
        study.relative_discarded_energy(
            jacobian, train, dense["task_only"][:, :2]
        ),
        atol=1e-12,
    )
    source, _ = study.span_basis(train)
    moment = train.T @ train / len(train)
    hessian = jacobian.T @ (
        moment + delta * (np.eye(problem.m) - source @ source.T)
    ) @ jacobian
    def loss(basis: np.ndarray) -> float:
        return float(
            np.trace(hessian)
            - np.trace(basis[:, :2].T @ hessian @ basis[:, :2])
        )
    assert loss(robust) == pytest.approx(loss(dense["complement_robust"]), abs=1e-12)
    assert operator.counts.vjp == problem.m
    assert source_rank == np.linalg.matrix_rank(train)
    assert action_count == problem.m


def test_task_only_action_path_uses_exactly_source_rank_queries() -> None:
    problem = _small_problem()
    train, _ = study.protocol_tasks(problem, np.ones(problem.m), "full_objective")
    operator = QAOASensitivity(problem, _theta(problem))
    _, _, source_rank, action_count = study.task_and_robust_action_bases(
        operator, train, 0.0, include_completion=False
    )
    assert action_count == source_rank
    assert operator.counts.vjp == source_rank
    assert source_rank < problem.m


def test_span_transfer_triangle_bound() -> None:
    rng = np.random.default_rng(912)
    jacobian = rng.normal(size=(9, 13))
    source = rng.normal(size=(5, 9))
    target = rng.normal(size=(4, 9))
    q, _ = np.linalg.qr(rng.normal(size=(13, 4)))
    u, _ = study.span_basis(source)
    parallel = u @ (u.T @ target.T)
    orthogonal = target.T - parallel
    lhs = np.linalg.norm((np.eye(13) - q @ q.T) @ jacobian.T @ target.T)
    rhs = (
        np.linalg.norm((np.eye(13) - q @ q.T) @ jacobian.T @ parallel)
        + np.linalg.norm(jacobian, ord=2) * np.linalg.norm(orthogonal)
    )
    assert lhs <= rhs + 1e-12


def test_adjointness() -> None:
    problem = _small_problem()
    assert adjointness_gap(
        QAOASensitivity(problem, _theta(problem)), jvp_mode="autograd"
    ) < 1e-12


def test_locked_study_when_authenticated_cache_is_available(monkeypatch) -> None:
    cache = Path(os.environ.get("RSQAOA_GSET_CACHE", ""))
    if not cache.is_dir() or not all((cache / name).is_file() for name in study.GSET_HASHES):
        pytest.skip("authenticated Gset cache not supplied")
    actual = study.run_study(allow_download=False)
    expected = json.loads((CODE / "results/locked_results.json").read_text())
    study._compare(actual, expected)
    assert all(actual["registered_findings"].values())


def test_manuscript_uses_only_generated_scientific_assets() -> None:
    manuscript = (CODE.parent / "main.tex").read_text(encoding="utf-8")
    for name in (
        "robust_transfer_audit",
        "oracle_cost_audit",
    ):
        assert f"code/manuscript_assets/figures/{name}.pdf" in manuscript
    assert "0.05086" not in manuscript
    assert "untouched test tasks" not in manuscript.lower()
