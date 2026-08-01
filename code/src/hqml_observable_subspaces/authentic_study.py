"""Authenticated source-span audit for repeated QAOA objectives.

The deterministic study has two purposes.

* It detects whether a nominally held-out task bank is actually outside the
  linear span used to construct a task-weighted active subspace.
* It evaluates a complement-robust moment that protects directions missing
  from that source span.  The construction uses only transpose-Jacobian
  products; a dense Jacobian is formed afterward on ten-qubit instances for
  independent audit and baselines.

Every public graph file is accepted only after its registered SHA-256 digest
matches.  Timings are deliberately separated from deterministic replay.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import shutil
import statistics
import sys
import time
import urllib.request
from pathlib import Path
from urllib.parse import urlparse

import matplotlib.pyplot as plt
import numpy as np
import torch

from rsqaoa.circuits import MaxCutProblem, RDTYPE
from rsqaoa.operator import QAOASensitivity, adjointness_gap


GSET_ROOT = "https://web.stanford.edu/~yyye/yyye/Gset"
GSET_HASHES = {
    "G1": "73bf704d8ffc55ba42260ab4cb659e3dcb6e729be70404d2cf476ba4e46d1665",
    "G6": "e3dd8b29205280f1ad5806849321cdff6f11324c75cb95647d5783f3e47a1103",
    "G11": "c2a760d2926db4fefd23b25c098dcd6311f711b355dbd1cc689fa25660c73174",
    "G14": "dc769b978a40d458f693d5bd2cf8b8cceabd430b8e976204746696179c3d5945",
    "G18": "6185af31fae0f7d59e232dad91178753574b7caa111dd6e6bf52f11c80e928d8",
    "G22": "9baeee06eb147b1c9ca42b43be86592d4e6fc60784a85af9be5b63d1362ef28e",
    "G32": "9760fce6b601832db1b1173c1a09c8556d64bfb27ba603ef52ca24ee6c8527cc",
    "G43": "9af5445b4b066cbf1eabe218d4e0d907cb6f211651cae557c761ec344dc37be8",
}
PROTOCOLS = ("full_objective", "localized_contrast")
METHODS = ("complement_robust", "task_only", "unweighted", "coordinate")
RANKS = (1, 2, 3, 4)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def data_cache() -> Path:
    override = os.environ.get("RSQAOA_GSET_CACHE")
    if override:
        return Path(override).expanduser()
    xdg = os.environ.get("XDG_CACHE_HOME")
    if xdg:
        return Path(xdg).expanduser() / "rsqaoa" / "Stanford-Gset"
    return Path.home() / ".cache" / "rsqaoa" / "Stanford-Gset"


def authenticated_gset(name: str, *, allow_download: bool = True) -> Path:
    if name not in GSET_HASHES:
        raise ValueError(f"unregistered Gset instance: {name}")
    destination = data_cache() / name
    if destination.is_file() and sha256(destination) == GSET_HASHES[name]:
        return destination
    if not allow_download:
        raise FileNotFoundError(f"authenticated cache entry is absent: {destination}")
    url = f"{GSET_ROOT}/{name}"
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname != "web.stanford.edu":
        raise RuntimeError("the data URL is outside the Stanford host allowlist")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f"{name}.{os.getpid()}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "rsqaoa/0.5"})
    try:
        with urllib.request.urlopen(request, timeout=90) as response:
            with temporary.open("wb") as output:
                shutil.copyfileobj(response, output)
        actual = sha256(temporary)
        if actual != GSET_HASHES[name]:
            raise ValueError(
                f"SHA-256 mismatch for {name}: expected {GSET_HASHES[name]}, got {actual}"
            )
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return destination


def parse_gset(path: Path) -> tuple[int, list[tuple[int, int, float]]]:
    with path.open(encoding="utf-8") as stream:
        n_vertices, n_edges = map(int, stream.readline().split()[:2])
        edges = []
        for line in stream:
            u, v, weight = line.split()
            edges.append((int(u) - 1, int(v) - 1, float(weight)))
    if len(edges) != n_edges:
        raise ValueError(f"header reports {n_edges} edges; parsed {len(edges)}")
    return n_vertices, edges


def bfs_vertices(
    n_vertices: int,
    edges: list[tuple[int, int, float]],
    start: int,
    size: int = 10,
) -> list[int]:
    adjacency = [[] for _ in range(n_vertices)]
    for u, v, _ in edges:
        adjacency[u].append(v)
        adjacency[v].append(u)
    for neighbors in adjacency:
        neighbors.sort()
    order = [int(start) % n_vertices]
    seen = set(order)
    cursor = 0
    while cursor < len(order) and len(order) < size:
        for neighbor in adjacency[order[cursor]]:
            if neighbor not in seen:
                seen.add(neighbor)
                order.append(neighbor)
                if len(order) == size:
                    break
        cursor += 1
    if len(order) < size:
        for vertex in range(n_vertices):
            if vertex not in seen:
                order.append(vertex)
                if len(order) == size:
                    break
    return order


def local_problem(
    path: Path, *, start: int, size: int = 10
) -> tuple[MaxCutProblem, np.ndarray, list[int]]:
    n_vertices, global_edges = parse_gset(path)
    vertices = bfs_vertices(n_vertices, global_edges, start, size=size)
    lookup = {vertex: local for local, vertex in enumerate(vertices)}
    selected = []
    for u, v, weight in global_edges:
        if u in lookup and v in lookup:
            selected.append((lookup[u], lookup[v], weight))
    selected.sort(key=lambda edge: (edge[0], edge[1]))
    if not selected:
        raise ValueError("the deterministic neighborhood has no induced edges")
    edge_pairs = [(u, v) for u, v, _ in selected]
    original_weights = np.asarray([weight for _, _, weight in selected], dtype=float)
    return MaxCutProblem(size, edge_pairs, p=1), original_weights, vertices


def _signed_weights(original_weights: np.ndarray) -> np.ndarray:
    sign = np.sign(np.asarray(original_weights, dtype=float))
    sign[sign == 0.0] = 1.0
    return sign


def _normalized(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm == 0.0:
        raise ValueError("task vector must be nonzero")
    return np.asarray(vector, dtype=float) / norm


def vertex_task_bank(
    problem: MaxCutProblem,
    original_weights: np.ndarray,
    vertices: list[int] | range,
    *,
    protocol: str,
    strength: float,
) -> np.ndarray:
    """Construct normalized topology-conditioned objective vectors.

    ``full_objective`` retains the signed source objective and changes the
    incident-edge weights.  ``localized_contrast`` removes the common source
    component and keeps only the signed incident-edge change.
    """

    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol: {protocol}")
    sign = _signed_weights(original_weights)
    tasks = []
    for vertex in vertices:
        incident = np.asarray(
            [float(int(vertex) in edge) for edge in problem.edges], dtype=float
        )
        if protocol == "full_objective":
            vector = sign * (1.0 + float(strength) * incident)
        else:
            vector = sign * incident
        tasks.append(_normalized(vector))
    return np.asarray(tasks)


def protocol_tasks(
    problem: MaxCutProblem, original_weights: np.ndarray, protocol: str
) -> tuple[np.ndarray, np.ndarray]:
    development = vertex_task_bank(
        problem,
        original_weights,
        range(0, problem.n, 2),
        protocol=protocol,
        strength=0.4,
    )
    held_out = vertex_task_bank(
        problem,
        original_weights,
        range(1, problem.n, 2),
        protocol=protocol,
        strength=0.8,
    )
    return development, held_out


def legacy_tasks(
    problem: MaxCutProblem, original_weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """The rejected split, retained only to quantify its span leakage."""

    sign = _signed_weights(original_weights)
    banks: dict[float, list[np.ndarray]] = {0.4: [], 0.8: []}
    for strength in banks:
        for vertex in range(problem.n):
            incident = np.asarray(
                [float(vertex in edge) for edge in problem.edges], dtype=float
            )
            banks[strength].append(_normalized(sign * (1.0 + strength * incident)))
    development = banks[0.4] + [banks[0.8][v] for v in range(0, problem.n, 2)]
    held_out = [banks[0.8][v] for v in range(1, problem.n, 2)]
    return np.asarray(development), np.asarray(held_out)


def span_basis(weights: np.ndarray, *, tolerance: float = 1e-12) -> tuple[np.ndarray, int]:
    weights = np.asarray(weights, dtype=float)
    u, singular, _ = np.linalg.svd(weights.T, full_matrices=True)
    if singular.size == 0 or singular[0] == 0.0:
        return u[:, :0], 0
    rank = int(np.sum(singular > tolerance * singular[0]))
    return u[:, :rank], rank


def relative_span_residual(source: np.ndarray, target: np.ndarray) -> float:
    basis, _ = span_basis(source)
    residual = target.T - basis @ (basis.T @ target.T)
    return float(np.linalg.norm(residual) / max(np.linalg.norm(target), 1e-30))


def leave_one_out_radius(weights: np.ndarray) -> float:
    """Average squared source-task innovation; uses no target task or gradient."""

    weights = np.asarray(weights, dtype=float)
    residuals = []
    for index, vector in enumerate(weights):
        remainder = np.delete(weights, index, axis=0)
        basis, _ = span_basis(remainder)
        innovation = vector - basis @ (basis.T @ vector)
        residuals.append(float(innovation @ innovation))
    return float(np.mean(residuals))


def relative_discarded_energy(
    jacobian: np.ndarray, weights: np.ndarray, basis: np.ndarray
) -> float:
    gradients = np.asarray(weights) @ np.asarray(jacobian)
    residual = gradients - (gradients @ basis) @ basis.T
    return float(
        np.sum(residual * residual) / max(float(np.sum(gradients * gradients)), 1e-30)
    )


def task_and_robust_action_bases(
    operator: QAOASensitivity,
    train_weights: np.ndarray,
    delta: float,
    *,
    include_completion: bool = True,
) -> tuple[np.ndarray, np.ndarray, int, int]:
    """Return exact task-only and complement-robust bases from VJPs."""

    weights = np.asarray(train_weights, dtype=float)
    u, singular, _ = np.linalg.svd(weights.T, full_matrices=True)
    rank = int(np.sum(singular > 1e-12 * singular[0]))
    compact = u[:, :rank] * (singular[:rank] / math.sqrt(len(weights)))
    task_columns = [
        operator.vjp(torch.as_tensor(compact[:, j], dtype=RDTYPE)).numpy()
        for j in range(rank)
    ]
    task_action = np.column_stack(task_columns)
    task_basis = np.linalg.svd(task_action, full_matrices=False)[0]

    complement = u[:, rank:] if include_completion else u[:, :0]
    completion_columns = [
        operator.vjp(
            torch.as_tensor(math.sqrt(delta) * complement[:, j], dtype=RDTYPE)
        ).numpy()
        for j in range(complement.shape[1])
    ]
    robust_action = (
        np.column_stack([task_action, np.column_stack(completion_columns)])
        if completion_columns
        else task_action
    )
    robust_basis = np.linalg.svd(robust_action, full_matrices=False)[0]
    return task_basis, robust_basis, rank, len(task_columns) + len(completion_columns)


def dense_bases(
    jacobian: np.ndarray, train_weights: np.ndarray, delta: float
) -> dict[str, np.ndarray]:
    source, rank = span_basis(train_weights)
    complement_projector = np.eye(jacobian.shape[0]) - source @ source.T
    moment = train_weights.T @ train_weights / len(train_weights)
    robust_hessian = jacobian.T @ (moment + delta * complement_projector) @ jacobian
    robust = np.linalg.eigh(robust_hessian)[1][:, ::-1]
    task = np.linalg.svd(jacobian.T @ train_weights.T, full_matrices=False)[0]
    unweighted = np.linalg.svd(jacobian, full_matrices=False)[2].T
    gradients = train_weights @ jacobian
    order = np.argsort(-np.sum(gradients * gradients, axis=0))
    coordinate = np.eye(jacobian.shape[1])[:, order]
    return {
        "complement_robust": robust,
        "task_only": task,
        "unweighted": unweighted,
        "coordinate": coordinate,
        "source_rank": rank,
    }


def run_instance(name: str, *, allow_download: bool = True) -> dict:
    graph_index = list(GSET_HASHES).index(name)
    path = authenticated_gset(name, allow_download=allow_download)
    n_global = parse_gset(path)[0]
    problem, original_weights, vertices = local_problem(
        path, start=(97 * graph_index) % n_global
    )
    theta = torch.cat(
        (
            torch.full((problem.m,), 0.55, dtype=RDTYPE),
            torch.full((problem.n,), 0.32, dtype=RDTYPE),
        )
    )
    jacobian = QAOASensitivity(problem, theta).dense_jacobian().numpy()
    legacy_train, legacy_test = legacy_tasks(problem, original_weights)
    protocols = []
    for protocol in PROTOCOLS:
        train, test = protocol_tasks(problem, original_weights, protocol)
        delta = leave_one_out_radius(train)
        action_operator = QAOASensitivity(problem, theta)
        task_action, robust_action, source_rank, robust_actions = (
            task_and_robust_action_bases(action_operator, train, delta)
        )
        bases = dense_bases(jacobian, train, delta)
        source_basis, _ = span_basis(train)
        source_complement = np.eye(problem.m) - source_basis @ source_basis.T
        empirical_moment = train.T @ train / len(train)
        robust_hessian = (
            jacobian.T
            @ (empirical_moment + delta * source_complement)
            @ jacobian
        )
        robust_scale = max(float(np.trace(robust_hessian)), 1e-30)
        robust_loss_action = float(
            np.trace(robust_hessian)
            - np.trace(robust_action[:, :4].T @ robust_hessian @ robust_action[:, :4])
        )
        robust_loss_dense = float(
            np.trace(robust_hessian)
            - np.trace(
                bases["complement_robust"][:, :4].T
                @ robust_hessian
                @ bases["complement_robust"][:, :4]
            )
        )
        task_loss_action = relative_discarded_energy(
            jacobian, train, task_action[:, :4]
        )
        task_loss_dense = relative_discarded_energy(
            jacobian, train, bases["task_only"][:, :4]
        )
        robust_gap = float(
            np.linalg.norm(
                robust_action[:, :4] @ robust_action[:, :4].T
                - bases["complement_robust"][:, :4]
                @ bases["complement_robust"][:, :4].T,
                ord=2,
            )
        )
        task_gap = float(
            np.linalg.norm(
                task_action[:, :4] @ task_action[:, :4].T
                - bases["task_only"][:, :4] @ bases["task_only"][:, :4].T,
                ord=2,
            )
        )
        metrics = []
        for method in METHODS:
            basis = {
                "complement_robust": robust_action,
                "task_only": task_action,
                "unweighted": bases["unweighted"],
                "coordinate": bases["coordinate"],
            }[method]
            for retained in RANKS:
                metrics.append(
                    {
                        "method": method,
                        "rank": retained,
                        "train_discarded_energy": relative_discarded_energy(
                            jacobian, train, basis[:, :retained]
                        ),
                        "test_discarded_energy": relative_discarded_energy(
                            jacobian, test, basis[:, :retained]
                        ),
                    }
                )
        protocols.append(
            {
                "protocol": protocol,
                "development_tasks": len(train),
                "held_out_tasks": len(test),
                "source_rank": source_rank,
                "leave_one_out_radius": delta,
                "held_out_span_residual": relative_span_residual(train, test),
                "task_only_vjps": source_rank,
                "robust_vjps": robust_actions,
                "direct_task_vjps": len(train),
                "dense_jacobian_vjps": problem.m,
                "rank_four_task_projector_gap": task_gap,
                "rank_four_robust_projector_gap": robust_gap,
                "rank_four_task_loss_gap": abs(task_loss_action - task_loss_dense),
                "rank_four_robust_loss_gap": abs(
                    robust_loss_action - robust_loss_dense
                )
                / robust_scale,
                "metrics": metrics,
            }
        )
    return {
        "graph": name,
        "sha256": sha256(path),
        "global_vertices": n_global,
        "local_vertices": problem.n,
        "local_edges": problem.m,
        "parameter_dimension": problem.dim,
        "global_vertex_ids_one_based": [vertex + 1 for vertex in vertices],
        "legacy_held_out_span_residual": relative_span_residual(
            legacy_train, legacy_test
        ),
        "adjointness_gap": adjointness_gap(
            QAOASensitivity(problem, theta), seed=3000 + graph_index, jvp_mode="autograd"
        ),
        "protocols": protocols,
    }


def summarize(instances: list[dict]) -> dict:
    aggregate = []
    for protocol in PROTOCOLS:
        for method in METHODS:
            for retained in RANKS:
                rows = [
                    metric
                    for instance in instances
                    for record in instance["protocols"]
                    if record["protocol"] == protocol
                    for metric in record["metrics"]
                    if metric["method"] == method and metric["rank"] == retained
                ]
                train = np.asarray([row["train_discarded_energy"] for row in rows])
                test = np.asarray([row["test_discarded_energy"] for row in rows])
                aggregate.append(
                    {
                        "protocol": protocol,
                        "method": method,
                        "rank": retained,
                        "mean_train_discarded_energy": float(train.mean()),
                        "mean_test_discarded_energy": float(test.mean()),
                        "median_test_discarded_energy": float(np.median(test)),
                        "max_test_discarded_energy": float(test.max()),
                    }
                )

    def mean_test(protocol: str, method: str, retained: int = 4) -> float:
        return next(
            row["mean_test_discarded_energy"]
            for row in aggregate
            if row["protocol"] == protocol
            and row["method"] == method
            and row["rank"] == retained
        )

    all_protocols = [record for item in instances for record in item["protocols"]]
    return {
        "instances": instances,
        "aggregate": aggregate,
        "registered_findings": {
            "legacy_split_is_in_span": max(
                item["legacy_held_out_span_residual"] for item in instances
            )
            < 1e-12,
            "revised_full_split_is_out_of_span": min(
                row["held_out_span_residual"]
                for row in all_protocols
                if row["protocol"] == "full_objective"
            )
            > 0.15,
            "contrast_split_is_strongly_out_of_span": min(
                row["held_out_span_residual"]
                for row in all_protocols
                if row["protocol"] == "localized_contrast"
            )
            > 0.80,
            "robust_improves_task_only_on_both_protocol_means": all(
                mean_test(protocol, "complement_robust")
                < mean_test(protocol, "task_only")
                for protocol in PROTOCOLS
            ),
            "robust_beats_unweighted_on_both_protocol_means": all(
                mean_test(protocol, "complement_robust")
                < mean_test(protocol, "unweighted")
                for protocol in PROTOCOLS
            ),
            "action_and_dense_objectives_agree": max(
                max(
                    row["rank_four_task_loss_gap"],
                    row["rank_four_robust_loss_gap"],
                )
                for row in all_protocols
            )
            < 1e-10,
        },
    }


def run_study(*, allow_download: bool = True) -> dict:
    return summarize(
        [run_instance(name, allow_download=allow_download) for name in GSET_HASHES]
    )


def _method_label(method: str) -> str:
    return {
        "complement_robust": "Complement-robust",
        "task_only": "Task-only active subspace",
        "unweighted": "Unweighted observable SVD",
        "coordinate": "Coordinate energy",
    }[method]


def _protocol_label(protocol: str) -> str:
    return {
        "full_objective": "Full objective",
        "localized_contrast": "Localized contrast",
    }[protocol]


def _protocol_record(instance: dict, protocol: str) -> dict:
    return next(row for row in instance["protocols"] if row["protocol"] == protocol)


def benchmark_study(*, allow_download: bool = True, repeats: int = 7) -> dict:
    """Measure complete basis-build paths; results are host-specific evidence."""

    rows = []
    for graph_index, name in enumerate(GSET_HASHES):
        path = authenticated_gset(name, allow_download=allow_download)
        n_global = parse_gset(path)[0]
        problem, original_weights, _ = local_problem(
            path, start=(97 * graph_index) % n_global
        )
        train, _ = protocol_tasks(problem, original_weights, "full_objective")
        delta = leave_one_out_radius(train)
        theta = torch.cat(
            (
                torch.full((problem.m,), 0.55, dtype=RDTYPE),
                torch.full((problem.n,), 0.32, dtype=RDTYPE),
            )
        )

        def task_path() -> None:
            op = QAOASensitivity(problem, theta)
            task_and_robust_action_bases(
                op, train, 0.0, include_completion=False
            )[0]

        def robust_path() -> None:
            op = QAOASensitivity(problem, theta)
            task_and_robust_action_bases(op, train, delta)[1]

        def dense_task_path() -> None:
            jacobian = QAOASensitivity(problem, theta).dense_jacobian().numpy()
            dense_bases(jacobian, train, delta)["task_only"]

        def dense_robust_path() -> None:
            jacobian = QAOASensitivity(problem, theta).dense_jacobian().numpy()
            dense_bases(jacobian, train, delta)["complement_robust"]

        timings = {}
        for label, function in (
            ("task_only_action", task_path),
            ("task_only_dense", dense_task_path),
            ("complement_robust_action", robust_path),
            ("complement_robust_dense", dense_robust_path),
        ):
            function()
            samples = []
            for _ in range(repeats):
                start = time.perf_counter()
                function()
                samples.append(1000.0 * (time.perf_counter() - start))
            timings[label] = {
                "median_ms": float(statistics.median(samples)),
                "q1_ms": float(np.quantile(samples, 0.25)),
                "q3_ms": float(np.quantile(samples, 0.75)),
                "repeats": repeats,
            }
        rows.append({"graph": name, "local_edges": problem.m, "timings": timings})
    return {
        "environment": {
            "platform": platform.platform(),
            "python": sys.version.split()[0],
            "torch": torch.__version__,
            "numpy": np.__version__,
            "threads": torch.get_num_threads(),
            "timer": "time.perf_counter",
        },
        "instances": rows,
    }


def write_figures(study: dict, runtime: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    graphs = [item["graph"] for item in study["instances"]]
    colors = {
        "complement_robust": "#0072B2",
        "task_only": "#D55E00",
        "unweighted": "#009E73",
        "coordinate": "#6B6B6B",
    }

    x = np.arange(len(graphs))
    width = 0.25
    legacy = [item["legacy_held_out_span_residual"] for item in study["instances"]]
    full = [
        _protocol_record(item, "full_objective")["held_out_span_residual"]
        for item in study["instances"]
    ]
    contrast = [
        _protocol_record(item, "localized_contrast")["held_out_span_residual"]
        for item in study["instances"]
    ]
    plt.figure(figsize=(7.2, 4.1))
    plt.bar(x - width, legacy, width, label="Rejected split")
    plt.bar(x, full, width, label="Revised full objective")
    plt.bar(x + width, contrast, width, label="Localized contrast")
    plt.xticks(x, graphs)
    plt.ylabel("Relative distance from development span")
    plt.yscale("log")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    for suffix in ("pdf", "png"):
        plt.savefig(output / f"span_leakage_audit.{suffix}", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35), sharey=True)
    for axis, protocol in zip(axes, PROTOCOLS):
        for method in METHODS:
            rows = [
                row
                for row in study["aggregate"]
                if row["protocol"] == protocol and row["method"] == method
            ]
            axis.plot(
                [row["rank"] for row in rows],
                [row["mean_test_discarded_energy"] for row in rows],
                marker="o",
                linewidth=1.8,
                color=colors[method],
                label=_method_label(method),
            )
        axis.set_title(_protocol_label(protocol))
        axis.set_xlabel("Retained dimension")
        axis.set_xticks(RANKS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean held-out discarded energy")
    axes[1].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"rank_frontier.{suffix}", dpi=220)
    plt.close(fig)

    # Primary diagnostic for the rewritten manuscript.  It places the split
    # audit, rank frontier, paired graph outcomes, and rank-four aggregate in
    # one view so that the proposed completion cannot be judged from a single
    # favorable average.
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 5.8), constrained_layout=True)
    axes[0, 0].bar(x - width, legacy, width, label="rejected split", color="#777777")
    axes[0, 0].bar(x, full, width, label="full objective", color="#56B4E9")
    axes[0, 0].bar(x + width, contrast, width, label="localized contrast", color="#CC79A7")
    axes[0, 0].set_xticks(x, graphs, rotation=20)
    axes[0, 0].set(ylabel="distance from development span", yscale="log")
    axes[0, 0].grid(axis="y", alpha=0.24, which="both")
    axes[0, 0].legend(frameon=False, fontsize=7)

    for method in ("complement_robust", "task_only", "unweighted"):
        rows = [row for row in study["aggregate"] if row["protocol"] == "localized_contrast" and row["method"] == method]
        axes[0, 1].plot(
            [row["rank"] for row in rows],
            [row["mean_test_discarded_energy"] for row in rows],
            "o-", color=colors[method], label=_method_label(method),
        )
    axes[0, 1].set(xlabel="retained dimension", ylabel="mean contrast discarded energy")
    axes[0, 1].set_xticks(RANKS)
    axes[0, 1].grid(alpha=0.24)
    axes[0, 1].legend(frameon=False, fontsize=7)

    task_values, robust_values = [], []
    for item in study["instances"]:
        record = _protocol_record(item, "localized_contrast")
        by_method = {row["method"]: row["test_discarded_energy"] for row in record["metrics"] if row["rank"] == 4}
        task_values.append(by_method["task_only"])
        robust_values.append(by_method["complement_robust"])
    axes[1, 0].plot([0, 1], [0, 1], "--", color="#555555", lw=1, label="equal loss")
    axes[1, 0].scatter(task_values, robust_values, color=colors["complement_robust"])
    for name, left, right in zip(graphs, task_values, robust_values):
        axes[1, 0].annotate(name, (left, right), xytext=(3, 2), textcoords="offset points", fontsize=6.5)
    axes[1, 0].set(xlabel="task-only rank-four loss", ylabel="robust rank-four loss", xlim=(0, 1), ylim=(0, 1))
    axes[1, 0].grid(alpha=0.24)
    axes[1, 0].legend(frameon=False, fontsize=7)

    methods = ("complement_robust", "task_only", "unweighted", "coordinate")
    labels = ("robust", "task-only", "unweighted", "coordinate")
    xpos = np.arange(len(methods))
    for offset, protocol, hatch in ((-0.18, "full_objective", ""), (0.18, "localized_contrast", "//")):
        vals = [next(row["mean_test_discarded_energy"] for row in study["aggregate"] if row["protocol"] == protocol and row["method"] == method and row["rank"] == 4) for method in methods]
        axes[1, 1].bar(xpos + offset, vals, 0.36, color=[colors[m] for m in methods], alpha=0.86, hatch=hatch,
                       edgecolor="white", label="full objective" if protocol == "full_objective" else "localized contrast")
    axes[1, 1].set_xticks(xpos, labels, rotation=20, ha="right")
    axes[1, 1].set_ylabel("mean rank-four held-out loss")
    axes[1, 1].grid(axis="y", alpha=0.24)
    axes[1, 1].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"robust_transfer_audit.{suffix}", dpi=220)
    plt.close(fig)

    width = 0.25
    values = {method: [] for method in METHODS[:3]}
    for item in study["instances"]:
        record = _protocol_record(item, "localized_contrast")
        for method in values:
            values[method].append(
                next(
                    row["test_discarded_energy"]
                    for row in record["metrics"]
                    if row["method"] == method and row["rank"] == 4
                )
            )
    plt.figure(figsize=(7.2, 4.1))
    for index, method in enumerate(values):
        plt.bar(
            x + (index - 1) * width,
            values[method],
            width,
            label=_method_label(method),
            color=colors[method],
        )
    plt.xticks(x, graphs)
    plt.ylabel("Rank-four held-out discarded energy")
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    for suffix in ("pdf", "png"):
        plt.savefig(output / f"contrast_per_graph.{suffix}", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.1))
    edge_counts = [item["local_edges"] for item in study["instances"]]
    order = np.argsort(edge_counts)
    for label, marker, linestyle in (
        ("task_only_action", "o", "-"),
        ("task_only_dense", "o", "--"),
        ("complement_robust_action", "s", "-"),
        ("complement_robust_dense", "s", "--"),
    ):
        medians = [item["timings"][label]["median_ms"] for item in runtime["instances"]]
        plt.plot(
            np.asarray(edge_counts)[order],
            np.asarray(medians)[order],
            marker=marker,
            linestyle=linestyle,
            linewidth=1.8,
            label={
                "task_only_action": "Task-only, action",
                "task_only_dense": "Task-only, dense",
                "complement_robust_action": "Complement-robust, action",
                "complement_robust_dense": "Complement-robust, dense",
            }[label],
        )
    plt.xlabel("Edges in ten-vertex neighborhood")
    plt.ylabel("Median basis-build time (ms)")
    plt.yscale("log")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False, fontsize=8)
    plt.tight_layout()
    for suffix in ("pdf", "png"):
        plt.savefig(output / f"runtime_scaling.{suffix}", dpi=220)
    plt.close()

    plt.figure(figsize=(7.2, 4.1))
    source_rank = [
        _protocol_record(item, "full_objective")["task_only_vjps"]
        for item in study["instances"]
    ]
    dense = [item["local_edges"] for item in study["instances"]]
    plt.plot(graphs, source_rank, marker="o", linewidth=1.8, label="Task-only VJPs")
    plt.plot(graphs, dense, marker="s", linewidth=1.8, label="Robust / dense-row VJPs")
    plt.ylabel("Transpose-Jacobian products")
    plt.xlabel("Authenticated graph")
    plt.grid(alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    for suffix in ("pdf", "png"):
        plt.savefig(output / f"operator_actions.{suffix}", dpi=220)
    plt.close()

    fig, axes = plt.subplots(1, 3, figsize=(10.0, 3.0), constrained_layout=True)
    axes[0].plot(graphs, source_rank, "o-", label="task-only", color=colors["task_only"])
    axes[0].plot(graphs, dense, "s-", label="robust", color=colors["complement_robust"])
    axes[0].set(xlabel="authenticated graph", ylabel="transpose-Jacobian products")
    axes[0].tick_params(axis="x", rotation=22)
    axes[0].grid(alpha=0.24)
    axes[0].legend(frameon=False, fontsize=7)
    task_ratio, robust_ratio = [], []
    for item in runtime["instances"]:
        timing = item["timings"]
        task_ratio.append(timing["task_only_action"]["median_ms"] / timing["task_only_dense"]["median_ms"])
        robust_ratio.append(timing["complement_robust_action"]["median_ms"] / timing["complement_robust_dense"]["median_ms"])
    axes[1].axhline(1.0, color="#555555", ls="--", lw=1)
    axes[1].plot(graphs, task_ratio, "o-", color=colors["task_only"], label="task-only")
    axes[1].plot(graphs, robust_ratio, "s-", color=colors["complement_robust"], label="robust")
    axes[1].set(xlabel="authenticated graph", ylabel="action time / dense time")
    axes[1].tick_params(axis="x", rotation=22)
    axes[1].grid(alpha=0.24)
    axes[1].legend(frameon=False, fontsize=7)
    task_gap, robust_gap = [], []
    for item in study["instances"]:
        record = _protocol_record(item, "full_objective")
        task_gap.append(max(record["rank_four_task_loss_gap"], 1e-18))
        robust_gap.append(max(record["rank_four_robust_loss_gap"], 1e-18))
    axes[2].semilogy(graphs, task_gap, "o-", color=colors["task_only"], label="task-only")
    axes[2].semilogy(graphs, robust_gap, "s-", color=colors["complement_robust"], label="robust")
    axes[2].set(xlabel="authenticated graph", ylabel="action--dense loss gap")
    axes[2].tick_params(axis="x", rotation=22)
    axes[2].grid(alpha=0.24, which="both")
    axes[2].legend(frameon=False, fontsize=7)
    for suffix in ("pdf", "png"):
        fig.savefig(output / f"oracle_cost_audit.{suffix}", dpi=220)
    plt.close(fig)


def write_tables(study: dict, runtime: dict, output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "rank_four_aggregate.tex").open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{llrr}\\toprule\n")
        stream.write("Protocol & Method & train & held-out \\\\\n\\midrule\n")
        for protocol in PROTOCOLS:
            for method in METHODS:
                row = next(
                    value
                    for value in study["aggregate"]
                    if value["protocol"] == protocol
                    and value["method"] == method
                    and value["rank"] == 4
                )
                stream.write(
                    f"{_protocol_label(protocol)} & {_method_label(method)} & "
                    f"{row['mean_train_discarded_energy']:.4f} & "
                    f"{row['mean_test_discarded_energy']:.4f} \\\\\n"
                )
            if protocol != PROTOCOLS[-1]:
                stream.write("\\midrule\n")
        stream.write("\\bottomrule\\end{tabular}\n")

    with (output / "contrast_per_graph.tex").open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        stream.write("Graph & span residual & $\\widehat\\delta$ & robust & task-only & unweighted \\\\\n\\midrule\n")
        for item in study["instances"]:
            record = _protocol_record(item, "localized_contrast")
            metric = {
                row["method"]: row["test_discarded_energy"]
                for row in record["metrics"]
                if row["rank"] == 4
            }
            stream.write(
                f"{item['graph']} & {record['held_out_span_residual']:.3f} & "
                f"{record['leave_one_out_radius']:.3f} & "
                f"{metric['complement_robust']:.3f} & {metric['task_only']:.3f} & "
                f"{metric['unweighted']:.3f} \\\\\n"
            )
        stream.write("\\bottomrule\\end{tabular}\n")

    with (output / "resource_audit.tex").open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{lrrrrr}\\toprule\n")
        stream.write("Graph & $m$ & $d$ & $\\rho$ & task VJP & robust VJP \\\\\n\\midrule\n")
        for item in study["instances"]:
            record = _protocol_record(item, "full_objective")
            stream.write(
                f"{item['graph']} & {item['local_edges']} & {item['parameter_dimension']} & "
                f"{record['source_rank']} & {record['task_only_vjps']} & "
                f"{record['robust_vjps']} \\\\\n"
            )
        stream.write("\\bottomrule\\end{tabular}\n")

    with (output / "runtime.tex").open("w", encoding="utf-8") as stream:
        stream.write("\\begin{tabular}{lrrrr}\\toprule\n")
        stream.write(
            "Graph & task action & task dense & robust action & robust dense \\\\\n"
            "\\midrule\n"
        )
        for item in runtime["instances"]:
            stream.write(
                f"{item['graph']} & "
                f"{item['timings']['task_only_action']['median_ms']:.2f} & "
                f"{item['timings']['task_only_dense']['median_ms']:.2f} & "
                f"{item['timings']['complement_robust_action']['median_ms']:.2f} & "
                f"{item['timings']['complement_robust_dense']['median_ms']:.2f} \\\\\n"
            )
        stream.write("\\bottomrule\\end{tabular}\n")


def write_outputs(
    study: dict, runtime: dict, results: Path, figures: Path, tables: Path
) -> None:
    results.mkdir(parents=True, exist_ok=True)
    (results / "locked_results.json").write_text(
        json.dumps(study, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (results / "runtime_results.json").write_text(
        json.dumps(runtime, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_figures(study, runtime, figures)
    write_tables(study, runtime, tables)
    evidence = [
        results / "locked_results.json",
        results / "runtime_results.json",
        *sorted(figures.glob("*.pdf")),
        *sorted(figures.glob("*.png")),
        *sorted(tables.glob("*.tex")),
    ]
    root = Path.cwd().resolve()
    manifest = {
        "schema": 2,
        "dataset_source": GSET_ROOT,
        "dataset_sha256": GSET_HASHES,
        "evidence_files": {
            str(path.resolve().relative_to(root)): sha256(path) for path in evidence
        },
    }
    (results / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _compare(actual, expected, path: str = "root") -> None:
    if isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise AssertionError(f"mapping mismatch at {path}")
        for key in expected:
            _compare(actual[key], expected[key], f"{path}.{key}")
        return
    if isinstance(expected, list):
        if not isinstance(actual, list) or len(actual) != len(expected):
            raise AssertionError(f"list mismatch at {path}")
        for index, (left, right) in enumerate(zip(actual, expected)):
            _compare(left, right, f"{path}[{index}]")
        return
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or not math.isclose(
            float(actual), expected, rel_tol=2e-9, abs_tol=2e-11
        ):
            raise AssertionError(f"numeric mismatch at {path}: {actual} != {expected}")
        return
    if actual != expected:
        raise AssertionError(f"value mismatch at {path}: {actual!r} != {expected!r}")


def verify_outputs(study: dict, results: Path) -> None:
    locked = json.loads((results / "locked_results.json").read_text(encoding="utf-8"))
    _compare(study, locked)
    if not all(study["registered_findings"].values()):
        raise AssertionError(f"registered finding failed: {study['registered_findings']}")
    manifest = json.loads((results / "manifest.json").read_text(encoding="utf-8"))
    for relative, expected in manifest["evidence_files"].items():
        path = Path(relative)
        if not path.is_file() or sha256(path) != expected:
            raise AssertionError(f"evidence hash mismatch: {relative}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--offline", action="store_true")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--summary-only", action="store_true")
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--verify", action="store_true")
    mode.add_argument("--benchmark", action="store_true")
    mode.add_argument("--download-only", action="store_true")
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--results", type=Path, default=Path("code/results"))
    parser.add_argument(
        "--figures", type=Path, default=Path("code/manuscript_assets/figures")
    )
    parser.add_argument("--tables", type=Path, default=Path("code/results/tables"))
    args = parser.parse_args(argv)
    if args.download_only:
        paths = [
            authenticated_gset(name, allow_download=not args.offline)
            for name in GSET_HASHES
        ]
        print(json.dumps({path.name: sha256(path) for path in paths}, indent=2))
        return
    if args.benchmark:
        print(
            json.dumps(
                benchmark_study(
                    allow_download=not args.offline, repeats=args.repeats
                ),
                indent=2,
            )
        )
        return
    study = run_study(allow_download=not args.offline)
    if args.summary_only:
        print(
            json.dumps(
                {
                    "registered_findings": study["registered_findings"],
                    "aggregate_rank_four": [
                        row for row in study["aggregate"] if row["rank"] == 4
                    ],
                },
                indent=2,
            )
        )
    elif args.write:
        runtime = benchmark_study(
            allow_download=not args.offline, repeats=args.repeats
        )
        write_outputs(study, runtime, args.results, args.figures, args.tables)
        print(json.dumps(study["registered_findings"], sort_keys=True))
    else:
        verify_outputs(study, args.results)
        print("authenticated numerical replay: PASS")


if __name__ == "__main__":
    main()
