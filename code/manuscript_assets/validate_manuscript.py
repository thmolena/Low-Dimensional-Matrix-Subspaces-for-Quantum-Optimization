#!/usr/bin/env python3
"""Validate the manuscript, numerical assets, provenance, and public layout."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
CODE = ROOT / "code"
TEX = ROOT / "main.tex"
PDF = ROOT / "main.pdf"
BIB = CODE / "manuscript_assets/references.bib"


def fail(message: str) -> None:
    raise SystemExit(f"manuscript contract: FAIL\n- {message}")


def page_count() -> int:
    executable = shutil.which("pdfinfo")
    if executable is None:
        fail("pdfinfo is required")
    output = subprocess.run(
        [executable, str(PDF)], check=True, capture_output=True, text=True
    ).stdout
    match = re.search(r"(?m)^Pages:\s+(\d+)$", output)
    if match is None:
        fail("main.pdf page count is unreadable")
    return int(match.group(1))


def manuscript_contract(tex: str, bib: str) -> None:
    expected_class = r"\documentclass[10pt,letterpaper,twoside]{article}"
    if not tex.startswith(expected_class):
        fail("main.tex does not use the one-column preprint configuration")
    if r"\usepackage{code/manuscript_assets/researchpreprint}" not in tex:
        fail("main.tex does not load the shared preprint style")
    if page_count() not in range(5, 16):
        fail("main.pdf is not 5--15 pages")
    if tex.count(r"\begin{table") != 2:
        fail("main.tex must contain exactly two scientific tables")
    figures = re.findall(r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex)
    if len(figures) != 2 or len(figures) != len(set(figures)):
        fail("main.tex must contain two distinct scientific figures")
    for relative in figures:
        path = ROOT / relative
        if path.suffix.lower() != ".pdf" or not path.is_file():
            fail(f"missing PDF figure: {relative}")
    theorem_count = len(
        re.findall(r"\\begin\{(?:theorem|proposition|lemma|corollary)\}", tex)
    )
    if theorem_count != 3:
        fail(f"expected three theorem-like results, found {theorem_count}")
    for phrase in (
        "rank-counterfactual operator loss",
        "matched synthetic control",
        "reader guide",
        "claim ledger",
    ):
        if phrase in tex.lower():
            fail(f"rejected legacy manuscript content remains: {phrase}")

    entries = re.findall(r"(?m)^@\w+\s*\{\s*([^,]+),", bib)
    if len(entries) < 12 or len(entries) != len(set(entries)):
        fail("bibliography does not contain at least 12 unique records")
    cited = set()
    for group in re.findall(r"\\cite\w*\s*\{([^}]*)\}", tex):
        cited.update(key.strip() for key in group.split(",") if key.strip())
    missing = cited - set(entries)
    if missing:
        fail(f"citation keys are absent from the bibliography: {sorted(missing)}")
    if len(cited) < 7:
        fail(f"main.tex cites only {len(cited)} unique sources")


def numerical_contract(tex: str) -> None:
    expected_figures = {
        "robust_transfer_audit",
        "oracle_cost_audit",
    }
    for stem in expected_figures:
        for suffix in ("pdf", "png"):
            if not (CODE / f"manuscript_assets/figures/{stem}.{suffix}").is_file():
                fail(f"missing figure asset: {stem}.{suffix}")
    expected_tables = {
        "rank_four_aggregate", "contrast_per_graph", "resource_audit", "runtime"
    }
    for stem in expected_tables:
        if not (CODE / f"results/tables/{stem}.tex").is_file():
            fail(f"missing numerical table: {stem}.tex")
    locked = json.loads((CODE / "results/locked_results.json").read_text())
    if len(locked.get("instances", [])) != 8:
        fail("locked result does not contain eight graph instances")
    findings = locked.get("registered_findings", {})
    if len(findings) != 6 or not all(findings.values()):
        fail(f"registered findings did not pass: {findings}")
    for token in expected_figures:
        if token not in tex:
            fail(f"manuscript does not use registered figure {token}")

    source = (CODE / "src/hqml_observable_subspaces/authentic_study.py").read_text()
    forbidden_plot = re.compile(
        r"\.(?:scatter|hist|imshow|contour|contourf|pcolormesh|pie|boxplot|violinplot)\s*\("
    )
    if forbidden_plot.search(source):
        fail("a forbidden plot primitive appears in the scientific generator")
    if source.count(".plot(") < 2 or source.count(".bar(") < 2:
        fail("the scientific generator must use both line and bar plots")


def provenance_contract() -> None:
    source = (CODE / "src/hqml_observable_subspaces/authentic_study.py").read_text()
    checksums = (CODE / "data/checksums.sha256").read_text().splitlines()
    if len([line for line in checksums if line.strip() and not line.startswith("#")]) != 8:
        fail("data checksum registry does not contain eight entries")
    for marker in (
        "https://web.stanford.edu/~yyye/yyye/Gset",
        "RSQAOA_GSET_CACHE",
        "urllib.request.urlopen",
        '"G1"', '"G6"', '"G11"', '"G14"', '"G18"', '"G22"', '"G32"', '"G43"',
    ):
        if marker not in source:
            fail(f"data provenance marker is absent: {marker}")


def language_contract() -> None:
    public = [ROOT / "main.tex", ROOT / "README.md", ROOT / "index.html"]
    forbidden = re.compile(
        r"(?i)(chatgpt|openai|gpt[- ]?(?:2|3|4|5)|acknowledg(?:e)?ments|"
        r"author contributions)"
    )
    offenders = []
    for path in public:
        if forbidden.search(path.read_text(encoding="utf-8")):
            offenders.append(path.name)
    if offenders:
        fail(f"forbidden public wording appears in {offenders}")


def layout_and_hygiene_contract() -> None:
    allowed_root = {
        ".git", ".gitignore", "LICENSE", "README.md", "index.html",
        "main.tex", "main.pdf", "code",
    }
    actual = {path.name for path in ROOT.iterdir()}
    if actual != allowed_root:
        fail(f"root layout differs: {sorted(actual ^ allowed_root)}")
    required_code = {
        "README.md", "LICENSE", "pyproject.toml", "src", "tests", "scripts",
        "configs", "data", "results", "manuscript_assets",
    }
    absent = required_code - {path.name for path in CODE.iterdir()}
    if absent:
        fail(f"required code entries are absent: {sorted(absent)}")
    if len(list(ROOT.rglob("main.tex"))) != 1 or len(list(ROOT.rglob("main.pdf"))) != 1:
        fail("duplicate manuscript source or PDF exists")

    forbidden_dirs = {"__pycache__", ".pytest_cache", ".mypy_cache", ".ruff_cache", "build", "dist"}
    artifacts = []
    for path in ROOT.rglob("*"):
        if path.is_symlink():
            artifacts.append(path)
        elif path.is_dir() and (path.name in forbidden_dirs or path.name.endswith(".egg-info")):
            artifacts.append(path)
        elif path.is_file() and (
            path.suffix in {".pyc", ".pyo", ".aux", ".log", ".blg"}
            or path.name == ".DS_Store"
        ):
            artifacts.append(path)
    if artifacts:
        fail(f"generated artifacts remain: {[str(p.relative_to(ROOT)) for p in artifacts[:12]]}")


def main() -> int:
    for path in (TEX, PDF, BIB, CODE / "pyproject.toml"):
        if not path.is_file():
            fail(f"required file is absent: {path}")
    tex = TEX.read_text(encoding="utf-8")
    manuscript_contract(tex, BIB.read_text(encoding="utf-8"))
    numerical_contract(tex)
    provenance_contract()
    language_contract()
    layout_and_hygiene_contract()
    print("manuscript contract: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
