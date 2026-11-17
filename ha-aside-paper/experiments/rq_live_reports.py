"""Atomic provisional snapshots, source-matched comparisons and scientific plots."""

import csv
import html
import importlib.metadata
import json
from pathlib import Path
import uuid

from .common import digest, file_hash, write_json
from .rq_live_io import selected_cases, committed_rows, validate_records, complete_groups
from .rq_one_week import PROFILE
from .rq_reporting import export_reports

STATIC = ["role_authority", "task_controls", "iheval"]
DISPLAY = {"ha_aside": "HA-ASIDE", "aside_orig": "ASIDE", "vanilla": "Vanilla", "ise": "ISE-2",
           "ise_4role": "ISE-4", "ha_no_user_rotation": "HA (U=0)"}


def csv_file(path, records, columns):
    with Path(path).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def filter_view(records, complete, names):
    common = {suite: set.intersection(*(complete[(name, suite)] for name in names)) for suite in STATIC}
    return {(name, suite): [r for r in records[(name, suite)].values() if r["group_id"] in common[suite]]
            for name in names for suite in STATIC}, {suite: len(groups) for suite, groups in common.items()}


def chart_snapshot(folder, names, coverage, *, test_fixture=False):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    import numpy as np
    plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none", "font.size": 10})
    folder = Path(folder)
    graphs = folder / "graphs"
    graphs.mkdir()
    metrics_path = folder / "common_all/metrics.json"
    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))["metrics"]
    outputs = []

    def save(fig, name):
        prefix = "TEST FIXTURE - NOT GPU RESULTS. " if test_fixture else ""
        fig.text(.5, .01, prefix + "PROVISIONAL - common source groups; inspect coverage.csv; missing scores are not zero", ha="center", fontsize=8)
        fig.tight_layout(rect=(0, .04, 1, .89 if fig._suptitle is not None else 1))
        for extension in ["png", "svg"]:
            path = graphs / f"{name}.{extension}"
            fig.savefig(path, dpi=180)
            outputs.append(path.relative_to(folder).as_posix())
        plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(15, 4))
    for ax, suite in zip(axes, [*STATIC, "role_probe"]):
        rows = [r for r in coverage if r["suite"] == suite]
        rates = [r["completed_records"] / r["planned_records"] if r["planned_records"] else 0 for r in rows]
        ax.bar(range(len(rows)), rates, color=plt.cm.tab10.colors[:len(rows)])
        ax.set_xticks(range(len(rows)), [DISPLAY.get(r["model"], r["model"]) for r in rows], rotation=45, ha="right")
        ax.set_ylim(0, 1.15)
        ax.set_title(suite)
        for i, r in enumerate(rows):
            ax.text(i, rates[i] + .025, f"{r['completed_records']}/{r['planned_records']}", ha="center", fontsize=8)
    axes[0].set_ylabel("Completed selected records / planned")
    save(fig, "coverage")

    def bars(suite, metric, title, filename, *, source=None, owners=None):
        selected_names = owners or names
        observed = [r for r in (metrics if source is None else source) if r["suite"] == suite and r["metric"] == metric and r["value"] is not None]
        role_order = {"system": 0, "user": 1, "data": 2, "assistant": 3}
        conditions = sorted({r["condition"] for r in observed}, key=lambda c: (role_order.get(c, 4), c))
        fig, ax = plt.subplots(figsize=(max(8, len(conditions) * 1.3), 4.8))
        if not conditions:
            ax.text(.5, .5, "Pending: no complete source groups shared by all selected models", ha="center", transform=ax.transAxes)
        else:
            lookup = {(r["model"], r["condition"]): r for r in observed}
            width = .8 / len(selected_names)
            for i, model in enumerate(selected_names):
                values = [lookup.get((model, c), {}).get("value", np.nan) for c in conditions]
                ax.bar(np.arange(len(conditions)) + (i - (len(selected_names) - 1) / 2) * width, values, width, label=DISPLAY.get(model, model))
            labels = [f"{c.upper() if c in role_order else c}\nn={max(r['n_groups'] for r in observed if r['condition']==c)} groups" for c in conditions]
            ax.set_xticks(range(len(conditions)), labels, rotation=25, ha="right")
            ax.legend(loc="lower center", bbox_to_anchor=(.5, 1.02), ncol=min(4, len(selected_names)))
        ax.set_ylim(0, 1.05)
        fig.suptitle(title, y=.985, fontsize=12)
        ax.set_ylabel({"authority_correct": "Authority correctness", "utility": "Task utility"}.get(metric, metric))
        save(fig, filename)

    bars("role_authority", "authority_correct", "Authority correctness - common complete clauses", "role_authority")
    bars("task_controls", "utility", "Clean utility - common complete source groups", "task_utility")
    if "ha_aside" in names:
        for baseline in names:
            path = folder / ("pair_ha_aside_" + baseline) / "metrics.json"
            if baseline != "ha_aside" and path.exists():
                source = json.loads(path.read_text(encoding="utf-8"))["metrics"]
                bars("role_authority", "authority_correct", "HA-ASIDE vs " + DISPLAY.get(baseline, baseline) + " - pairwise matched clauses",
                     "pair_ha_aside_" + baseline + "_authority", source=source, owners=["ha_aside", baseline])

    fig, ax = plt.subplots(figsize=(8, 4.8))
    probes = [r for r in metrics if r["suite"] == "role_probe" and r["metric"] == "macro_f1" and str(r["condition"]).isdigit()]
    for model in names:
        line = sorted([r for r in probes if r["model"] == model], key=lambda r: int(r["condition"]))
        if line:
            ax.plot([int(r["condition"]) for r in line], [r["value"] for r in line], label=DISPLAY.get(model, model))
    if probes:
        ax.legend()
    else:
        ax.text(.5, .5, "Pending: all selected models must finish their matched probes", ha="center", transform=ax.transAxes)
    ax.set(xlabel="Layer", ylabel="Macro-F1", ylim=(0, 1.05), title="Four-role probe - identical held-out clauses")
    save(fig, "probe_macro_f1")

    summary = json.loads((folder / "common_all/IHEval_native_summary.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 4.8))
    settings = ["reference", "aligned", "conflict"]
    plotted = False
    for i, model in enumerate(names):
        points = [summary.get(model, {}).get(setting, {}) for setting in settings]
        values = [p.get("overall", np.nan) if p.get("n_tasks") == 9 else np.nan for p in points]
        if any(np.isfinite(v) for v in values):
            plotted = True
        width = .8 / len(names)
        ax.bar(np.arange(3) + (i - (len(names) - 1) / 2) * width, values, width, label=DISPLAY.get(model, model))
    if plotted:
        ax.legend()
    else:
        ax.text(.5, .5, "Pending: source scores for all nine tasks are required", ha="center", transform=ax.transAxes)
    ax.set_xticks(range(3), settings)
    ax.set(ylim=(0, 1.05), ylabel="Source macro score", title="IHEval subset - common groups, all 9 tasks")
    save(fig, "iheval_settings")
    return outputs


def export_live(run_dir, *, with_ci=False, context=None):
    run_dir = Path(run_dir).resolve()
    parent_path = run_dir / "manifest.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    config = parent["config"]
    if config.get("benchmark_profile") != PROFILE:
        raise ValueError("Live export requires the frozen one-week profile")
    selection, cases = context if context is not None else selected_cases(config)
    if config["subset_fingerprint"] != selection["fingerprint"]:
        raise ValueError("Run/subset fingerprint mismatch")
    names = config["methods"] + [a["name"] for a in config.get("ablations", [])]
    records, complete, inventory, coverage, probes = {}, {}, {}, [], {}
    for name in names:
        for suite in STATIC:
            values, info = committed_rows(run_dir / name / f"{suite}.jsonl")
            indexed = validate_records(values, cases[suite])
            records[(name, suite)] = indexed
            complete[(name, suite)] = complete_groups(indexed, cases[suite])
            inventory[f"{name}/{suite}.jsonl"] = info
            skipped = name not in config["methods"] and suite == "task_controls"
            planned = 0 if skipped else len(cases[suite])
            coverage.append({"model": name, "suite": suite, "completed_records": len(indexed),
                             "planned_records": planned, "complete_source_groups": len(complete[(name, suite)]),
                             "status": "not_requested" if skipped else "complete" if len(indexed) == planned else "in_progress" if indexed else "not_started"})
        probe_path = run_dir / name / "role_probe.json"
        if probe_path.exists():
            value = json.loads(probe_path.read_text(encoding="utf-8"))
            if (value["n_examples"] != selection["audit"]["counts"]["probe_examples"] or
                    set(value["source_groups"]) != {c["group_id"] for c in cases["role_authority"].values()}):
                raise ValueError("Probe output has incomplete source coverage")
            probes[name] = value
            inventory[f"{name}/role_probe.json"] = {"sha256": file_hash(probe_path)}
        coverage.append({"model": name, "suite": "role_probe", "completed_records": value["n_examples"] if name in probes else 0,
                         "planned_records": selection["audit"]["counts"]["probe_examples"],
                         "complete_source_groups": probes[name]["n_groups"] if name in probes else 0,
                         "status": "complete" if name in probes else "pending_or_cached_features"})
    identity = {"format": "weekly_live_snapshot_v1", "parent_manifest_sha256": file_hash(parent_path),
                "subset_fingerprint": selection["fingerprint"], "committed_inputs": inventory,
                "reporter_sha256": file_hash(Path(__file__)), "with_ci": with_ci,
                "source_reporter_sha256": file_hash(Path(__file__).with_name("rq_reporting.py")),
                "matplotlib": importlib.metadata.version("matplotlib")}
    sid = digest(identity)
    live = run_dir / "live"
    # Keep directory components shorter for Windows while storing the full
    # SHA-256 identity in latest.json/snapshot.json and checking collisions.
    directory_id = sid[:32]
    final = live / "snapshots" / directory_id
    if not final.exists():
        building = live / "snapshots" / (".building-" + uuid.uuid4().hex)
        building.mkdir(parents=True)
        csv_file(building / "coverage.csv", coverage, ["model", "suite", "completed_records", "planned_records", "complete_source_groups", "status"])
        per_case, diagnostics = [], []
        helper_ids = set(selection["audit"]["iheval_reference_helper_ids"])
        for (name, suite), indexed in records.items():
            for rid, row in indexed.items():
                meta, score = row["metadata"], row["scores"]
                calls = [row, *row.get("interventions", {}).values()]
                per_case.append({"model": name, "suite": suite, "case_id": rid, "group_id": row["group_id"],
                                 "condition": meta.get("condition", meta.get("task", meta.get("task_family"))),
                                 "setting": meta.get("setting"), "complete_source_group": row["group_id"] in complete[(name, suite)],
                                 "calibration_helper": rid in helper_ids, "case_sha256": row["case_sha256"],
                                 "generation_calls": len(calls), "seconds": sum(c["seconds"] for c in calls),
                                 "generated_tokens": sum(c["generated_tokens"] for c in calls),
                                 **{k: score.get(k) for k in ["authority_correct", "executed", "utility", "strict", "loose", "task_correct"]}})
                for role, diagnostic in row.get("interventions", {}).items():
                    diagnostics.append({"model": name, "case_id": rid, "group_id": row["group_id"],
                                        "encoding": "DATA->" + role, "executed": diagnostic["scores"]["executed"],
                                        "execution_delta": float(diagnostic["scores"]["executed"]) - float(score["executed"])})
        csv_file(building / "observations.csv", per_case, ["model", "suite", "case_id", "group_id", "condition", "setting",
                 "complete_source_group", "calibration_helper", "case_sha256", "generation_calls", "seconds", "generated_tokens",
                 "authority_correct", "executed", "utility", "strict", "loose", "task_correct"])
        csv_file(building / "interventions.csv", diagnostics, ["model", "case_id", "group_id", "encoding", "executed", "execution_delta"])
        available = {(n, s): [r for r in records[(n, s)].values() if r["group_id"] in complete[(n, s)]] for n in names for s in STATIC}
        matched, common_counts = filter_view(records, complete, names)
        common_probe = {}
        if set(probes) == set(names):
            signatures = {digest([(p["group_id"], p["gold"]) for p in probes[n]["layers"][0]["test_predictions"]]) for n in names}
            if len(signatures) != 1:
                raise ValueError("Probe test cohorts differ across methods")
            common_probe = probes
        views = [("available", names, available, probes), ("common_all", names, matched, common_probe)]
        pairwise = []
        if "ha_aside" in names:
            for baseline in names:
                if baseline == "ha_aside":
                    continue
                owners = ["ha_aside", baseline]
                observed, _ = filter_view(records, complete, owners)
                pair_probes = {n: probes[n] for n in owners} if all(n in probes for n in owners) else {}
                views.append(("pair_ha_aside_" + baseline, owners, observed, pair_probes))
        for label, owners, observations, probe_values in views:
            subconfig = {**config, "methods": [n for n in config["methods"] if n in owners],
                         "ablations": [a for a in config.get("ablations", []) if a["name"] in owners]}
            destination = building / label
            export_reports(run_dir, subconfig, destination=destination, observations=observations, probes=probe_values,
                           provisional=True, source_cases=cases["iheval"], live_ci=with_ci)
            if label.startswith("pair_"):
                pairwise.extend(json.loads((destination / "metrics.json").read_text(encoding="utf-8"))["paired_differences"])
        csv_file(building / "pairwise.csv", pairwise, ["rq", "suite", "condition", "metric", "comparison", "value", "n_groups", "n_observations", "status"])
        graph_paths = chart_snapshot(building, names, coverage, test_fixture=bool(parent.get("test_fixture")))
        note = {"identity": identity, "status": "provisional", "common_complete_groups_all_methods": common_counts,
                "coverage": coverage, "graphs": graph_paths,
                "interpretation": "available tables are per-model observations and not a fair ranking; common_all uses identical complete source groups across all selected models; pair_* uses pairwise common groups. Missing metrics are pending, never zero. Partial prefixes are provisional, not full subset results."}
        write_json(building / "snapshot.json", note)
        page = "<html><head><meta charset='utf-8'><title>Weekly benchmark snapshot</title></head><body><h1>PROVISIONAL benchmark comparison</h1>"
        page += "<p>Use common_all tables for matched comparisons; available tables have different coverage. See coverage.csv and snapshot.json. No missing score is zero.</p>"
        page += "<p><a href='coverage.csv'>Coverage CSV</a> | <a href='common_all/RQ1.csv'>Matched RQ1</a> | <a href='common_all/RQ2.csv'>Matched RQ2</a> | <a href='pairwise.csv'>Pairwise differences</a></p>"
        page += "".join(f"<img style='max-width:100%' src='{html.escape(p)}'>" for p in graph_paths if p.endswith(".svg")) + "</body></html>"
        (building / "index.html").write_text(page, encoding="utf-8")
        write_json(building / "files.sha256.json", {p.relative_to(building).as_posix(): file_hash(p) for p in sorted(building.rglob("*")) if p.is_file()})
        try:
            building.rename(final)
        except OSError:
            # Another read-only exporter may have completed the same snapshot.
            if not final.is_dir():
                raise
    if json.loads((final / "snapshot.json").read_text(encoding="utf-8"))["identity"] != identity:
        raise ValueError("Snapshot directory identity collision or modification")
    expected_files = json.loads((final / "files.sha256.json").read_text(encoding="utf-8"))
    if any(file_hash(final / relative) != expected for relative, expected in expected_files.items()):
        raise ValueError("An immutable live snapshot was modified")
    pointer = {"snapshot": final.relative_to(live).as_posix(), "snapshot_id": sid, "status": "provisional",
               "completed_records": sum(len(r) for r in records.values()), "completed_probes": len(probes)}
    if (live / "latest.json").exists():
        old = json.loads((live / "latest.json").read_text(encoding="utf-8"))
        if (old.get("completed_records", 0), old.get("completed_probes", 0)) > (pointer["completed_records"], pointer["completed_probes"]):
            return final
    pointer_temp = live / ("latest." + uuid.uuid4().hex + ".tmp")
    pointer_temp.write_text(json.dumps(pointer, indent=2), encoding="utf-8")
    pointer_temp.replace(live / "latest.json")
    temp_page = live / ("index." + uuid.uuid4().hex + ".tmp")
    temp_page.write_text(f"<html><meta charset='utf-8'><meta http-equiv='refresh' content='0;url=snapshots/{directory_id}/index.html'><a href='snapshots/{directory_id}/index.html'>Open latest completed snapshot</a></html>", encoding="utf-8")
    temp_page.replace(live / "index.html")
    print("[LIVE]", final / "index.html", flush=True)
    return final
