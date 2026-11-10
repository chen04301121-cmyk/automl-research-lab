"""Markdown summary + self-contained HTML dashboard for the Part 3 evaluation."""

from __future__ import annotations

import html
import json
from collections import defaultdict
from pathlib import Path

SYSTEM_ORDER = ["adaptive", "class_agent_loop", "fixed_airflow"]
LABELS = {"adaptive": "Adaptive multi-agent (Part 3)", "class_agent_loop": "Class agent loop (baseline 2)",
          "fixed_airflow": "Fixed Airflow workflow (baseline 1)"}
SHORT = {"adaptive": "Adaptive", "class_agent_loop": "Class loop", "fixed_airflow": "Fixed Airflow"}


def summarize(out: dict) -> dict:
    by_sys = defaultdict(list)
    for r in out["results"]:
        by_sys[r["system"]].append(r)
    summary = {}
    for s in SYSTEM_ORDER:
        rows = by_sys.get(s, [])
        if not rows:
            continue
        groups = defaultdict(lambda: [0, 0])
        for row in rows:
            for c in row["criteria"]:
                groups[c["group"]][0] += int(c["passed"])
                groups[c["group"]][1] += 1
        f1s = [(row["result"].get("holdout") or {}).get("f1") for row in rows
               if row["case"].startswith(("C4", "C5", "C6", "C8"))]
        f1s = [f if f is not None else 0.0 for f in f1s]
        summary[s] = {
            "cases_passed": sum(r["passed"] for r in rows),
            "cases": len(rows),
            "mean_score": round(sum(r["score"] for r in rows) / len(rows), 3),
            "completed": sum(r["result"].get("status") == "completed" for r in rows),
            "group_rates": {g: round(p / t, 3) for g, (p, t) in groups.items()},
            "mean_f1_signal_cases": round(sum(f1s) / len(f1s), 3) if f1s else None,
            "agent_calls": sum(r["result"].get("agent_calls", 0) for r in rows),
            "experiments": sum(r["result"].get("experiments", 0) for r in rows),
            "retries": sum(r["result"].get("retries", 0) for r in rows),
            "runtime_s": round(sum(r["result"].get("runtime_s", 0) for r in rows), 1),
        }
    return summary


def _cell(row: dict | None) -> str:
    if row is None:
        return "-"
    f1 = (row["result"].get("holdout") or {}).get("f1")
    f1s = f" · F1 {f1:.2f}" if f1 is not None else ""
    return f"{'✅' if row['passed'] else '❌'} {row['score']:.2f}{f1s}"


def write_markdown(out: dict, path: Path) -> None:
    summ = summarize(out)
    idx = {(r["case"], r["system"]): r for r in out["results"]}
    systems = [s for s in SYSTEM_ORDER if s in summ]
    L = ["### Overall", "",
         "| System | Cases passed | Mean score | Completed runs | Intermediate checks | Final checks | Coordination checks | Mean holdout F1 (signal cases) | Agent/step calls | Experiments | Retries | Runtime (s) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for s in systems:
        x = summ[s]
        g = x["group_rates"]
        L.append(f"| {LABELS[s]} | **{x['cases_passed']}/{x['cases']}** | {x['mean_score']:.2f} | {x['completed']}/{x['cases']} | "
                 f"{g.get('intermediate', 0):.0%} | {g.get('final', 0):.0%} | {g.get('coordination', 0):.0%} | "
                 f"{x['mean_f1_signal_cases']} | {x['agent_calls']} | {x['experiments']} | {x['retries']} | {x['runtime_s']} |")
    L += ["", "### Per case (pass/fail · weighted score · holdout F1)", "",
          "| Case | " + " | ".join(SHORT[s] for s in systems) + " |",
          "|---|" + "---|" * len(systems)]
    for c in out["cases"]:
        L.append(f"| {c['id']} — {c['title']} | " + " | ".join(_cell(idx.get((c['id'], s))) for s in systems) + " |")
    L += ["", "### Criteria detail", ""]
    for c in out["cases"]:
        L.append(f"**{c['id']} — {c['title']}** (pass threshold {c['pass_threshold']}) — *{c['expected_behavior']}*")
        L.append("")
        L.append("| Criterion | Group | Weight | " + " | ".join(SHORT[s] for s in systems) + " |")
        L.append("|---|---|---|" + "---|" * len(systems))
        for i, crit in enumerate(c["criteria"]):
            name = crit["check"] + (f" {crit['args']}" if crit.get("args") else "")
            cells = []
            for s in systems:
                row = idx.get((c["id"], s))
                if row is None:
                    cells.append("-")
                    continue
                cr = row["criteria"][i]
                cells.append(f"{'✅' if cr['passed'] else '❌'} {cr['observed']}")
            L.append(f"| `{name}` | {crit['group']} | {crit.get('weight', 1)} | " + " | ".join(cells) + " |")
        L.append("")
    path.write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
CSS = """
:root{--bg:#f7f8f6;--panel:#ffffff;--ink:#1d2321;--muted:#5d6762;--line:#dfe4e0;--pass:#1f7a4d;--pass-bg:#e3f2e9;
--fail:#b4362c;--fail-bg:#f8e5e2;--a:#2f6f9f;--b:#c08a2b;--c:#7d6aa8;--chip:#eef1ee}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#141816;--panel:#1c211f;--ink:#e6ebe8;--muted:#9aa5a0;
--line:#2d3431;--pass:#5cc08d;--pass-bg:#173326;--fail:#ef8a7f;--fail-bg:#3a1f1c;--a:#6fa9d6;--b:#e0b25a;--c:#b3a2dc;--chip:#252b28;color-scheme:dark}}
:root[data-theme="dark"]{--bg:#141816;--panel:#1c211f;--ink:#e6ebe8;--muted:#9aa5a0;--line:#2d3431;--pass:#5cc08d;--pass-bg:#173326;
--fail:#ef8a7f;--fail-bg:#3a1f1c;--a:#6fa9d6;--b:#e0b25a;--c:#b3a2dc;--chip:#252b28;color-scheme:dark}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 "IBM Plex Sans",system-ui,sans-serif}
main{max-width:1180px;margin:0 auto;padding-block:28px 60px;padding-left:max(16px,env(safe-area-inset-left,0px));padding-right:max(16px,env(safe-area-inset-right,0px))}h1{font:600 28px/1.2 "IBM Plex Serif",Georgia,serif;margin:0 0 4px}
h2{font:600 19px/1.3 "IBM Plex Serif",Georgia,serif;margin:34px 0 12px}.sub{color:var(--muted);margin:0 0 20px}
.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.card h3{margin:0 0 8px;font-size:14px;font-weight:600;display:flex;align-items:center;gap:8px}
.dot{width:10px;height:10px;border-radius:50%;display:inline-block}
.big{font:600 34px/1 "IBM Plex Mono",monospace}.big small{font-size:16px;color:var(--muted)}
.kv{display:grid;grid-template-columns:1fr auto;gap:2px 10px;margin-top:10px;font-size:13px}.kv span:nth-child(odd){color:var(--muted)}
.tablewrap{overflow-x:auto;background:var(--panel);border:1px solid var(--line);border-radius:10px}
table{border-collapse:collapse;width:100%;font-size:13px}th,td{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font:600 12px "IBM Plex Mono",monospace}
.ok{background:var(--pass-bg);color:var(--pass)}.no{background:var(--fail-bg);color:var(--fail)}
.obs{color:var(--muted);font-size:12px}.bars{display:grid;gap:10px}
.barrow{display:grid;grid-template-columns:96px minmax(60px,1fr) 44px;align-items:center;gap:10px;font-size:13px}
.track{background:var(--chip);border-radius:6px;height:14px;overflow:hidden}.fill{height:100%;border-radius:6px}
.route{font:12px/1.7 "IBM Plex Mono",monospace;color:var(--muted)}.route b{color:var(--ink);font-weight:600}
.graph{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:12px;overflow-x:auto}
details{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:10px 14px;margin:8px 0}
summary{cursor:pointer;font-weight:600}.muted{color:var(--muted)}
.barrow span:last-child{text-align:right;font-variant-numeric:tabular-nums}
"""
COLORS = {"adaptive": "var(--a)", "class_agent_loop": "var(--b)", "fixed_airflow": "var(--c)"}


def write_dashboard(out: dict, path: Path, artifact_path: Path | None = None) -> None:
    """Write a standalone HTML file (path) and, optionally, a version for a claude.ai artifact."""
    from part3.adaptive_system import mermaid

    summ = summarize(out)
    idx = {(r["case"], r["system"]): r for r in out["results"]}
    systems = [s for s in SYSTEM_ORDER if s in summ]
    e = html.escape

    cards = []
    for s in systems:
        x = summ[s]
        g = x["group_rates"]
        cards.append(
            f'<div class="card"><h3><span class="dot" style="background:{COLORS[s]}"></span>{e(LABELS[s])}</h3>'
            f'<div class="big">{x["cases_passed"]}<small>/{x["cases"]} cases passed</small></div>'
            f'<div class="kv"><span>Mean weighted score</span><span>{x["mean_score"]:.2f}</span>'
            f'<span>Runs completed</span><span>{x["completed"]}/{x["cases"]}</span>'
            f'<span>Intermediate checks</span><span>{g.get("intermediate", 0):.0%}</span>'
            f'<span>Final-output checks</span><span>{g.get("final", 0):.0%}</span>'
            f'<span>Coordination checks</span><span>{g.get("coordination", 0):.0%}</span>'
            f'<span>Mean holdout F1 (signal cases)</span><span>{x["mean_f1_signal_cases"]}</span>'
            f'<span>Experiments / retries</span><span>{x["experiments"]} / {x["retries"]}</span>'
            f'<span>Runtime</span><span>{x["runtime_s"]} s</span></div></div>')

    def bars(title, key, fmt):
        rows = []
        for s in systems:
            v = key(summ[s]) or 0
            rows.append(f'<div class="barrow"><span>{e(SHORT[s])}</span><div class="track"><div class="fill" '
                        f'style="width:{max(0, min(1, v)) * 100:.0f}%;background:{COLORS[s]}"></div></div><span>{fmt(v)}</span></div>')
        return f'<div class="card"><h3>{e(title)}</h3><div class="bars">{"".join(rows)}</div></div>'

    bar_html = "".join([
        bars("Cases passed", lambda x: x["cases_passed"] / x["cases"], lambda v: f"{v:.0%}"),
        bars("Intermediate-output checks", lambda x: x["group_rates"].get("intermediate", 0), lambda v: f"{v:.0%}"),
        bars("Coordination checks", lambda x: x["group_rates"].get("coordination", 0), lambda v: f"{v:.0%}"),
        bars("Mean holdout F1 (C4/C5/C6/C8)", lambda x: x["mean_f1_signal_cases"], lambda v: f"{v:.2f}"),
    ])

    head = "".join(f"<th>{e(SHORT[s])}</th>" for s in systems)
    case_rows = []
    for c in out["cases"]:
        cells = []
        for s in systems:
            row = idx.get((c["id"], s))
            if row is None:
                cells.append("<td>-</td>")
                continue
            f1 = (row["result"].get("holdout") or {}).get("f1")
            cls = "ok" if row["passed"] else "no"
            cells.append(f'<td><span class="pill {cls}">{"PASS" if row["passed"] else "FAIL"} {row["score"]:.2f}</span>'
                         f'<div class="obs">F1 {"-" if f1 is None else f"{f1:.3f}"} · '
                         f'{row["result"].get("experiments", 0)} exps · {row["result"].get("runtime_s", 0)} s</div></td>')
        case_rows.append(f'<tr><td><b>{e(c["id"])}</b><div class="obs">{e(c["title"])}</div></td>{"".join(cells)}</tr>')

    details = []
    for c in out["cases"]:
        crit_rows = []
        for i, crit in enumerate(c["criteria"]):
            name = crit["check"] + (f" {json.dumps(crit['args'])}" if crit.get("args") else "")
            tds = []
            for s in systems:
                row = idx.get((c["id"], s))
                cr = row["criteria"][i] if row else None
                tds.append("<td>-</td>" if cr is None else
                           f'<td><span class="pill {"ok" if cr["passed"] else "no"}">{"✓" if cr["passed"] else "✗"}</span> '
                           f'<span class="obs">{e(str(cr["observed"]))}</span></td>')
            crit_rows.append(f'<tr><td><code>{e(name)}</code></td><td>{crit["group"]}</td><td>{crit.get("weight", 1)}</td>{"".join(tds)}</tr>')
        ad = idx.get((c["id"], "adaptive"))
        route = ""
        if ad:
            steps = [ev for ev in ad["result"].get("events", [])]
            route = "<div class='route'>" + " → ".join(
                f"<b>{e(ev['agent'])}</b> <span>({e(ev['note'][:60])})</span>" for ev in steps) + "</div>"
        inp = json.dumps(c["input"])
        details.append(
            f'<details><summary>{e(c["id"])} — {e(c["title"])}</summary>'
            f'<p><b>Input</b> <code>{e(inp)}</code><br><b>Expected</b> {e(c["expected_behavior"])}<br>'
            f'<b>Pass threshold</b> weighted score ≥ {c["pass_threshold"]}</p>'
            f'<div class="tablewrap"><table><tr><th>Criterion</th><th>Group</th><th>Weight</th>{head}</tr>{"".join(crit_rows)}</table></div>'
            f'<p class="muted" style="margin:10px 0 4px">Adaptive route chosen by the supervisor:</p>{route}</details>')

    fonts = ('<link rel="preconnect" href="https://fonts.googleapis.com"><link href="https://fonts.googleapis.com/css2?'
             'family=IBM+Plex+Mono:wght@400;600&family=IBM+Plex+Sans:wght@400;600&family=IBM+Plex+Serif:wght@600&display=swap" rel="stylesheet">')
    content = f"""<main>
<h1>Adaptive multi-agent system: evaluation</h1>
<p class="sub">IDS706 Week 11 · Part 3 · {len(out["cases"])} test cases × {len(systems)} systems · generated {e(out["generated_at"])}</p>
<div class="cards">{"".join(cards)}</div>
<h2>Headline comparison</h2><div class="cards">{bar_html}</div>
<h2>Per-case results</h2><div class="tablewrap"><table><tr><th>Case</th>{head}</tr>{"".join(case_rows)}</table></div>
<h2>Workflow graph (supervisor + 9 specialists)</h2><div class="graph"><pre class="mermaid">{e(mermaid())}</pre></div>
<h2>Test cases: input, expected behaviour, scoring, threshold</h2>{"".join(details)}
</main>"""
    title = "<title>Adaptive Agent Scorecard</title>"
    # Standalone file: full document + mermaid from a CDN so it renders when opened locally
    page = (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1,viewport-fit=cover">{title}{fonts}'
            f'<style>{CSS}</style></head><body>{content}'
            '<script src="https://cdn.jsdelivr.net/npm/mermaid@10.9.1/dist/mermaid.min.js"></script>'
            "<script>try{mermaid.initialize({startOnLoad:true,theme:matchMedia('(prefers-color-scheme: dark)').matches?'dark':'neutral'})}catch(e){}</script>"
            '</body></html>')
    path.write_text(page)
    if artifact_path is not None:
        # claude.ai artifact: the host adds the document skeleton and renders mermaid itself
        artifact_path.write_text(f'{title}<meta name="claude:viewport-fit-sides" content="cover">{fonts}'
                                 f'<style>{CSS}</style>{content}')
