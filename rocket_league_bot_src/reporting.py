from __future__ import annotations

import csv
import html
import json
from pathlib import Path

from .checkpoints import find_latest_compatible_checkpoint


def _read_rows(metrics_path: Path) -> list[dict[str, str]]:
    if not metrics_path.exists():
        return []
    with metrics_path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _to_float(row: dict[str, str], key: str) -> float:
    try:
        return float(row.get(key, "") or 0.0)
    except Exception:
        return 0.0


def _with_iteration(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    indexed_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows, start=1):
        next_row = dict(row)
        next_row["_iteration"] = str(idx)
        indexed_rows.append(next_row)
    return indexed_rows


def _filter_rows(
    rows: list[dict[str, str]],
    max_rows: int | None = 160,
    max_age_hours: float | None = 12.0,
) -> list[dict[str, str]]:
    filtered = rows
    if max_age_hours is not None and filtered:
        latest_time = max(_to_float(row, "unix_time") for row in filtered)
        if latest_time > 0:
            min_time = latest_time - (max_age_hours * 3600.0)
            filtered = [row for row in filtered if _to_float(row, "unix_time") >= min_time]
    if max_rows is not None and max_rows > 0 and len(filtered) > max_rows:
        filtered = filtered[-max_rows:]
    return filtered


def _x_values(rows: list[dict[str, str]], x_axis: str) -> list[float]:
    if not rows:
        return []
    if x_axis == "time":
        values = [_to_float(row, "unix_time") for row in rows]
        if any(value > 0 for value in values):
            return values
    return [_to_float(row, "_iteration") for row in rows]


def _polyline_points(
    x_values: list[float],
    y_values: list[float],
    width: int,
    height: int,
    pad: int,
) -> str:
    if not y_values:
        return ""
    if len(y_values) == 1:
        x = width // 2
        y = height // 2
        return f"{x},{y}"

    x_lo = min(x_values)
    x_hi = max(x_values)
    if abs(x_hi - x_lo) < 1e-9:
        x_lo -= 1.0
        x_hi += 1.0

    y_lo = min(y_values)
    y_hi = max(y_values)
    if abs(y_hi - y_lo) < 1e-9:
        y_lo -= 1.0
        y_hi += 1.0

    points: list[str] = []
    usable_w = max(1, width - 2 * pad)
    usable_h = max(1, height - 2 * pad)
    for x_value, y_value in zip(x_values, y_values):
        x_norm = (x_value - x_lo) / (x_hi - x_lo)
        y_norm = (y_value - y_lo) / (y_hi - y_lo)
        x = pad + x_norm * usable_w
        y = height - pad - y_norm * usable_h
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _stage_spans(rows: list[dict[str, str]], x_values: list[float], width: int, pad: int) -> str:
    if not x_values:
        return ""
    if len(rows) < 2:
        return ""

    stage_colors = {
        "CONTACT": "#d8f3dc",
        "DRIBBLE": "#bee1e6",
        "SHOOT": "#ffd6a5",
        "DEFEND": "#ffcad4",
        "SELF_PLAY": "#cddafd",
    }

    usable_w = max(1, width - 2 * pad)
    x_lo = min(x_values)
    x_hi = max(x_values)
    if abs(x_hi - x_lo) < 1e-9:
        x_lo -= 1.0
        x_hi += 1.0

    def map_x(value: float) -> float:
        return pad + ((value - x_lo) / (x_hi - x_lo)) * usable_w

    spans: list[str] = []
    start = 0
    current = rows[0].get("stage", "")
    for idx, row in enumerate(rows[1:], start=1):
        stage = row.get("stage", "")
        if stage != current:
            x0 = map_x(x_values[start])
            x1 = map_x(x_values[idx - 1])
            spans.append(
                f'<rect x="{x0:.1f}" y="0" width="{max(1.0, x1 - x0):.1f}" height="220" '
                f'fill="{stage_colors.get(current, "#f1f3f5")}" opacity="0.55" />'
            )
            start = idx
            current = stage

    x0 = map_x(x_values[start])
    x1 = map_x(x_values[-1])
    spans.append(
        f'<rect x="{x0:.1f}" y="0" width="{max(1.0, x1 - x0):.1f}" height="220" '
        f'fill="{stage_colors.get(current, "#f1f3f5")}" opacity="0.55" />'
    )
    return "\n".join(spans)


def _chart_svg(
    title: str,
    rows: list[dict[str, str]],
    key: str,
    color: str,
    x_axis: str,
) -> str:
    width = 900
    height = 220
    pad = 28
    values = [_to_float(row, key) for row in rows]
    x_values = _x_values(rows, x_axis)
    points = _polyline_points(x_values, values, width, height, pad)
    latest = values[-1] if values else 0.0
    lo = min(values) if values else 0.0
    hi = max(values) if values else 0.0
    spans = _stage_spans(rows, x_values, width, pad)

    return f"""
    <section class="chart-card">
      <div class="chart-head">
        <h2>{html.escape(title)}</h2>
        <div class="chart-meta">latest {latest:.3f} | min {lo:.3f} | max {hi:.3f}</div>
      </div>
      <svg viewBox="0 0 {width} {height}" class="chart">
        {spans}
        <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />
        <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="axis" />
        <polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />
      </svg>
    </section>
    """


def _read_eval_summary(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def _read_eval_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _filter_eval_history_rows(
    rows: list[dict[str, str]],
    max_current_points: int = 40,
) -> list[dict[str, str]]:
    if not rows:
        return []

    deduped: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        current_ts = int(_to_float(row, "current_timesteps"))
        opponent_ts = int(_to_float(row, "opponent_timesteps"))
        if current_ts <= 0 or opponent_ts <= 0:
            continue
        key = (current_ts, opponent_ts)
        prev = deduped.get(key)
        if prev is None or _to_float(row, "unix_time") >= _to_float(prev, "unix_time"):
            deduped[key] = row

    filtered = sorted(
        deduped.values(),
        key=lambda row: (
            int(_to_float(row, "current_timesteps")),
            int(_to_float(row, "opponent_timesteps")),
        ),
    )
    unique_current_ts = sorted({int(_to_float(row, "current_timesteps")) for row in filtered})
    keep_current_ts = set(unique_current_ts[-max_current_points:])
    return [row for row in filtered if int(_to_float(row, "current_timesteps")) in keep_current_ts]


def _eval_history_section(eval_rows: list[dict[str, str]]) -> str:
    rows = _filter_eval_history_rows(eval_rows)
    if not rows:
        return """
        <section class="eval-card">
          <div class="chart-head">
            <h2>Evaluation History</h2>
            <div class="chart-meta">not available yet</div>
          </div>
          <div class="eval-note">Run the ladder a few times to build a historical trend.</div>
        </section>
        """

    opponent_ts_values = sorted({int(_to_float(row, "opponent_timesteps")) for row in rows})
    palette = ["#2563eb", "#dc2626", "#0891b2", "#ca8a04", "#7c3aed", "#0f766e"]
    color_map = {
        ts: palette[idx % len(palette)]
        for idx, ts in enumerate(opponent_ts_values)
    }

    width = 900
    height = 240
    pad = 36
    x_values = [int(_to_float(row, "current_timesteps")) for row in rows]
    x_lo = min(x_values)
    x_hi = max(x_values)
    if x_lo == x_hi:
        x_lo -= 1
        x_hi += 1

    def map_x(value: float) -> float:
        usable_w = max(1, width - 2 * pad)
        return pad + ((value - x_lo) / (x_hi - x_lo)) * usable_w

    def map_y(value: float) -> float:
        usable_h = max(1, height - 2 * pad)
        return height - pad - value * usable_h

    polylines: list[str] = []
    legend_items: list[str] = []
    summary_rows: list[str] = []

    for opponent_ts in opponent_ts_values:
        series = [row for row in rows if int(_to_float(row, "opponent_timesteps")) == opponent_ts]
        series.sort(key=lambda row: int(_to_float(row, "current_timesteps")))
        points = " ".join(
            f"{map_x(_to_float(row, 'current_timesteps')):.1f},{map_y(_to_float(row, 'blue_win_rate')):.1f}"
            for row in series
        )
        color = color_map[opponent_ts]
        polylines.append(
            f'<polyline fill="none" stroke="{color}" stroke-width="3" points="{points}" />'
        )
        latest = series[-1]
        first = series[0]
        latest_wr = _to_float(latest, "blue_win_rate")
        first_wr = _to_float(first, "blue_win_rate")
        latest_gd = _to_float(latest, "goal_diff_per_episode")
        legend_items.append(
            f'<span class="legend-item"><span class="legend-swatch" style="background:{color}"></span>'
            f'opp {opponent_ts}</span>'
        )
        summary_rows.append(
            "<tr>"
            f"<td>{opponent_ts}</td>"
            f"<td>{int(_to_float(first, 'current_timesteps'))}</td>"
            f"<td>{first_wr:.3f}</td>"
            f"<td>{int(_to_float(latest, 'current_timesteps'))}</td>"
            f"<td>{latest_wr:.3f}</td>"
            f"<td>{latest_wr - first_wr:+.3f}</td>"
            f"<td>{latest_gd:.3f}</td>"
            "</tr>"
        )

    y_guides = []
    for tick in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = map_y(tick)
        y_guides.append(
            f'<line x1="{pad}" y1="{y:.1f}" x2="{width - pad}" y2="{y:.1f}" class="grid" />'
        )
        y_guides.append(
            f'<text x="{pad - 8}" y="{y + 4:.1f}" text-anchor="end" class="axis-label">{tick:.2f}</text>'
        )

    return f"""
    <section class="eval-card">
      <div class="chart-head">
        <h2>Evaluation History</h2>
        <div class="chart-meta">blue win rate vs current checkpoint timesteps</div>
      </div>
      <div class="legend-row">
        {''.join(legend_items)}
      </div>
      <svg viewBox="0 0 {width} {height}" class="chart">
        {''.join(y_guides)}
        <line x1="{pad}" y1="{height - pad}" x2="{width - pad}" y2="{height - pad}" class="axis" />
        <line x1="{pad}" y1="{pad}" x2="{pad}" y2="{height - pad}" class="axis" />
        {''.join(polylines)}
      </svg>
      <div class="table-wrap">
        <table class="eval-table">
          <thead>
            <tr>
              <th>Opponent TS</th>
              <th>First Current TS</th>
              <th>First Win Rate</th>
              <th>Latest Current TS</th>
              <th>Latest Win Rate</th>
              <th>Delta</th>
              <th>Latest Goal Diff</th>
            </tr>
          </thead>
          <tbody>
            {''.join(summary_rows)}
          </tbody>
        </table>
      </div>
    </section>
    """


def _eval_section(
    summary: dict[str, object],
    latest_checkpoint: str,
    eval_error: str,
) -> str:
    if summary:
        summary_checkpoint = str(summary.get("current_checkpoint_dir", ""))
        stale = bool(latest_checkpoint and summary_checkpoint != latest_checkpoint)
        rows = summary.get("rows", [])
        row_html = ""
        if isinstance(rows, list) and rows:
            rendered_rows: list[str] = []
            for row in rows:
                if not isinstance(row, dict):
                    continue
                rendered_rows.append(
                    "<tr>"
                    f"<td>{int(row.get('anchor_slot', 0))}</td>"
                    f"<td>{int(row.get('opponent_timesteps', 0))}</td>"
                    f"<td>{float(row.get('blue_win_rate', 0.0)):.3f}</td>"
                    f"<td>{float(row.get('goal_diff_per_episode', 0.0)):.3f}</td>"
                    f"<td>{int(row.get('blue_goals', 0))}</td>"
                    f"<td>{int(row.get('orange_goals', 0))}</td>"
                    f"<td>{int(row.get('draws', 0))}</td>"
                    "</tr>"
                )
            row_html = "\n".join(rendered_rows)
        if not row_html:
            row_html = (
                '<tr><td colspan="7" class="eval-empty">'
                "No ladder results yet for the current checkpoint."
                "</td></tr>"
            )

        error_line = (
            f'<div class="eval-note error">Refresh error: {html.escape(eval_error)}</div>'
            if eval_error
            else ""
        )
        stale_line = f"stale {'yes' if stale else 'no'}"
        return f"""
        <section class="eval-card">
          <div class="chart-head">
            <h2>Evaluation Ladder</h2>
            <div class="chart-meta">
              avg blue win rate {float(summary.get("avg_blue_win_rate", 0.0)):.3f} |
              avg goal diff {float(summary.get("avg_goal_diff_per_episode", 0.0)):.3f} |
              {stale_line}
            </div>
          </div>
          <div class="summary-grid">
            <div class="summary-card"><span>Eval Stage</span><strong>{html.escape(str(summary.get("stage", "")))}</strong></div>
            <div class="summary-card"><span>Eval Difficulty</span><strong>{float(summary.get("difficulty", 0.0)):.3f}</strong></div>
            <div class="summary-card"><span>Current TS</span><strong>{int(summary.get("current_timesteps", 0))}</strong></div>
            <div class="summary-card"><span>Refresh After</span><strong>{int(summary.get("refresh_after_timesteps", 0))}</strong></div>
          </div>
          {error_line}
          <div class="table-wrap">
            <table class="eval-table">
              <thead>
                <tr>
                  <th>Slot</th>
                  <th>Opponent TS</th>
                  <th>Blue Win</th>
                  <th>Goal Diff</th>
                  <th>Blue Goals</th>
                  <th>Orange Goals</th>
                  <th>Draws</th>
                </tr>
              </thead>
              <tbody>
                {row_html}
              </tbody>
            </table>
          </div>
        </section>
        """

    pending_note = "Waiting for the first evaluation refresh."
    if eval_error:
        pending_note = f"Refresh error: {html.escape(eval_error)}"
    return f"""
    <section class="eval-card">
      <div class="chart-head">
        <h2>Evaluation Ladder</h2>
        <div class="chart-meta">not available yet</div>
      </div>
      <div class="eval-note{' error' if eval_error else ''}">{pending_note}</div>
    </section>
    """


def write_training_report(
    metrics_path: str = "data/training_metrics.csv",
    output_path: str = "data/training_report.html",
    x_axis: str = "iteration",
    max_rows: int | None = 160,
    max_age_hours: float | None = 12.0,
    eval_summary_path: str = "data/eval/latest_summary.json",
    eval_results_path: str = "data/eval/results.csv",
    eval_error: str = "",
    checkpoint_root: str = "data/checkpoints",
) -> None:
    metrics = Path(metrics_path)
    all_rows = _with_iteration(_read_rows(metrics))
    rows = _filter_rows(all_rows, max_rows=max_rows, max_age_hours=max_age_hours)
    eval_summary = _read_eval_summary(Path(eval_summary_path))
    eval_rows = _read_eval_rows(Path(eval_results_path))
    latest_checkpoint = find_latest_compatible_checkpoint(checkpoint_root)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        last = rows[-1]
        first_iter = int(_to_float(rows[0], "_iteration"))
        last_iter = int(_to_float(rows[-1], "_iteration"))
        if x_axis == "time":
            x_axis_label = "Wall clock time"
        else:
            x_axis_label = f"Training iteration {first_iter} to {last_iter}"
        window_label = f"Showing {len(rows)} of {len(all_rows)} rows"
        if max_age_hours is not None:
            window_label += f" from the last {max_age_hours:g} hours"
        summary = f"""
        <div class="summary-grid">
          <div class="summary-card"><span>Stage</span><strong>{html.escape(last.get("stage", ""))}</strong></div>
          <div class="summary-card"><span>Difficulty</span><strong>{html.escape(last.get("difficulty", ""))}</strong></div>
          <div class="summary-card"><span>Touch Rate</span><strong>{html.escape(last.get("touch_rate", ""))}</strong></div>
          <div class="summary-card"><span>Goal Rate</span><strong>{html.escape(last.get("goal_rate", ""))}</strong></div>
          <div class="summary-card"><span>Blue Goal Rate</span><strong>{html.escape(last.get("blue_goal_rate", ""))}</strong></div>
          <div class="summary-card"><span>Median T First</span><strong>{html.escape(last.get("median_t_first", ""))}</strong></div>
          <div class="summary-card"><span>Median T Goal</span><strong>{html.escape(last.get("median_t_goal", ""))}</strong></div>
        </div>
        <div class="report-meta">{html.escape(x_axis_label)} | {html.escape(window_label)}</div>
        """
        charts = "\n".join(
            [
                _chart_svg("Average Return", rows, "avg_return", "#0f766e", x_axis),
                _chart_svg("Touch Rate", rows, "touch_rate", "#2563eb", x_axis),
                _chart_svg("Goal Rate", rows, "goal_rate", "#dc2626", x_axis),
                _chart_svg("Blue Goal Rate", rows, "blue_goal_rate", "#0891b2", x_axis),
                _chart_svg("Median T First", rows, "median_t_first", "#ca8a04", x_axis),
                _chart_svg("Median T Goal", rows, "median_t_goal", "#c2410c", x_axis),
                _chart_svg("Difficulty", rows, "difficulty", "#7c3aed", x_axis),
            ]
        )
    else:
        summary = '<p class="empty">No metrics yet. Start training and refresh this page.</p>'
        charts = ""
    eval_html = _eval_section(eval_summary, latest_checkpoint, eval_error)
    eval_history_html = _eval_history_section(eval_rows)

    html_text = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="color-scheme" content="light dark" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rocket League Bot Training Report</title>
  <style>
    :root {{
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
      --page-top: #f3f4f6;
      --page-bottom: #f7f7f2;
      --chart-bg: #fcfcfb;
      --axis: #9ca3af;
      --shadow: rgba(15, 23, 42, 0.05);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0b1220;
        --panel: #111827;
        --ink: #e5eefb;
        --muted: #94a3b8;
        --border: #243041;
        --page-top: #09111d;
        --page-bottom: #0b1220;
        --chart-bg: #0f172a;
        --axis: #475569;
        --shadow: rgba(2, 6, 23, 0.35);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, var(--page-top) 0%, var(--page-bottom) 100%);
      color: var(--ink);
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 24px;
    }}
    h1 {{
      margin: 0 0 8px 0;
      font-size: 32px;
    }}
    .sub {{
      color: var(--muted);
      margin-bottom: 20px;
    }}
    .summary-grid {{
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
      gap: 12px;
      margin-bottom: 18px;
    }}
    .report-meta {{
      color: var(--muted);
      margin: 0 0 18px 0;
      font-size: 14px;
    }}
    .summary-card, .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 8px 30px var(--shadow);
    }}
    .summary-card {{
      padding: 14px;
    }}
    .summary-card span {{
      display: block;
      color: var(--muted);
      font-size: 12px;
      margin-bottom: 4px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }}
    .summary-card strong {{
      font-size: 20px;
    }}
    .chart-card {{
      padding: 14px 14px 8px 14px;
      margin-bottom: 16px;
    }}
    .chart-head {{
      display: flex;
      justify-content: space-between;
      gap: 16px;
      align-items: baseline;
    }}
    .chart-head h2 {{
      margin: 0 0 10px 0;
      font-size: 18px;
    }}
    .chart-meta {{
      color: var(--muted);
      font-size: 13px;
    }}
    .chart {{
      width: 100%;
      height: auto;
      display: block;
      border-radius: 10px;
      background: var(--chart-bg);
    }}
    .axis {{
      stroke: var(--axis);
      stroke-width: 1;
    }}
    .empty {{
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
    }}
    .eval-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 8px 30px var(--shadow);
      padding: 14px;
      margin-bottom: 16px;
    }}
    .table-wrap {{
      overflow-x: auto;
    }}
    .eval-table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 14px;
    }}
    .eval-table th, .eval-table td {{
      text-align: left;
      padding: 10px 8px;
      border-top: 1px solid var(--border);
    }}
    .eval-empty {{
      color: var(--muted);
      font-style: italic;
    }}
    .eval-note {{
      margin: 0 0 12px 0;
      color: var(--muted);
      font-size: 14px;
    }}
    .eval-note.error {{
      color: #dc2626;
    }}
  </style>
</head>
<body>
  <main id="report-root">
    <h1>Training Report</h1>
    <div class="sub">Auto-refreshes every 5 seconds. Stage backgrounds show curriculum transitions.</div>
    {summary}
    {eval_html}
    {eval_history_html}
    {charts}
  </main>
  <script>
    (() => {{
      const root = document.getElementById("report-root");
      if (!root) return;

      let lastHtml = root.innerHTML;

      async function refreshReport() {{
        try {{
          const url = new URL(window.location.href);
          url.searchParams.set("_ts", String(Date.now()));
          const response = await fetch(url, {{ cache: "no-store" }});
          if (!response.ok) return;

          const text = await response.text();
          const parser = new DOMParser();
          const doc = parser.parseFromString(text, "text/html");
          const nextRoot = doc.getElementById("report-root");
          if (!nextRoot) return;

          const nextHtml = nextRoot.innerHTML;
          if (nextHtml !== lastHtml) {{
            root.innerHTML = nextHtml;
            lastHtml = nextHtml;
          }}
        }} catch (_err) {{
        }}
      }}

      window.setInterval(refreshReport, 5000);
    }})();
  </script>
</body>
</html>
"""
    temp_output = output.with_suffix(output.suffix + ".tmp")
    temp_output.write_text(html_text, encoding="utf-8")
    temp_output.replace(output)
