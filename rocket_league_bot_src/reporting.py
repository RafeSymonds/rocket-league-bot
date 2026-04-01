from __future__ import annotations

import csv
import html
from pathlib import Path


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


def _polyline_points(values: list[float], width: int, height: int, pad: int) -> str:
    if not values:
        return ""
    if len(values) == 1:
        x = width // 2
        y = height // 2
        return f"{x},{y}"

    lo = min(values)
    hi = max(values)
    if abs(hi - lo) < 1e-9:
        lo -= 1.0
        hi += 1.0

    points: list[str] = []
    usable_w = max(1, width - 2 * pad)
    usable_h = max(1, height - 2 * pad)
    for idx, value in enumerate(values):
        x = pad + usable_w * idx / (len(values) - 1)
        norm = (value - lo) / (hi - lo)
        y = height - pad - norm * usable_h
        points.append(f"{x:.1f},{y:.1f}")
    return " ".join(points)


def _stage_spans(rows: list[dict[str, str]], width: int, pad: int) -> str:
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
    spans: list[str] = []
    start = 0
    current = rows[0].get("stage", "")
    for idx, row in enumerate(rows[1:], start=1):
        stage = row.get("stage", "")
        if stage != current:
            x0 = pad + usable_w * start / (len(rows) - 1)
            x1 = pad + usable_w * (idx - 1) / (len(rows) - 1)
            spans.append(
                f'<rect x="{x0:.1f}" y="0" width="{max(1.0, x1 - x0):.1f}" height="220" '
                f'fill="{stage_colors.get(current, "#f1f3f5")}" opacity="0.55" />'
            )
            start = idx
            current = stage

    x0 = pad + usable_w * start / (len(rows) - 1)
    x1 = pad + usable_w
    spans.append(
        f'<rect x="{x0:.1f}" y="0" width="{max(1.0, x1 - x0):.1f}" height="220" '
        f'fill="{stage_colors.get(current, "#f1f3f5")}" opacity="0.55" />'
    )
    return "\n".join(spans)


def _chart_svg(title: str, rows: list[dict[str, str]], key: str, color: str) -> str:
    width = 900
    height = 220
    pad = 28
    values = [_to_float(row, key) for row in rows]
    points = _polyline_points(values, width, height, pad)
    latest = values[-1] if values else 0.0
    lo = min(values) if values else 0.0
    hi = max(values) if values else 0.0
    spans = _stage_spans(rows, width, pad)

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


def write_training_report(
    metrics_path: str = "data/training_metrics.csv",
    output_path: str = "data/training_report.html",
) -> None:
    metrics = Path(metrics_path)
    rows = _read_rows(metrics)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if rows:
        last = rows[-1]
        summary = f"""
        <div class="summary-grid">
          <div class="summary-card"><span>Stage</span><strong>{html.escape(last.get("stage", ""))}</strong></div>
          <div class="summary-card"><span>Difficulty</span><strong>{html.escape(last.get("difficulty", ""))}</strong></div>
          <div class="summary-card"><span>Touch Rate</span><strong>{html.escape(last.get("touch_rate", ""))}</strong></div>
          <div class="summary-card"><span>Goal Rate</span><strong>{html.escape(last.get("goal_rate", ""))}</strong></div>
          <div class="summary-card"><span>Median T First</span><strong>{html.escape(last.get("median_t_first", ""))}</strong></div>
          <div class="summary-card"><span>Median T Goal</span><strong>{html.escape(last.get("median_t_goal", ""))}</strong></div>
        </div>
        """
        charts = "\n".join(
            [
                _chart_svg("Average Return", rows, "avg_return", "#0f766e"),
                _chart_svg("Touch Rate", rows, "touch_rate", "#2563eb"),
                _chart_svg("Goal Rate", rows, "goal_rate", "#dc2626"),
                _chart_svg("Difficulty", rows, "difficulty", "#7c3aed"),
            ]
        )
    else:
        summary = '<p class="empty">No metrics yet. Start training and refresh this page.</p>'
        charts = ""

    output.write_text(
        f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta http-equiv="refresh" content="5" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Rocket League Bot Training Report</title>
  <style>
    :root {{
      --bg: #f7f7f2;
      --panel: #ffffff;
      --ink: #1f2937;
      --muted: #6b7280;
      --border: #e5e7eb;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: ui-sans-serif, system-ui, sans-serif;
      background: linear-gradient(180deg, #f3f4f6 0%, #f7f7f2 100%);
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
    .summary-card, .chart-card {{
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
      box-shadow: 0 8px 30px rgba(15, 23, 42, 0.05);
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
      background: #fcfcfb;
    }}
    .axis {{
      stroke: #9ca3af;
      stroke-width: 1;
    }}
    .empty {{
      padding: 18px;
      background: var(--panel);
      border: 1px solid var(--border);
      border-radius: 14px;
    }}
  </style>
</head>
<body>
  <main>
    <h1>Training Report</h1>
    <div class="sub">Auto-refreshes every 5 seconds. Stage backgrounds show curriculum transitions.</div>
    {summary}
    {charts}
  </main>
</body>
</html>
""",
        encoding="utf-8",
    )
