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


def _time_values(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    values = [_to_float(row, "unix_time") for row in rows]
    if any(value > 0 for value in values):
        return values
    return [float(i) for i in range(len(rows))]


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


def _chart_svg(title: str, rows: list[dict[str, str]], key: str, color: str) -> str:
    width = 900
    height = 220
    pad = 28
    values = [_to_float(row, key) for row in rows]
    x_values = _time_values(rows)
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
  </style>
</head>
<body>
  <main id="report-root">
    <h1>Training Report</h1>
    <div class="sub">Auto-refreshes every 5 seconds. Stage backgrounds show curriculum transitions.</div>
    {summary}
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
""",
        encoding="utf-8",
    )
