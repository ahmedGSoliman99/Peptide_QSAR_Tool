"""Reporting and export helpers."""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

import pandas as pd
import plotly.graph_objects as go


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")


def dataframe_to_excel_bytes(
    tables: dict[str, pd.DataFrame],
) -> bytes:
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        for name, table in tables.items():
            safe_name = str(name)[:31] or "Sheet1"
            table.to_excel(writer, sheet_name=safe_name, index=False)
    buffer.seek(0)
    return buffer.read()


def _table_html(df: pd.DataFrame, max_rows: int = 25) -> str:
    if df is None or df.empty:
        return "<p><i>No data available.</i></p>"
    trimmed = df.head(max_rows).copy()
    return trimmed.to_html(index=False, classes="data-table")


def _metrics_html(metrics: dict[str, Any]) -> str:
    if not metrics:
        return "<p><i>No metrics available.</i></p>"
    rows: list[str] = []
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            rows.append(f"<tr><td>{key}</td><td>{value:.5g}</td></tr>")
        else:
            rows.append(f"<tr><td>{key}</td><td>{value}</td></tr>")
    return f"<table class='data-table'><tbody>{''.join(rows)}</tbody></table>"


def _figures_html(figures: dict[str, go.Figure] | None) -> str:
    if not figures:
        return "<p><i>No plots were captured for this report.</i></p>"

    blocks: list[str] = []
    include_js = True
    for title, fig in figures.items():
        if fig is None:
            continue
        block = fig.to_html(
            full_html=False,
            include_plotlyjs="cdn" if include_js else False,
            config={"displaylogo": False},
        )
        include_js = False
        blocks.append(f"<h4>{title}</h4>{block}")
    if not blocks:
        return "<p><i>No plots were captured for this report.</i></p>"
    return "\n".join(blocks)


def generate_html_report(
    input_summary: dict[str, Any],
    descriptor_summary: dict[str, Any],
    model_summary: dict[str, Any],
    metrics: dict[str, Any],
    prediction_df: pd.DataFrame | None = None,
    comparison_df: pd.DataFrame | None = None,
    figures: dict[str, go.Figure] | None = None,
) -> str:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    input_table = _metrics_html(input_summary)
    descriptor_table = _metrics_html(descriptor_summary)
    model_table = _metrics_html(model_summary)
    metrics_table = _metrics_html(metrics)
    prediction_table = _table_html(prediction_df)
    comparison_table = _table_html(comparison_df)
    figures_block = _figures_html(figures)

    return f"""
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Peptide QSAR Report</title>
  <style>
    body {{
      font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      margin: 24px;
      color: #10233f;
      background: #f6fbff;
    }}
    h1, h2, h3 {{
      color: #0b3a5e;
    }}
    .card {{
      background: #ffffff;
      border: 1px solid #d7e5f3;
      border-radius: 12px;
      padding: 16px;
      margin-bottom: 16px;
      box-shadow: 0 4px 12px rgba(11, 58, 94, 0.07);
    }}
    .meta {{
      color: #35526f;
      margin-bottom: 12px;
    }}
    table.data-table {{
      border-collapse: collapse;
      width: 100%;
      font-size: 0.92rem;
    }}
    table.data-table th, table.data-table td {{
      border: 1px solid #d7e5f3;
      padding: 6px 8px;
      text-align: left;
    }}
    table.data-table th {{
      background: #eaf4ff;
    }}
  </style>
</head>
<body>
  <h1>Peptide QSAR Prediction Tool Report</h1>
  <div class="meta">
    <strong>Generated:</strong> {timestamp}
  </div>

  <div class="card">
    <h2>1. Input Summary</h2>
    {input_table}
  </div>

  <div class="card">
    <h2>2. Descriptor Summary</h2>
    {descriptor_table}
  </div>

  <div class="card">
    <h2>3. Model Summary</h2>
    {model_table}
    <h3>Model Comparison Table</h3>
    {comparison_table}
  </div>

  <div class="card">
    <h2>4. Evaluation Metrics</h2>
    {metrics_table}
  </div>

  <div class="card">
    <h2>5. Prediction Results (Top Rows)</h2>
    {prediction_table}
  </div>

  <div class="card">
    <h2>6. Visualizations</h2>
    {figures_block}
  </div>
</body>
</html>
""".strip()

