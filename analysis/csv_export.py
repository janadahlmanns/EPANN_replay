"""Generic CSV writer shared by every plot-saving function in run_evolution.py and
composite_plot.py: each PNG a plot function saves is paired with one CSV holding
exactly the (already normalized/filtered/grouped) data that went into that plot --
nothing more, nothing less -- so the two outputs can never drift apart in meaning
and the CSV alone is enough to recreate the PNG by eye."""

import csv


# ==== 1) GENERIC ROW WRITER ======================================================
def write_csv(path, rows):
    """Write rows (list of dicts, all sharing the same keys) to path as CSV, with a
    header taken from the first row's keys and column order matching that row's
    key insertion order."""
    if not rows:
        raise ValueError(f"No rows to write for {path}")
    fieldnames = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
