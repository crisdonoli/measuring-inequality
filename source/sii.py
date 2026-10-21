"""Helpers for SII tax tabulations (PUB_Total.xlsb).

Logic for BFM; plotting stays in figures.py.
"""
import pandas as pd
import numpy as np
from pathlib import Path


def load_sii_pub_total(path, sheet=0):
    """Load SII PUB_Total.xlsb — handles the double header (row 6 + row 7).

    The SII sheet has:
      row 6: group labels (GC, 2a Cat., Consolidado) merged across 3 cols each
      row 7: sub-headers (Año Comercial, Tramo, N° Personas, Renta..., Impuesto...)
      row 8+: data (one row per año x tramo)

    We forward-fill row 6 into row 7 to build unique column names and
    then filter the Consolidado block for BFM (N + Renta).

    Args:
        path (str|Path): Path to PUB_Total.xlsb
        sheet (int|str): Sheet index/name.

    Returns:
        tuple[pd.DataFrame, pd.DataFrame]: (raw, cleaned)
          cleaned columns: ['año_comercial','tramo',
            'gc_n','gc_renta','gc_impuesto',
            'cat2_n','cat2_renta','cat2_impuesto',
            'cons_n','cons_renta','cons_impuesto']
    """
    p = Path(path)
    engine = "pyxlsb" if p.suffix == ".xlsb" else None
    raw = pd.read_excel(p, sheet_name=sheet, header=None, engine=engine)

    # The real header is row 7 (0-indexed), groups in row 6
    # Row 6 has gaps (NaN) due to merged cells — forward-fill to spread group label
    if len(raw) < 9:
        return raw, None

    grp = raw.iloc[6].ffill().astype(str)  # e.g. "Per. Naturales contribuyentes de GC" repeated 3x, then "Consolidado"
    sub = raw.iloc[7].astype(str)  # "Año Comercial", "Tramo de Rentas", "N° de Personas", ...

    # Build unique names: "<group> | <sub>" except for first two cols which are standalone
    cols = []
    for g, s in zip(grp, sub):
        g = str(g).strip()
        s = str(s).strip()
        if s in ["Año Comercial", "Tramo de Rentas"]:
            cols.append(s)
        elif g.lower() == "nan" or g == "":
            cols.append(s)
        else:
            cols.append(f"{g} | {s}")

    # Simplify names for the 3 blocks
    rename = {}
    for c in cols:
        lc = c.lower()
        if "consolidado" in lc and "n°" in lc:
            rename[c] = "cons_n"
        elif "consolidado" in lc and "renta" in lc:
            rename[c] = "cons_renta"
        elif "consolidado" in lc and "impuesto" in lc:
            rename[c] = "cons_impuesto"
        elif "contribuyentes de gc" in lc and "n°" in lc:
            rename[c] = "gc_n"
        elif "contribuyentes de gc" in lc and "renta" in lc:
            rename[c] = "gc_renta"
        elif "contribuyentes de gc" in lc and "impuesto" in lc:
            rename[c] = "gc_impuesto"
        elif "2a cat" in lc and "n°" in lc:
            rename[c] = "cat2_n"
        elif "2a cat" in lc and "renta" in lc:
            rename[c] = "cat2_renta"
        elif "2a cat" in lc and "impuesto" in lc:
            rename[c] = "cat2_impuesto"
        elif c == "Año Comercial":
            rename[c] = "año_comercial"
        elif c == "Tramo de Rentas":
            rename[c] = "tramo"

    data = raw.iloc[8:].copy()
    data.columns = cols
    data = data.rename(columns=rename)

    # Keep only the tidy columns we need
    keep = ["año_comercial", "tramo", "gc_n", "gc_renta", "gc_impuesto",
            "cat2_n", "cat2_renta", "cat2_impuesto", "cons_n", "cons_renta", "cons_impuesto"]
    keep = [k for k in keep if k in data.columns]
    cleaned = data[keep].copy()

    # Types: año as int, numerics as float (MM$)
    cleaned["año_comercial"] = pd.to_numeric(cleaned["año_comercial"], errors="coerce").astype("Int64")
    for c in keep[2:]:
        if c in cleaned.columns:
            cleaned[c] = pd.to_numeric(cleaned[c], errors="coerce")

    # Drop empty rows (totally NaN)
    cleaned = cleaned.dropna(subset=["año_comercial", "tramo"], how="all")
    cleaned = cleaned[cleaned["año_comercial"].notna()]

    return raw, cleaned


def sii_for_at(cleaned, at_comercial=2017):
    """Filter cleaned SII table for one commercial year (AT = comercial+1).

    Args:
        cleaned (pd.DataFrame): from load_sii_pub_total
        at_comercial (int): e.g. 2017 -> AT 2018

    Returns:
        pd.DataFrame: rows for that year, sorted by tramo
    """
    return cleaned[cleaned["año_comercial"] == at_comercial].copy()


def sii_summary_for_report(cleaned, at_comercial=2017):
    """Quick summary for LaTeX: totals and top bracket.

    Returns dict with total taxpayers, total rent, top bracket share.
    """
    df = sii_for_at(cleaned, at_comercial)
    if df.empty:
        return {}
    total_n = df["cons_n"].sum()
    total_r = df["cons_renta"].sum()
    top = df.iloc[-1]  # last tramo = "150 y más" / top
    return {
        "año_comercial": at_comercial,
        "at": at_comercial + 1,
        "total_n": int(total_n),
        "total_renta_mm": float(total_r),
        "top_tramo": str(top["tramo"]),
        "top_n": int(top["cons_n"]),
        "top_renta_mm": float(top["cons_renta"]),
        "top_n_share": float(top["cons_n"] / total_n) if total_n else np.nan,
        "top_renta_share": float(top["cons_renta"] / total_r) if total_r else np.nan,
    }
