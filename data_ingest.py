"""
data_ingest.py
================
COSMIC VECTOR — Real ISRO Payload Ingestion Engine
----------------------------------------------------
This module replaces the old "robust_clean_upload()" CSV-only parser.

The judges' actual Aditya-L1 data does NOT arrive as clean CSVs — it arrives as:

  • SoLEXS  ->  AL1_SLX_L1_<date>_v1_0.zip
                   SDD1/ or SDD2/ *.lc.gz   (gzipped FITS "RATE" light-curve, cols: TIME, COUNTS)

  • HEL1OS  ->  HLS_<date>_<dur>sec_lev1_V2xx.zip
                   czt/lightcurve_czt1.fits / czt2.fits
                   Several BinTableHDUs, one per energy band, cols: MJD, ISOT, CTR, STAT_ERR

This module opens the zip in-memory, finds the right extension automatically,
decodes the FITS binary tables with astropy, converts on-board mission time to
real UTC timestamps, and hands back a clean 2-column DataFrame — exactly the
shape the rest of the dashboard already expects. A generic CSV/zip-of-CSV path
is kept as a fallback so hand-made judge test files still work too.
"""

import io
import gzip
import re
import zipfile

import numpy as np
import pandas as pd

try:
    from astropy.io import fits
    ASTROPY_OK = True
except ImportError:
    ASTROPY_OK = False

MJD_UNIX_EPOCH = 40587.0  # MJD value of 1970-01-01T00:00:00 UTC


# ----------------------------------------------------------------------
# Low level helpers
# ----------------------------------------------------------------------
def _maybe_gunzip(name: str, raw: bytes) -> bytes:
    if name.lower().endswith(".gz"):
        try:
            return gzip.decompress(raw)
        except OSError:
            return raw
    return raw


def _mjd_to_utc(mjd_array):
    unix_seconds = (np.asarray(mjd_array, dtype="float64") - MJD_UNIX_EPOCH) * 86400.0
    return pd.to_datetime(unix_seconds, unit="s", utc=True)


# ----------------------------------------------------------------------
# Real instrument parsers
# ----------------------------------------------------------------------
def parse_solexs_zip(raw_zip_bytes: bytes):
    """AL1_SLX_L1_*.zip -> DataFrame[time, soft_flux]  (Soft X-ray, SoLEXS)"""
    if not ASTROPY_OK:
        return None
    with zipfile.ZipFile(io.BytesIO(raw_zip_bytes)) as z:
        candidates = [
            n for n in z.namelist()
            if re.search(r"\.lc(\.gz)?$", n, re.IGNORECASE) and "__MACOSX" not in n
        ]
        if not candidates:
            return None
        # Prefer SDD2 (secondary detector, typically the cleaner science channel)
        candidates.sort(key=lambda n: ("sdd2" not in n.lower(), n))
        raw = _maybe_gunzip(candidates[0], z.read(candidates[0]))

    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
        hdu = hdul["RATE"] if "RATE" in hdul else hdul[1]
        data, hdr = hdu.data, hdu.header
        t_raw = np.asarray(data["TIME"], dtype="float64")
        counts = np.asarray(data["COUNTS"], dtype="float64")

        mjdrefi = float(hdr.get("MJDREFI", MJD_UNIX_EPOCH))
        mjdreff = float(hdr.get("MJDREFF", 0.0))
        ref_unix_s = (mjdrefi + mjdreff - MJD_UNIX_EPOCH) * 86400.0
        time = pd.to_datetime(ref_unix_s + t_raw, unit="s", utc=True)

    df = pd.DataFrame({"time": time, "soft_flux": counts})
    return df.dropna().sort_values("time").reset_index(drop=True)


def parse_hel1os_zip(raw_zip_bytes: bytes):
    """HLS_*.zip -> DataFrame[time, hard_flux]  (Hard X-ray, HEL1OS/CZT)"""
    if not ASTROPY_OK:
        return None
    with zipfile.ZipFile(io.BytesIO(raw_zip_bytes)) as z:
        candidates = [
            n for n in z.namelist()
            if "lightcurve_czt" in n.lower() and n.lower().endswith(".fits")
        ]
        if not candidates:
            return None
        candidates.sort()
        raw = z.read(candidates[0])

    with fits.open(io.BytesIO(raw), memmap=False) as hdul:
        # Pick the widest energy band extension (broadband channel, e.g. 18-160 keV)
        best_hdu, best_span = None, -1.0
        for hdu in hdul[1:]:
            name = (hdu.name or "").upper()
            m = re.findall(r"([\d.]+)KEV_TO_([\d.]+)KEV", name)
            if m:
                lo, hi = float(m[0][0]), float(m[0][1])
                if (hi - lo) > best_span:
                    best_span, best_hdu = hi - lo, hdu
        hdu = best_hdu or hdul[1]
        data = hdu.data
        cols = hdu.columns.names

        if "ISOT" in cols:
            time = pd.to_datetime(np.asarray(data["ISOT"]).astype(str), utc=True, errors="coerce")
        elif "MJD" in cols:
            time = _mjd_to_utc(data["MJD"])
        else:
            return None
        ctr = np.asarray(data["CTR"], dtype="float64") if "CTR" in cols else np.asarray(data[cols[-1]], dtype="float64")

    df = pd.DataFrame({"time": time, "hard_flux": ctr})
    return df.dropna().sort_values("time").reset_index(drop=True)


# ----------------------------------------------------------------------
# Generic CSV fallback (hand-built judge test files, non-FITS payloads)
# ----------------------------------------------------------------------
def _finalize_csv_df(df_up: pd.DataFrame, target_col_name: str):
    df_up.columns = df_up.columns.str.strip().str.lower()
    time_cols = [c for c in df_up.columns if "time" in c or "date" in c or "ts" in c]
    tcol = time_cols[0] if time_cols else df_up.columns[0]
    df_up = df_up.rename(columns={tcol: "time"})

    parsed_time = pd.to_datetime(df_up["time"], utc=True, errors="coerce")
    if parsed_time.notna().mean() < 0.5:
        numeric_time = pd.to_numeric(df_up["time"], errors="coerce")
        if numeric_time.median(skipna=True) > 1e8:
            parsed_time = pd.to_datetime(numeric_time, unit="s", utc=True)
        else:
            parsed_time = pd.to_datetime("2026-06-11", utc=True) + pd.to_timedelta(numeric_time, unit="s")
    df_up["time"] = parsed_time
    df_up = df_up.dropna(subset=["time"])

    num_cols = df_up.select_dtypes(include=[np.number]).columns.tolist()
    flux_cols = [c for c in num_cols if c != "time"]
    if not flux_cols:
        return None
    df_up = df_up.rename(columns={flux_cols[0]: target_col_name})
    return df_up[["time", target_col_name]].sort_values("time").reset_index(drop=True)


def parse_generic_payload(raw_bytes: bytes, filename: str, target_col_name: str):
    try:
        if filename.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(raw_bytes)) as z:
                valid = [f for f in z.namelist() if not f.startswith("__MACOSX") and f.lower().endswith(".csv")]
                if not valid:
                    return None
                with z.open(valid[0]) as f:
                    df_up = pd.read_csv(f, comment="#")
        else:
            df_up = pd.read_csv(io.BytesIO(raw_bytes), comment="#")
        return _finalize_csv_df(df_up, target_col_name)
    except Exception:
        return None


# ----------------------------------------------------------------------
# Public entrypoint
# ----------------------------------------------------------------------
def load_instrument_upload(uploaded_file, kind: str):
    """
    kind: 'solexs' -> soft X-ray channel   |   'hel1os' -> hard X-ray channel
    Auto-detects real ISRO FITS-zip payloads first, falls back to CSV/zip-of-CSV.
    Returns a DataFrame[time, <soft_flux|hard_flux>] or None.
    """
    target_col = "soft_flux" if kind == "solexs" else "hard_flux"
    raw = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()
    name = uploaded_file.name

    df = None
    if name.lower().endswith(".zip"):
        try:
            df = parse_solexs_zip(raw) if kind == "solexs" else parse_hel1os_zip(raw)
        except Exception:
            df = None

    if df is None or df.empty:
        df = parse_generic_payload(raw, name, target_col)

    return df


def sync_and_engineer(df_sol: pd.DataFrame, df_hel: pd.DataFrame) -> pd.DataFrame:
    """Time-synchronise both channels onto one clock and derive all AI features."""
    # Different FITS/CSV sources can yield different datetime64 resolutions
    # (s / us / ns). merge_asof requires an exact dtype match, so normalise both.
    df_sol = df_sol.copy()
    df_hel = df_hel.copy()
    df_sol["time"] = pd.to_datetime(df_sol["time"], utc=True).astype("datetime64[ns, UTC]")
    df_hel["time"] = pd.to_datetime(df_hel["time"], utc=True).astype("datetime64[ns, UTC]")

    m = pd.merge_asof(
        df_sol.sort_values("time"),
        df_hel.sort_values("time"),
        on="time",
        direction="nearest",
        tolerance=pd.Timedelta("5s"),
    ).ffill().bfill()

    # If the two uploaded files don't share any real overlap, ffill/bfill will
    # leave an entirely-empty column — fall back to 0 rather than propagate NaN.
    m["soft_flux"] = m["soft_flux"].fillna(0.0)
    m["hard_flux"] = m["hard_flux"].fillna(0.0)

    dt = m["time"].diff().dt.total_seconds().fillna(1.0)
    dt = dt.replace(0, 1.0)

    m["heating_slope"] = m["soft_flux"].diff().fillna(0.0) / dt
    m["hardness_ratio"] = m["hard_flux"] / (m["soft_flux"] + 1e-8)
    m["neupert_proxy"] = m["hard_flux"] - m["heating_slope"]
    m["activity_level"] = 0
    return m


def overlap_diagnostics(df_sol: pd.DataFrame, df_hel: pd.DataFrame) -> dict:
    """Quick sanity check so the UI can warn the user if the two uploaded
    instrument files don't actually cover the same time window."""
    sol_range = (df_sol["time"].min(), df_sol["time"].max())
    hel_range = (df_hel["time"].min(), df_hel["time"].max())
    latest_start = max(sol_range[0], hel_range[0])
    earliest_end = min(sol_range[1], hel_range[1])
    overlap_seconds = max(0.0, (earliest_end - latest_start).total_seconds())
    return {
        "solexs_range": sol_range,
        "hel1os_range": hel_range,
        "overlap_seconds": overlap_seconds,
        "has_overlap": overlap_seconds > 0,
    }
