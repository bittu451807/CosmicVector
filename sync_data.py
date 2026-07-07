import pandas as pd
import numpy as np
import os

def sync_datasets():
    path_hel1os = 'data/Master_HEL1OS.csv'
    path_solexs = 'data/Master_SoLEXS.csv'
    output_path = 'data/synchronized_telemetry.csv'
    
    if not os.path.exists(path_hel1os) or not os.path.exists(path_solexs):
        print("Waiting for data files in the 'data/' folder...")
        return

    print("1. Loading raw PRADAN CSVs...")
    df_h = pd.read_csv(path_hel1os)
    df_s = pd.read_csv(path_solexs)

    print("2. Normalizing Timestamps (Eliminating 1970 Error)...")
    # If the timestamps are large floats, treat them as Unix seconds.
    df_h['time'] = pd.to_datetime(df_h['time'], unit='s', origin='unix')
    df_s['time'] = pd.to_datetime(df_s['time'], unit='s', origin='unix')

    # If the year is still 1970, it means it's a relative time (seconds of day).
    # We anchor HEL1OS to the SoLEXS start date.
    if df_h['time'].dt.year.min() == 1970 and df_s['time'].dt.year.min() > 2020:
        anchor = df_s['time'].min()
        df_h['time'] = anchor + pd.to_timedelta(df_h['time'].dt.hour * 3600 + df_h['time'].dt.minute * 60 + df_h['time'].dt.second, unit='s')

    print("3. Fusing sensors (5-second tolerance)...")
    df_h = df_h.sort_values('time')
    df_s = df_s.sort_values('time')
    df_unified = pd.merge_asof(df_s, df_h, on='time', direction='nearest', tolerance=pd.Timedelta('5s'))

    df_unified['hard_flux'] = df_unified['hard_flux'].ffill().bfill()
    df_unified['soft_flux'] = df_unified['soft_flux'].ffill().bfill()
    df_unified = df_unified.dropna(subset=['hard_flux', 'soft_flux'])

    df_unified.to_csv(output_path, index=False)
    print(f"✅ Sync complete! Real Date Range: {df_unified['time'].min().date()} to {df_unified['time'].max().date()}")

if __name__ == "__main__":
    sync_datasets()
