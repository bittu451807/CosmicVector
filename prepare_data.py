import pandas as pd
import numpy as np

def build_training_tensors(seq_length=60, forecast_mins=45):
    print("Loading synchronized telemetry...")
    df = pd.read_csv('data/synchronized_telemetry.csv', parse_dates=['time'])
    
    print("Extracting physical features...")
    dt = df['time'].diff().dt.total_seconds().fillna(1.0)
    df['heating_slope'] = df['soft_flux'].diff().fillna(0.0) / dt
    df['hardness_ratio'] = df['hard_flux'] / (df['soft_flux'] + 1e-8)
    df['neupert_proxy'] = df['hard_flux'] - df['heating_slope']
    
    print("Calculating Target Future Max Flux...")
    window_sec = forecast_mins * 60
    df['future_peak'] = df['soft_flux'].iloc[::-1].rolling(window=window_sec, min_periods=1).max().iloc[::-1]
    
    # REALITY CHECK: Using statistical percentiles instead of fake GOES W/m^2
    thresh_1 = df['future_peak'].quantile(0.90) # Top 10%
    thresh_2 = df['future_peak'].quantile(0.97) # Top 3%
    thresh_3 = df['future_peak'].quantile(0.995) # Top 0.5% (X-Class equivalent)

    cond = [
        (df['future_peak'] >= thresh_3),
        (df['future_peak'] >= thresh_2) & (df['future_peak'] < thresh_3),
        (df['future_peak'] >= thresh_1) & (df['future_peak'] < thresh_2)
    ]
    df['activity_level'] = np.select(cond, [3, 2, 1], default=0)
    
    print("Slicing time-series matrices...")
    features = ['soft_flux', 'hard_flux', 'heating_slope', 'hardness_ratio', 'neupert_proxy']
    feature_data = df[features].values.astype(np.float32)
    
    num_samples = len(feature_data) - seq_length
    X_arr = np.lib.stride_tricks.as_strided(
        feature_data, shape=(num_samples, seq_length, len(features)), 
        strides=(feature_data.strides[0], feature_data.strides[0], feature_data.strides[1])
    )
    y_arr = df['activity_level'].values[seq_length:].astype(np.int64)
    
    np.save('data/X_train.npy', np.ascontiguousarray(X_arr))
    np.save('data/y_train.npy', y_arr)
    df.to_csv('data/dashboard_feed.csv', index=False)
    print("\n✅ Tensors and UI Feed generated.")

if __name__ == "__main__":
    build_training_tensors()
