import glob
import os
import sys
import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # always run with the scripts folder as working directory

# W44C folder = parent of the scripts folder (holds logs/ and output/)
BASE = os.path.dirname(SCRIPT_DIR)
csv_files = sorted(glob.glob(os.path.join(BASE, "logs", "sensor_data*.csv")))
if not csv_files:
    raise SystemExit("No log files found")

combined_df = pd.concat(
    (pd.read_csv(f).assign(source_file=f) for f in csv_files),
    ignore_index=True,
)

# timestamps are yyyy-mm-dd hh:mm:ss (older dd/mm/yy rows also accepted); corrupt rows are dropped
ts = pd.to_datetime(combined_df["timestamp"], format="%Y-%m-%d %H:%M:%S", errors="coerce")
ts = ts.fillna(pd.to_datetime(combined_df["timestamp"], format="%d/%m/%y %H:%M:%S", errors="coerce"))
combined_df["timestamp"] = ts
n_bad = combined_df["timestamp"].isna().sum()
combined_df = combined_df.dropna(subset=["timestamp"]).reset_index(drop=True)

print(f"Combined {len(csv_files)} files -> {combined_df.shape[0]} rows (dropped {n_bad} corrupt rows)", file=sys.stderr)

# last 'bat' value for each device
sorted_df = combined_df.sort_values("timestamp")
device_id = sorted_df["device"].str.extract(r"(\d+)", expand=False).astype("Int64")

device_df = (
    sorted_df.groupby(device_id.rename("device"))
    .agg(
        rows=("timestamp", "size"),
        first_timestamp=("timestamp", "min"),
        last_timestamp=("timestamp", "max"),
        last_bat=("bat", "last"),
	co2=("co2", "last"),
        temp=("temp", "last"),
        hum=("hum", "last"),
    )
    .reset_index()
)

intervals = (device_df["rows"] - 1).clip(lower=1)
freq_secs = ((device_df["last_timestamp"] - device_df["first_timestamp"]).dt.total_seconds() / intervals).round().astype(int)
device_df["avg_interval"] = (freq_secs // 60).astype(str).str.zfill(2) + ":" + (freq_secs % 60).astype(str).str.zfill(2)
device_df = device_df[["device", "rows", "first_timestamp", "last_timestamp", "avg_interval",
                       "co2", "temp", "hum", "last_bat"]]

out = device_df.copy()
for c in ("first_timestamp", "last_timestamp"):
    out[c] = out[c].dt.strftime("%Y-%m-%d %H:%M:%S")

out.to_csv(os.path.join(SCRIPT_DIR, "device_summary.csv"), index=False)
print(out.to_json(orient="records"))