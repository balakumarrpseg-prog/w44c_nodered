"""
updateSensorInfo.py

Reads the sensor log and updates SensorInfo.csv with the most recent reading
of each sensor.

    Log  (input) : ..\\logs\\sensor_data.csv   columns: device,timestamp,co2,temp,hum,bat
    SensorInfo   : SensorInfo.csv (same folder as this script)
                   columns: Sensor,x-coordinate,y-coordinate,T,H,Co2,timestamp

Mapping: log device "NWS5" -> Sensor 5 in SensorInfo.csv.
  T          <- temp
  H          <- hum
  Co2        <- co2
  timestamp  <- timestamp of that latest reading (as written in the log)

Sensors with no reading in the log keep their old values (new columns stay blank).
Devices in the log that are not in SensorInfo.csv are ignored (and listed).

Usage:
    python updateSensorInfo.py [--log PATH] [--sensors PATH]
"""
import argparse
import os
import re
import sys

import pandas as pd

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
os.chdir(SCRIPT_DIR)  # always run with the scripts folder as working directory

DEFAULT_LOG = os.path.normpath(os.path.join(SCRIPT_DIR, '..', 'logs', 'sensor_data.csv'))
DEFAULT_SENSORS = os.path.join(SCRIPT_DIR, 'SensorInfo.csv')
TS_FORMAT = '%Y-%m-%d %H:%M:%S'  # e.g. 2026-11-03 17:17:39

NEW_COLUMNS = ['T', 'H', 'Co2', 'timestamp']


def load_latest_readings(log_path):
    """Return a DataFrame indexed by sensor number with the latest reading each."""
    log = pd.read_csv(log_path, skipinitialspace=True)
    log.columns = log.columns.str.strip().str.lower()

    missing = {'device', 'timestamp', 'co2', 'temp', 'hum'} - set(log.columns)
    if missing:
        sys.exit(f'ERROR: {log_path} is missing column(s): {sorted(missing)}')

    # "NWS5" -> 5 (rows whose device name has no number are dropped)
    log['Sensor'] = pd.to_numeric(
        log['device'].astype(str).str.extract(r'(\d+)\s*$', expand=False), errors='coerce')
    log['ts_parsed'] = pd.to_datetime(log['timestamp'], format=TS_FORMAT, errors='coerce')
    for col in ('co2', 'temp', 'hum'):
        log[col] = pd.to_numeric(log[col], errors='coerce')

    bad = log['Sensor'].isna() | log['ts_parsed'].isna()
    if bad.any():
        print(f'Skipping {int(bad.sum())} unreadable log row(s).')
    log = log[~bad]

    # Most recent row per sensor
    latest = (log.sort_values('ts_parsed')
                 .groupby('Sensor', as_index=False)
                 .tail(1)
                 .set_index('Sensor'))
    latest.index = latest.index.astype(int)
    return latest


def main():
    parser = argparse.ArgumentParser(description='Update SensorInfo.csv from sensor_data.csv')
    parser.add_argument('--log', default=DEFAULT_LOG, help='Path to sensor_data.csv')
    parser.add_argument('--sensors', default=DEFAULT_SENSORS, help='Path to SensorInfo.csv')
    args = parser.parse_args()

    if not os.path.isfile(args.log):
        sys.exit(f'ERROR: log file not found: {args.log}')
    if not os.path.isfile(args.sensors):
        sys.exit(f'ERROR: SensorInfo file not found: {args.sensors}')

    latest = load_latest_readings(args.log)

    info = pd.read_csv(args.sensors, skipinitialspace=True)
    info.columns = info.columns.str.strip()
    info = info.dropna(how='all')
    info['Sensor'] = info['Sensor'].astype(int)

    # Make sure the columns exist (object dtype for timestamp so strings fit)
    for col in NEW_COLUMNS:
        if col not in info.columns:
            info[col] = pd.NA
    info['timestamp'] = info['timestamp'].astype(object)

    updated, no_data = [], []
    for idx, row in info.iterrows():
        s = int(row['Sensor'])
        if s in latest.index:
            r = latest.loc[s]
            info.at[idx, 'T'] = r['temp']
            info.at[idx, 'H'] = r['hum']
            info.at[idx, 'Co2'] = r['co2']
            info.at[idx, 'timestamp'] = r['timestamp']
            updated.append(s)
        else:
            no_data.append(s)

    unknown = sorted(set(latest.index) - set(info['Sensor']))

    # CO2 is a whole number (keeps blanks for sensors without data)
    info['Co2'] = pd.to_numeric(info['Co2'], errors='coerce').round().astype('Int64')

    # Fixed column order, any extra columns kept at the end
    base = ['Sensor', 'x-coordinate', 'y-coordinate'] + NEW_COLUMNS
    extra = [c for c in info.columns if c not in base]
    info = info[base + extra]

    # Write to a temp file then replace, so a reader never sees a half-written file
    tmp_path = args.sensors + '.tmp'
    info.to_csv(tmp_path, index=False)
    os.replace(tmp_path, args.sensors)

    print(f'Updated {len(updated)} sensor(s): {updated}')
    if no_data:
        print(f'No reading in log (old values kept): {no_data}')
    if unknown:
        print(f'In log but not in SensorInfo (ignored): {unknown}')
    print(f'Saved {args.sensors}')


if __name__ == '__main__':
    main()
