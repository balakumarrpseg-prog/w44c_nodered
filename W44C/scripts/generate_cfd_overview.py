"""
Plot the CFD-only heatmaps of Temperature, Humidity and CO2 together in ONE image
(three panels stacked top to bottom), from CFD1_data.csv.

Output: heatmap_cfd_all.png

Units: CFD temperature is Kelvin (shown in C), CFD humidity is a fraction (shown in %).
Colour scales come from QUANTITIES in generate_heatmaps_cfd1.py.

Usage:
    python generate_cfd_overview.py [--cfd CFD1_data.csv] [--outdir DIR]
"""
import argparse
import os

import matplotlib.pyplot as plt
import pandas as pd

from generate_heatmaps import save_figure
from generate_heatmaps_cfd1 import CMAP, QUANTITIES


def main():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(script_dir)  # always run with the scripts folder as working directory

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cfd', default=os.path.join(script_dir, 'CFD1_data.csv'), help='Path to CFD1_data.csv')
    default_out = os.path.normpath(os.path.join(script_dir, '..', '..', 'static'))
    parser.add_argument('--outdir', default=default_out, help='Directory to save the image')
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    cfd_df = pd.read_csv(args.cfd, skipinitialspace=True)
    cfd_df.columns = cfd_df.columns.str.strip()

    fig, axes = plt.subplots(len(QUANTITIES), 1, figsize=(10, 12), constrained_layout=True)

    for ax, (name, cfd_col, convert, _sensor_col, unit, (mn, mx)) in zip(axes, QUANTITIES.values()):
        sc = ax.scatter(cfd_df['x-coordinate'], cfd_df['y-coordinate'], c=convert(cfd_df[cfd_col].to_numpy()),
                        cmap=CMAP, s=28, vmin=mn, vmax=mx)
        fig.colorbar(sc, ax=ax, orientation='horizontal', location='bottom', pad=0.02,
                     fraction=0.05, shrink=0.4, aspect=40)
        ax.set_xticks([])  # no ticks / labels on either axis
        ax.set_yticks([])
        ax.set_aspect('equal')
        ax.margins(x=0.03, y=0.04)
        ax.set_title(f'{name}_cfd', fontsize=16, fontweight='bold', loc='left', pad=10)

    out_path = os.path.join(args.outdir, 'heatmap_cfd_all.png')
    save_figure(fig, out_path)  # cropped to content
    plt.close(fig)
    print(f'Saved {out_path}')


if __name__ == '__main__':
    main()
