import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os

data = np.load('outputs/analysis/dataset_distribution_data.npz')

output_dir = 'outputs/analysis'

for name, phis, psis, suffix in [
    ('Peptide', data['phis'], data['psis'], ''),
    ('Receptor', data['rec_phis'], data['rec_psis'], '_receptor')
]:
    phis_deg = np.degrees(phis)
    psis_deg = np.degrees(psis)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#F5F5DC')

    xmin, xmax = -180, 180
    ymin, ymax = -180, 180
    xx, yy = np.mgrid[xmin:xmax:200j, ymin:ymax:200j]
    positions = np.vstack([xx.ravel(), yy.ravel()])
    values = np.vstack([phis_deg, psis_deg])
    kernel = stats.gaussian_kde(values)
    f = np.reshape(kernel(positions).T, xx.shape)

    levels = np.percentile(f.ravel(), [20, 40, 60, 80, 95])
    cmap = plt.cm.Blues
    ax.contourf(xx, yy, f, levels=50, cmap=cmap, alpha=0.9)
    ax.contour(xx, yy, f, levels=levels, colors='#D2B48C', linewidths=[1.0, 1.2, 1.5, 1.8, 2.0])

    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Psi (degrees)')
    ax.set_title(f'Ramachandran Plot ({name})')
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'09_ramachandran{suffix}.png'), dpi=150)
    plt.close()
    print(f'Saved: outputs/analysis/09_ramachandran{suffix}.png')