"""
PepMerge 数据集分布分析脚本

分析 data/pepmerge/ 目录下原始 PDB 数据的分布情况：
- 序列长度分布
- 氨基酸组成
- CA 原子间距（肽段连续性）
- 二面角分布 (phi, psi, omega)
- 手性检测 (L-peptide vs D-peptide)
- 坐标范围与中心分布
- 受体/肽段比例
"""

import os
import sys
import argparse
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.data.parsers import parse_pdb
from src.data.pep_constants import AA, BBHeavyAtom

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 13,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 9,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.linewidth': 0.8,
    'axes.spines.top': False,
    'axes.spines.right': False,
})

sns_palette = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#3B1F2B']


def compute_dihedral(a, b, c, d):
    """计算四个原子形成的二面角 (radians)"""
    b1 = b - a
    b2 = c - b
    b3 = d - c

    b1_x_b2 = np.cross(b1, b2)
    b2_x_b3 = np.cross(b2, b3)

    b1_x_b2_norm = b1_x_b2 / (np.linalg.norm(b1_x_b2) + 1e-8)
    b2_x_b3_norm = b2_x_b3 / (np.linalg.norm(b2_x_b3) + 1e-8)

    cos_angle = np.dot(b1_x_b2_norm, b2_x_b3_norm)
    cos_angle = np.clip(cos_angle, -1.0, 1.0)

    sign = np.sign(np.dot(b1, b2_x_b3_norm))
    angle = np.arccos(cos_angle) * sign

    return angle


def compute_backbone_torsions(pos, mask):
    """从原子坐标计算 backbone 二面角 (phi, psi, omega)

    标准定义:
      phi(i) = dihedral(C(i-1), N(i), CA(i), C(i))
      psi(i) = dihedral(N(i), CA(i), C(i), N(i+1))
      omega(i) = dihedral(CA(i), C(i), N(i+1), CA(i+1))
    """
    n_atoms = pos.shape[0]
    phis, psis, omegas = [], [], []

    for i in range(1, n_atoms - 1):
        if i == 0 or i >= n_atoms - 1:
            continue

        has_phi = (mask[i-1, BBHeavyAtom.C] and mask[i, BBHeavyAtom.N] and
                   mask[i, BBHeavyAtom.CA] and mask[i, BBHeavyAtom.C])
        if not has_phi:
            continue

        C_prev = pos[i-1, BBHeavyAtom.C]
        N = pos[i, BBHeavyAtom.N]
        CA = pos[i, BBHeavyAtom.CA]
        C = pos[i, BBHeavyAtom.C]

        try:
            phi = compute_dihedral(C_prev, N, CA, C)
            phis.append(phi)
        except:
            pass

    for i in range(1, n_atoms - 1):
        has_psi = (mask[i, BBHeavyAtom.N] and mask[i, BBHeavyAtom.CA] and
                   mask[i, BBHeavyAtom.C] and mask[i+1, BBHeavyAtom.N])
        if not has_psi:
            continue

        N = pos[i, BBHeavyAtom.N]
        CA = pos[i, BBHeavyAtom.CA]
        C = pos[i, BBHeavyAtom.C]
        N_next = pos[i+1, BBHeavyAtom.N]

        try:
            psi = compute_dihedral(N, CA, C, N_next)
            psis.append(psi)
        except:
            pass

    for i in range(1, n_atoms - 1):
        has_omega = (mask[i, BBHeavyAtom.CA] and mask[i, BBHeavyAtom.C] and
                     mask[i+1, BBHeavyAtom.N] and mask[i+1, BBHeavyAtom.CA])
        if not has_omega:
            continue

        CA = pos[i, BBHeavyAtom.CA]
        C = pos[i, BBHeavyAtom.C]
        N_next = pos[i+1, BBHeavyAtom.N]
        CA_next = pos[i+1, BBHeavyAtom.CA]

        try:
            omega = compute_dihedral(CA, C, N_next, CA_next)
            omegas.append(omega)
        except:
            pass

    return np.array(phis), np.array(psis), np.array(omegas)


def detect_chirality(pos, mask, aa):
    """检测手性: L-peptide vs D-peptide
    通过检测 CB 原子相对于 N-CA-C 平面的一致性来判断
    返回: 1 (L), -1 (D), 0 (unknown)
    """
    for i in range(len(aa)):
        if aa[i] >= 20:
            continue
        if not (mask[i, BBHeavyAtom.N] and mask[i, BBHeavyAtom.CA] and
                mask[i, BBHeavyAtom.C] and mask[i, BBHeavyAtom.CB]):
            continue

        N = pos[i, BBHeavyAtom.N]
        CA = pos[i, BBHeavyAtom.CA]
        C = pos[i, BBHeavyAtom.C]
        CB = pos[i, BBHeavyAtom.CB]

        v1 = N - CA
        v2 = C - CA
        normal = np.cross(v1, v2)

        cb_dir = CB - CA
        chirality = np.dot(normal, cb_dir)

        if chirality > 0:
            return 1
        else:
            return -1

    return 0


def plot_distribution(pep_lengths, rec_lengths, aa_counts, ca_distances, all_ca, output_dir):
    """绘制并保存所有分布图"""
    os.makedirs(output_dir, exist_ok=True)

    if pep_lengths is not None and len(pep_lengths) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(pep_lengths, bins=40, color='#2E86AB', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(np.mean(pep_lengths), color='#E63946', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(pep_lengths):.1f}')
        ax.axvline(np.median(pep_lengths), color='#F4A261', linestyle=':', linewidth=1.5, label=f'Median = {np.median(pep_lengths):.0f}')
        ax.set_xlabel('Peptide Length (residues)')
        ax.set_ylabel('Count')
        ax.set_title('Peptide Length Distribution')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '01_peptide_length.png'))
        plt.close()

    if rec_lengths is not None and len(rec_lengths) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(rec_lengths, bins=40, color='#2E86AB', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(np.mean(rec_lengths), color='#E63946', linestyle='--', linewidth=1.5, label=f'Mean = {np.mean(rec_lengths):.1f}')
        ax.axvline(np.median(rec_lengths), color='#F4A261', linestyle=':', linewidth=1.5, label=f'Median = {np.median(rec_lengths):.0f}')
        ax.set_xlabel('Receptor Length (residues)')
        ax.set_ylabel('Count')
        ax.set_title('Receptor Length Distribution')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '02_receptor_length.png'))
        plt.close()

    total_aa = sum(aa_counts.values())
    if total_aa > 0:
        fig, ax = plt.subplots(figsize=(10, 5))
        sorted_aa = sorted(aa_counts.items(), key=lambda x: -x[1])
        aa_names = [AA(aa_idx).name if aa_idx < 20 else "UNK" for aa_idx, _ in sorted_aa]
        aa_freqs = [count / total_aa * 100 for _, count in sorted_aa]
        colors = [sns_palette[i % len(sns_palette)] for i in range(len(aa_names))]
        bars = ax.bar(aa_names, aa_freqs, color=colors, edgecolor='white', linewidth=0.5)
        ax.set_xlabel('Amino Acid')
        ax.set_ylabel('Frequency (%)')
        ax.set_title('Amino Acid Composition (Peptide)')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        for bar, freq in zip(bars, aa_freqs):
            if freq > 3:
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2,
                        f'{freq:.1f}', ha='center', va='bottom', fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '03_amino_acid_composition.png'))
        plt.close()

    if ca_distances is not None and len(ca_distances) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.hist(ca_distances, bins=60, color='#6B4C9A', edgecolor='white', linewidth=0.5, alpha=0.85)
        ax.axvline(3.8, color='#2D6A4F', linestyle='--', linewidth=1.5, label='Ideal ~3.8 A')
        ax.axvline(np.mean(ca_distances), color='#E63946', linestyle=':', linewidth=1.5, label=f'Mean = {np.mean(ca_distances):.2f} A')
        ax.set_xlabel('CA-CA Distance (A)')
        ax.set_ylabel('Count')
        ax.set_title('CA Atom Distance Distribution (Adjacent Residues)')
        ax.legend(frameon=False)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, 15)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '04_ca_distance.png'))
        plt.close()

        outliers = ca_distances[ca_distances > 6.0]
        if len(outliers) > 0:
            fig, ax = plt.subplots(figsize=(7, 5))
            ax.hist(outliers, bins=40, color='#E63946', edgecolor='white', linewidth=0.5, alpha=0.85)
            ax.set_xlabel('CA-CA Distance (A)')
            ax.set_ylabel('Count')
            ax.set_title(f'Large CA Gaps (>6A): {len(outliers)} ({len(outliers)/len(ca_distances)*100:.1f}%)')
            ax.grid(True, alpha=0.3, linestyle='--')
            ax.set_xlim(6, min(20, np.max(outliers) + 1))
            plt.tight_layout()
            plt.savefig(os.path.join(output_dir, '05_ca_gaps_large.png'))
            plt.close()

    if pep_lengths is not None and rec_lengths is not None:
        fig, ax = plt.subplots(figsize=(6, 5))
        bp = ax.boxplot([pep_lengths, rec_lengths], tick_labels=['Peptide', 'Receptor'],
                        patch_artist=True, widths=0.6)
        colors_box = ['#2E86AB', '#A23B72']
        for patch, color in zip(bp['boxes'], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        ax.set_ylabel('Length (residues)')
        ax.set_title('Length Comparison: Peptide vs Receptor')
        ax.grid(True, alpha=0.3, axis='y', linestyle='--')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '06_length_boxplot.png'))
        plt.close()

    if pep_lengths is not None and len(pep_lengths) > 0:
        fig, ax = plt.subplots(figsize=(7, 5))
        cumlen = np.sort(list(pep_lengths))
        cumprob = np.arange(1, len(cumlen) + 1) / len(cumlen)
        ax.plot(cumlen, cumprob, color='#2E86AB', linewidth=2)
        ax.fill_between(cumlen, 0, cumprob, alpha=0.2, color='#2E86AB')
        ax.set_xlabel('Peptide Length (residues)')
        ax.set_ylabel('Cumulative Probability')
        ax.set_title('Peptide Length CDF (Cumulative Distribution Function)')
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlim(0, max(cumlen) + 5)
        ax.set_ylim(0, 1.02)
        percentiles = [25, 50, 75, 90]
        for p in percentiles:
            val = np.percentile(cumlen, p)
            ax.axvline(val, color='#E63946', linestyle=':', alpha=0.7)
            ax.axhline(p/100, color='#E63946', linestyle=':', alpha=0.7)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, '07_peptide_length_cdf.png'))
        plt.close()

    print(f"Distribution plots saved to: {output_dir}/")


def plot_torsion_analysis(phis, psis, omegas, output_dir, suffix=''):
    """绘制二面角分布 (Ramachandran plot)"""
    os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    if len(phis) > 0:
        axes[0].hist(phis, bins=60, color='#2E86AB', edgecolor='white', alpha=0.85)
        axes[0].set_xlabel('Phi (radians)')
        axes[0].set_ylabel('Count')
        axes[0].set_title(f'Phi Distribution\n(n={len(phis)})')
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(np.mean(phis), color='#E63946', linestyle='--', label=f'Mean={np.mean(phis):.2f}')
        axes[0].legend()

    if len(psis) > 0:
        axes[1].hist(psis, bins=60, color='#A23B72', edgecolor='white', alpha=0.85)
        axes[1].set_xlabel('Psi (radians)')
        axes[1].set_ylabel('Count')
        axes[1].set_title(f'Psi Distribution\n(n={len(psis)})')
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(np.mean(psis), color='#E63946', linestyle='--', label=f'Mean={np.mean(psis):.2f}')
        axes[1].legend()

    if len(omegas) > 0:
        axes[2].hist(omegas, bins=60, color='#F18F01', edgecolor='white', alpha=0.85)
        axes[2].set_xlabel('Omega (radians)')
        axes[2].set_ylabel('Count')
        axes[2].set_title(f'Omega Distribution\n(n={len(omegas)})')
        axes[2].grid(True, alpha=0.3)
        axes[2].axvline(np.mean(omegas), color='#E63946', linestyle='--', label=f'Mean={np.mean(omegas):.2f}')
        axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '08_backbone_torsions.png'), dpi=150)
    plt.close()

    phis_deg = np.degrees(phis)
    psis_deg = np.degrees(psis)

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_facecolor('#F5F5DC')

    from scipy import stats
    try:
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
    except Exception:
        pass

    ax.set_xlabel('Phi (degrees)')
    ax.set_ylabel('Psi (degrees)')
    ax.set_title('Ramachandran Plot' + (' (Receptor)' if suffix else ''))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-180, 180)
    ax.axhline(0, color='gray', lw=0.5, alpha=0.5)
    ax.axvline(0, color='gray', lw=0.5, alpha=0.5)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, '09_ramachandran.png'), dpi=150)
    plt.close()

    print(f"Torsion plots saved to: {output_dir}/")


def compute_frequency_matrix(aa_counts):
    """计算氨基酸频率表"""
    total = sum(aa_counts.values())
    if total == 0:
        return {}
    return {aa_idx: count / total for aa_idx, count in aa_counts.items()}


def analyze_pepmerge_dataset(structure_dir: str, max_samples: int = None, sample_fraction: float = 1.0):
    """分析 PepMerge 数据集的分布"""

    sample_dirs = []
    for root, dirs, files in os.walk(structure_dir):
        if 'peptide.pdb' in files and 'receptor.pdb' in files:
            sample_dirs.append(root)

    print(f"找到 {len(sample_dirs)} 个样本目录")

    if max_samples and len(sample_dirs) > max_samples:
        np.random.seed(42)
        indices = np.random.choice(len(sample_dirs), max_samples, replace=False)
        sample_dirs = [sample_dirs[i] for i in indices]
        print(f"随机采样 {len(sample_dirs)} 个样本进行分析")

    if sample_fraction < 1.0:
        n = max(1, int(len(sample_dirs) * sample_fraction))
        sample_dirs = sample_dirs[:n]
        print(f"使用前 {n} 个样本 ({sample_fraction*100:.0f}%)")

    pep_lengths = []
    rec_lengths = []
    aa_counts = Counter()
    rec_aa_counts = Counter()
    ca_distances = []
    ca_positions_all = []
    all_phis, all_psis, all_omegas = [], [], []
    rec_all_phis, rec_all_psis, rec_all_omegas = [], [], []
    chirality_counts = Counter()
    skipped_samples = []

    for sample_dir in tqdm(sample_dirs, desc="分析中"):
        try:
            pep_pdb = os.path.join(sample_dir, 'peptide.pdb')
            rec_pdb = os.path.join(sample_dir, 'receptor.pdb')

            pep_data, _ = parse_pdb(pep_pdb)
            rec_data, _ = parse_pdb(rec_pdb)

            if pep_data is None or rec_data is None:
                skipped_samples.append((os.path.basename(sample_dir), 'parse_failed'))
                continue

            pep_len = len(pep_data['aa'])
            rec_len = len(rec_data['aa'])
            pep_lengths.append(pep_len)
            rec_lengths.append(rec_len)

            for aa in pep_data['aa'].tolist():
                aa_counts[aa] += 1

            for aa in rec_data['aa'].tolist():
                rec_aa_counts[aa] += 1

            pep_ca = pep_data['pos_heavyatom'][:, BBHeavyAtom.CA]
            pep_ca_mask = pep_data['mask_heavyatom'][:, BBHeavyAtom.CA]
            valid_cas = pep_ca[pep_ca_mask]
            if len(valid_cas) > 1:
                diffs = torch.norm(valid_cas[1:] - valid_cas[:-1], dim=1)
                ca_distances.extend(diffs.tolist())

            ca_positions_all.append(pep_ca[pep_ca_mask].numpy())

            phis, psis, omegas = compute_backbone_torsions(
                pep_data['pos_heavyatom'].numpy(),
                pep_data['mask_heavyatom'].numpy()
            )
            all_phis.extend(phis.tolist())
            all_psis.extend(psis.tolist())
            all_omegas.extend(omegas.tolist())

            rec_phis, rec_psis, rec_omegas = compute_backbone_torsions(
                rec_data['pos_heavyatom'].numpy(),
                rec_data['mask_heavyatom'].numpy()
            )
            rec_all_phis.extend(rec_phis.tolist())
            rec_all_psis.extend(rec_psis.tolist())
            rec_all_omegas.extend(rec_omegas.tolist())

            chirality = detect_chirality(
                pep_data['pos_heavyatom'].numpy(),
                pep_data['mask_heavyatom'].numpy(),
                pep_data['aa'].numpy()
            )
            chirality_counts[chirality] += 1

        except Exception as e:
            skipped_samples.append((os.path.basename(sample_dir), str(e)))

    pep_lengths_arr = np.array(pep_lengths)
    rec_lengths_arr = np.array(rec_lengths)
    ca_distances_arr = np.array(ca_distances) if ca_distances else np.array([])
    all_ca = np.concatenate(ca_positions_all, axis=0) if ca_positions_all else np.array([])
    phis_arr = np.array(all_phis)
    psis_arr = np.array(all_psis)
    omegas_arr = np.array(all_omegas)
    rec_phis_arr = np.array(rec_all_phis)
    rec_psis_arr = np.array(rec_all_psis)
    rec_omegas_arr = np.array(rec_all_omegas)

    script_dir = Path(__file__).parent
    base_output_dir = script_dir.parents[1] / "outputs" / "analysis"
    peptide_output_dir = base_output_dir / "peptide"
    receptor_output_dir = base_output_dir / "receptor"
    os.makedirs(peptide_output_dir, exist_ok=True)
    os.makedirs(receptor_output_dir, exist_ok=True)

    plot_distribution(pep_lengths_arr, rec_lengths_arr, aa_counts, ca_distances_arr, all_ca, str(peptide_output_dir))
    if len(phis_arr) > 0:
        plot_torsion_analysis(phis_arr, psis_arr, omegas_arr, str(peptide_output_dir), suffix='_peptide')
    if len(rec_phis_arr) > 0:
        plot_torsion_analysis(rec_phis_arr, rec_psis_arr, rec_omegas_arr, str(receptor_output_dir), suffix='_receptor')

    stats_file_pep = os.path.join(peptide_output_dir, "analysis_statistics.txt")
    with open(stats_file_pep, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PepMerge 数据集 - 肽段分布分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("【数据概览】\n")
        f.write(f"  总样本目录数: {len(sample_dirs)}\n")
        f.write(f"  有效样本数: {len(pep_lengths_arr)}\n")
        f.write(f"  跳过样本数: {len(skipped_samples)}\n\n")

        f.write("【肽段长度分布】\n")
        f.write(f"  样本数: {len(pep_lengths_arr)}\n")
        f.write(f"  均值: {pep_lengths_arr.mean():.1f}\n")
        f.write(f"  标准差: {pep_lengths_arr.std():.1f}\n")
        f.write(f"  最小值: {pep_lengths_arr.min()}\n")
        f.write(f"  最大值: {pep_lengths_arr.max()}\n")
        f.write(f"  P25: {np.percentile(pep_lengths_arr, 25):.0f}\n")
        f.write(f"  P50: {np.percentile(pep_lengths_arr, 50):.0f}\n")
        f.write(f"  P75: {np.percentile(pep_lengths_arr, 75):.0f}\n")
        f.write(f"  P90: {np.percentile(pep_lengths_arr, 90):.0f}\n\n")

        f.write("【氨基酸组成】(肽段中)\n")
        total_aa = sum(aa_counts.values())
        sorted_aa = sorted(aa_counts.items(), key=lambda x: -x[1])
        for aa_idx, count in sorted_aa:
            pct = count / total_aa * 100
            aa_name = AA(aa_idx).name if aa_idx < 20 else "UNK"
            bar = "=" * int(pct)
            f.write(f"  {aa_name:3s} ({aa_idx:2d}): {count:5d} ({pct:5.1f}%) {bar}\n")
        f.write("\n")

        if len(ca_distances_arr) > 0:
            f.write("【CA 原子间距分布】(相邻残基, 肽段内)\n")
            f.write(f"  样本数: {len(ca_distances_arr)}\n")
            f.write(f"  均值: {ca_distances_arr.mean():.3f} A\n")
            f.write(f"  P50: {np.percentile(ca_distances_arr, 50):.3f} A\n\n")

        if len(phis_arr) > 0:
            f.write("【二面角分布】(肽段)\n")
            f.write(f"  Phi (n={len(phis_arr)}): mean={np.degrees(phis_arr).mean():.1f}deg, std={np.degrees(phis_arr).std():.1f}deg\n")
            f.write(f"  Psi (n={len(psis_arr)}): mean={np.degrees(psis_arr).mean():.1f}deg, std={np.degrees(psis_arr).std():.1f}deg\n")
            f.write(f"  Omega (n={len(omegas_arr)}): mean={np.degrees(omegas_arr).mean():.1f}deg, std={np.degrees(omegas_arr).std():.1f}deg\n\n")

        if chirality_counts:
            f.write("【手性检测】\n")
            total_chiral = sum(chirality_counts.values())
            for chirality, count in sorted(chirality_counts.items()):
                label = {1: 'L-peptide', -1: 'D-peptide', 0: 'Unknown'}.get(chirality, f'Unknown({chirality})')
                pct = count / total_chiral * 100
                f.write(f"  {label}: {count} ({pct:.1f}%)\n")
            f.write("\n")

        if len(all_ca) > 0:
            f.write("\n【CA 坐标分布】\n")
            center = all_ca.mean(axis=0)
            rms = np.sqrt(((all_ca - center)**2).sum(axis=1).mean())
            f.write(f"  中心点: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})\n")
            f.write(f"  RMS 半径: {rms:.1f} A\n\n")

        f.write("【输出文件】\n")
        f.write(f"  分布图: {peptide_output_dir}/\n")

    stats_file_rec = os.path.join(receptor_output_dir, "analysis_statistics.txt")
    with open(stats_file_rec, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("PepMerge 数据集 - 受体分布分析报告\n")
        f.write("=" * 60 + "\n\n")

        f.write("【受体长度分布】\n")
        f.write(f"  样本数: {len(rec_lengths_arr)}\n")
        f.write(f"  均值: {rec_lengths_arr.mean():.1f}\n")
        f.write(f"  标准差: {rec_lengths_arr.std():.1f}\n")
        f.write(f"  范围: [{rec_lengths_arr.min()}, {rec_lengths_arr.max()}]\n")
        f.write(f"  P50: {np.percentile(rec_lengths_arr, 50):.0f}\n")
        f.write(f"  P90: {np.percentile(rec_lengths_arr, 90):.0f}\n\n")

        f.write("【氨基酸组成】(受体中)\n")
        total_rec_aa = sum(rec_aa_counts.values())
        sorted_rec_aa = sorted(rec_aa_counts.items(), key=lambda x: -x[1])
        for aa_idx, count in sorted_rec_aa:
            pct = count / total_rec_aa * 100
            aa_name = AA(aa_idx).name if aa_idx < 20 else "UNK"
            bar = "=" * int(pct)
            f.write(f"  {aa_name:3s} ({aa_idx:2d}): {count:5d} ({pct:5.1f}%) {bar}\n")
        f.write("\n")

        if len(rec_phis_arr) > 0:
            f.write("【二面角分布】(受体)\n")
            f.write(f"  Phi (n={len(rec_phis_arr)}): mean={np.degrees(rec_phis_arr).mean():.1f}deg, std={np.degrees(rec_phis_arr).std():.1f}deg\n")
            f.write(f"  Psi (n={len(rec_psis_arr)}): mean={np.degrees(rec_psis_arr).mean():.1f}deg, std={np.degrees(rec_psis_arr).std():.1f}deg\n")
            f.write(f"  Omega (n={len(rec_omegas_arr)}): mean={np.degrees(rec_omegas_arr).mean():.1f}deg, std={np.degrees(rec_omegas_arr).std():.1f}deg\n\n")

        f.write("【输出文件】\n")
        f.write(f"  分布图: {receptor_output_dir}/\n")

    np.savez(os.path.join(base_output_dir, "dataset_distribution_data.npz"),
             pep_lengths=pep_lengths_arr,
             rec_lengths=rec_lengths_arr,
             ca_distances=ca_distances_arr,
             aa_counts=np.array([aa_counts.get(i, 0) for i in range(22)]),
             rec_aa_counts=np.array([rec_aa_counts.get(i, 0) for i in range(22)]),
             phis=phis_arr,
             psis=psis_arr,
             omegas=omegas_arr,
             rec_phis=rec_phis_arr,
             rec_psis=rec_psis_arr,
             rec_omegas=rec_omegas_arr,
    )

    skipped_file = os.path.join(base_output_dir, "skipped_samples.txt")
    with open(skipped_file, 'w', encoding='utf-8') as f:
        for name, reason in skipped_samples:
            f.write(f"{name}: {reason}\n")

    print(f"Peptide statistics saved to: {stats_file_pep}")
    print(f"Receptor statistics saved to: {stats_file_rec}")
    print(f"Skipped samples saved to: {skipped_file}")
    print(f"Raw data saved to: {base_output_dir}/dataset_distribution_data.npz")

    return {
        'pep_lengths': pep_lengths_arr,
        'rec_lengths': rec_lengths_arr,
        'ca_distances': ca_distances_arr,
        'aa_counts': aa_counts,
        'rec_aa_counts': rec_aa_counts,
        'phis': phis_arr,
        'psis': psis_arr,
        'omegas': omegas_arr,
        'rec_phis': rec_phis_arr,
        'rec_psis': rec_psis_arr,
        'rec_omegas': rec_omegas_arr,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分析 PepMerge 数据集分布")
    parser.add_argument("--structure_dir", type=str, default="data/pepmerge",
                        help="PepMerge 数据目录路径")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="最大采样样本数 (默认: 全部)")
    parser.add_argument("--fraction", type=float, default=1.0,
                        help="使用样本比例 (默认: 1.0)")

    args = parser.parse_args()

    analyze_pepmerge_dataset(
        structure_dir=args.structure_dir,
        max_samples=args.max_samples,
        sample_fraction=args.fraction,
    )