"""
镜像 PepMerge 数据集生成 D-肽版本 (XY 镜像)

将 data/pepmerge/ 下的 peptide.pdb 和 receptor.pdb 进行 XY 镜像，
保存到 data/pepmerge_mirrored/

镜像规则 (与 D-Flow 一致):
- 位置坐标: (x, y, z) -> (-x, -y, z)

这不是真正的 D-氨基酸，而是坐标变换模拟 D-肽手性。
"""

import os
import shutil
from pathlib import Path


def mirror_pdb_parity(input_path: str, output_path: str):
    """镜像 PDB 文件 - 使用论文 parity 规则: 空间反演 (x, y, z) -> (-x, -y, -z)

    Args:
        input_path: 输入 PDB 路径
        output_path: 输出 PDB 路径
    """
    with open(input_path, 'r') as f:
        lines = f.readlines()

    mirrored_lines = []
    for line in lines:
        # PDB 坐标行格式: ATOM/HETATM x y z ...
        if line.startswith(('ATOM', 'HETATM', 'ANISOU')):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])

                x_mirrored = -x
                y_mirrored = -y
                z_mirrored = -z

                line = line[:30] + f'{x_mirrored:8.3f}' + f'{y_mirrored:8.3f}' + f'{z_mirrored:8.3f}' + line[54:]
            except ValueError:
                pass
        mirrored_lines.append(line)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as f:
        f.writelines(mirrored_lines)


def process_sample(sample_dir: str, output_base: str):
    """处理单个样本目录"""
    sample_name = os.path.basename(sample_dir)
    output_dir = os.path.join(output_base, sample_name)
    os.makedirs(output_dir, exist_ok=True)

    # 镜像 peptide.pdb
    pep_input = os.path.join(sample_dir, 'peptide.pdb')
    pep_output = os.path.join(output_dir, 'peptide.pdb')
    if os.path.exists(pep_input):
        mirror_pdb_parity(pep_input, pep_output)

    # 镜像 receptor.pdb
    rec_input = os.path.join(sample_dir, 'receptor.pdb')
    rec_output = os.path.join(output_dir, 'receptor.pdb')
    if os.path.exists(rec_input):
        mirror_pdb_parity(rec_input, rec_output)

    # 复制 fasta 文件 (不变)
    for fasta in ['peptide.fasta', 'receptor.fasta']:
        fasta_path = os.path.join(sample_dir, fasta)
        if os.path.exists(fasta_path):
            shutil.copy2(fasta_path, os.path.join(output_dir, fasta))

    return 1


def main():
    import argparse
    parser = argparse.ArgumentParser(description='镜像 PepMerge 数据集生成 D-肽版本 (XY镜像)')
    parser.add_argument('--input_dir', type=str, default='data/pepmerge',
                        help='原始 PepMerge 目录')
    parser.add_argument('--output_dir', type=str, default='data/pepmerge_mirrored',
                        help='镜像输出目录')
    parser.add_argument('--max_samples', type=int, default=None,
                        help='最大处理样本数 (默认全部)')
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    # 获取所有样本目录
    sample_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    sample_dirs.sort()

    if args.max_samples:
        sample_dirs = sample_dirs[:args.max_samples]

    print(f"找到 {len(sample_dirs)} 个样本目录")
    print(f"输入: {input_dir}")
    print(f"输出: {output_dir}")
    print("镜像方式: parity镜像 (空间反演) (x, y, z) -> (-x, -y, -z)")
    print("  这是坐标变换模拟 D-肽手性，非真正的 D-氨基酸")

    # 创建输出根目录
    output_dir.mkdir(parents=True, exist_ok=True)

    # 处理每个样本
    from tqdm import tqdm
    for sample_dir in tqdm(sample_dirs, desc='镜像处理'):
        try:
            process_sample(str(sample_dir), str(output_dir))
        except Exception as e:
            print(f"处理失败 {sample_dir.name}: {e}")

    print(f"\n完成! 镜像数据已保存到: {output_dir}")
    print("你可以用以下命令对比训练:")
    print("  # 纯 L 训练:")
    print(f"  pdexflow-train data.structure_dir={input_dir}")
    print("  # 纯 D 训练 (parity镜像):")
    print(f"  pdexflow-train data.structure_dir={output_dir}")


if __name__ == '__main__':
    main()
