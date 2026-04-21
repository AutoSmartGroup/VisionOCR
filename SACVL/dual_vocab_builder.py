"""
双词表构建模块 v2
- 词表1：用于 OCR 和 GT（从数据中提取，包含少量中文）
- 词表2：用于产品名称（从数据中提取，包含中文字符）
"""

import pandas as pd
from collections import defaultdict
import re


def build_vocab_from_columns(df, columns, max_vocab_size=700, vocab_name="Unknown"):
    """
    从指定的列构建词表（保留所有字符，包括中文）

    Args:
        df: DataFrame
        columns: list of str, 要使用的列名
        max_vocab_size: 最大词表大小
        vocab_name: str, 词表名称（用于打印信息）

    Returns:
        char2id: dict, 字符到 ID 的映射
        id2char: dict, ID 到字符的映射
    """
    char_count = defaultdict(int)

    # 从指定列收集字符
    for col in columns:
        if col in df.columns:
            for s in df[col]:
                if pd.isna(s):
                    continue
                s = str(s).upper().strip()
                s = re.sub(r'\s+', '', s)

                for char in s:
                    char_count[char] += 1

    # 按频率排序
    sorted_chars = sorted(char_count.items(), key=lambda x: -x[1])

    # 构建词表
    special_tokens = ['<PAD>', '<SOS>', '<EOS>', '<UNK>']
    char2id = {token: idx for idx, token in enumerate(special_tokens)}

    # 添加高频字符
    for char, count in sorted_chars[:max_vocab_size - len(special_tokens)]:
        char2id[char] = len(char2id)

    id2char = {v: k for k, v in char2id.items()}

    # 打印词表统计
    print(f"\n{'='*80}")
    print(f"{vocab_name} VOCABULARY BUILDING")
    print(f"{'='*80}")
    print(f"Vocabulary size: {len(char2id)}")
    print(f"  Special tokens: {special_tokens}")
    print(f"  Unique characters: {len(char2id) - len(special_tokens)}")

    # 统计字符类型
    digit_chars = [c for c in char2id.keys() if c.isdigit()]
    alpha_chars = [c for c in char2id.keys() if c.isalpha() and not ('\u4e00' <= c <= '\u9fff')]
    chinese_chars = [c for c in char2id.keys() if '\u4e00' <= c <= '\u9fff']
    special_chars = [c for c in char2id.keys()
                     if not c.isdigit() and not c.isalpha() and
                     c not in special_tokens]

    print(f"\nVocabulary composition:")
    print(f"  Digits (0-9): {len(set(digit_chars))}")
    print(f"  Letters (A-Z, a-z, non-Chinese): {len(set(alpha_chars))}")
    print(f"  Chinese characters: {len(set(chinese_chars))}")
    print(f"  Special characters: {len(set(special_chars))}")

    # 打印中文字符
    if chinese_chars:
        print(f"\nChinese characters found: {len(set(chinese_chars))}")
        print(f"  First 20: {sorted(set(chinese_chars))[:20]}")
        if len(set(chinese_chars)) > 20:
            print(f"  Last 20: {sorted(set(chinese_chars))[-20:]}")
    else:
        print(f"\n✓ No Chinese characters found")

    # 打印前 20 个字符
    sorted_vocab_chars = sorted([c for c in char2id.keys() if c not in special_tokens])
    if len(sorted_vocab_chars) > 0:
        print(f"\nFirst 20 characters: {sorted_vocab_chars[:20]}")
    if len(sorted_vocab_chars) > 20:
        print(f"Last 20 characters: {sorted_vocab_chars[-20:]}")

    print(f"{'='*80}\n")

    return char2id, id2char


def build_vocab_for_ocr_gt(df, max_vocab_size=700):
    """
    构建用于 OCR 和 GT 的词表（从 ppocrstr3 和 cinvstd 列提取，包含少量中文）

    Args:
        df: DataFrame, 包含 ppocrstr3 和 cinvstd 列
        max_vocab_size: 最大词表大小

    Returns:
        char2id: dict, OCR/GT 字符到 ID 的映射
        id2char: dict, OCR/GT ID 到字符的映射
    """
    return build_vocab_from_columns(
        df,
        columns=['ppocrstr3', 'cinvstd'],
        max_vocab_size=max_vocab_size,
        vocab_name="OCR/GT"
    )


def build_vocab_for_name(df, max_vocab_size=700):
    """
    构建用于产品名称的词表（从 cinvname 列提取，包含中文字符）

    Args:
        df: DataFrame, 包含 cinvname 列
        max_vocab_size: 最大词表大小（默认 500）

    Returns:
        name2id: dict, Name 字符到 ID 的映射
        id2name: dict, Name ID 到字符的映射
    """
    return build_vocab_from_columns(
        df,
        columns=['cinvname'],
        max_vocab_size=max_vocab_size,
        vocab_name="NAME"
    )


def build_dual_vocabularies(df, vocab_size_ocr_gt=700, vocab_size_name=700):
    """
    同时构建两套词表

    Args:
        df: DataFrame, 包含 ppocrstr3, cinvstd, cinvname 列
        vocab_size_ocr_gt: OCR/GT 词表大小
        vocab_size_name: Name 词表大小

    Returns:
        char2id: dict, OCR/GT 字符到 ID 的映射
        id2char: dict, OCR/GT ID 到字符的映射
        name2id: dict, Name 字符到 ID 的映射
        id2name: dict, Name ID 到字符的映射
    """
    char2id, id2char = build_vocab_for_ocr_gt(df, vocab_size_ocr_gt)
    name2id, id2name = build_vocab_for_name(df, vocab_size_name)

    return char2id, id2char, name2id, id2name


def save_dual_vocabularies(char2id, name2id, output_dir='vocabularies'):
    """
    保存两套词表到文件

    Args:
        char2id: OCR/GT 词表
        name2id: Name 词表
        output_dir: 输出目录
    """
    import os
    import json

    os.makedirs(output_dir, exist_ok=True)
        # 生成 id2char 和 id2name
    id2char = {v: k for k, v in char2id.items()}
    id2name = {v: k for k, v in name2id.items()}

    # 保存 OCR/GT 词表
    with open(os.path.join(output_dir, 'char2id.json'), 'w', encoding='utf-8') as f:
        json.dump(char2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'id2char.json'), 'w', encoding='utf-8') as f:
        id2char_str_keys = {str(k): v for k, v in id2char.items()}
        json.dump(id2char_str_keys, f, ensure_ascii=False, indent=2)

    # 保存 Name 词表
    with open(os.path.join(output_dir, 'name2id.json'), 'w', encoding='utf-8') as f:
        json.dump(name2id, f, ensure_ascii=False, indent=2)

    with open(os.path.join(output_dir, 'id2name.json'), 'w', encoding='utf-8') as f:
        id2name_str_keys = {str(k): v for k, v in id2name.items()}
        json.dump(id2name_str_keys, f, ensure_ascii=False, indent=2)

    print(f"\nVocabularies saved to '{output_dir}':")
    print(f"  - char2id.json (OCR/GT)")
    print(f"  - id2char.json (OCR/GT)")
    print(f"  - name2id.json (Name)")
    print(f"  - id2name.json (Name)")


def load_dual_vocabularies(vocab_dir='vocabularies'):
    """
    从文件加载两套词表

    Args:
        vocab_dir: 词表目录

    Returns:
        char2id: dict, OCR/GT 字符到 ID 的映射
        id2char: dict, OCR/GT ID 到字符的映射
        name2id: dict, Name 字符到 ID 的映射
        id2name: dict, Name ID 到字符的映射
    """
    import os
    import json

    with open(os.path.join(vocab_dir, 'char2id.json'), 'r', encoding='utf-8') as f:
        char2id = json.load(f)

    with open(os.path.join(vocab_dir, 'id2char.json'), 'r', encoding='utf-8') as f:
        id2char_raw = json.load(f)
        id2char = {int(k): v for k, v in id2char_raw.items()}

    with open(os.path.join(vocab_dir, 'name2id.json'), 'r', encoding='utf-8') as f:
        name2id = json.load(f)

    with open(os.path.join(vocab_dir, 'id2name.json'), 'r', encoding='utf-8') as f:
        id2name_raw = json.load(f)
        id2name = {int(k): v for k, v in id2name_raw.items()}

    print(f"\nVocabularies loaded from '{vocab_dir}':")
    print(f"  - OCR/GT vocab size: {len(char2id)}")
    print(f"  - Name vocab size: {len(name2id)}")

    return char2id, id2char, name2id, id2name


if __name__ == "__main__":
    print(__doc__)
    print("\n" + "="*80)
    print("USAGE EXAMPLE")
    print("="*80)
    print("""
# 在训练脚本中
from dual_vocab_builder_v2 import build_dual_vocabularies, save_dual_vocabularies
import pandas as pd

# 加载数据
df = pd.read_excel("your_data.xlsx")

# 构建两套词表（从文件中提取，包含中文）
char2id, id2char, name2id, id2name = build_dual_vocabularies(df)

# 保存词表
save_dual_vocabularies(char2id, name2id, output_dir='vocabularies')

# 在测试脚本中
from dual_vocab_builder_v2 import load_dual_vocabularies

# 加载词表
char2id, id2char, name2id, id2name = load_dual_vocabularies(vocab_dir='vocabularies')
    """)
