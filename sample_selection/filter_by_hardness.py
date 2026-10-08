import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

def load_hardness_map(hardness_json_path: str) -> Dict[Tuple[str, str], str]:
    """
    从 hardness JSON 文件加载 (db_id, question) -> hardness 的映射。
    """
    with open(hardness_json_path, "r", encoding="utf-8") as f:
        hardness_list = json.load(f)
    
    hardness_map = {}
    for item in hardness_list:
        db_id = item.get("db_id")
        question = item.get("question")
        hardness = item.get("hardness")
        if db_id and question and hardness:
            # 使用 (db_id, question) 作为唯一键
            key = (db_id, question)
            if key in hardness_map:
                print(f"Warning: duplicate key found: {key}")
            hardness_map[key] = hardness
    return hardness_map


def add_hardness_column(df: pd.DataFrame, hardness_map: Dict[Tuple[str, str], str]) -> pd.DataFrame:
    """
    为 DataFrame 添加 'hardness' 列，通过 (db_id, question) 匹配。
    无法匹配的样本将被标记为 NaN，并在后续被过滤掉（或报错）。
    """
    def get_hardness(row):
        key = (row["db_id"], row["question"])
        return hardness_map.get(key, None)
    
    df = df.copy()
    df["hardness"] = df.apply(get_hardness, axis=1)
    
    # 检查匹配率
    total = len(df)
    matched = df["hardness"].notna().sum()
    print(f"Hardness 匹配成功: {matched} / {total} ({matched/total:.2%})")
    
    if matched < total:
        unmatched = df[df["hardness"].isna()]
        print("前5条未匹配样本示例:")
        print(unmatched[["db_id", "question"]].head())
        raise ValueError("存在未匹配的样本，请检查数据一致性！")
    
    return df


def filter_by_hardness_ratio(
    df: pd.DataFrame,
    keep_ratios: dict,
    seed: int = 42
) -> pd.DataFrame:
    """
    按难度比例筛选，保持原始顺序。
    """
    np.random.seed(seed)
    keep_indices = []

    for difficulty in ["easy", "medium", "hard", "extra"]:
        mask = df["hardness"] == difficulty
        indices = df.index[mask].tolist()
        ratio = keep_ratios.get(difficulty, 1.0)

        if ratio >= 1.0:
            keep_indices.extend(indices)
        elif ratio > 0:
            n_keep = max(1, int(len(indices) * ratio))
            selected = np.random.choice(indices, size=n_keep, replace=False)
            keep_indices.extend(sorted(selected))  # 保持原始顺序

    keep_indices = sorted(set(keep_indices))
    return df.loc[keep_indices].reset_index(drop=True)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="基于 hardness JSON 筛选 train.parquet")
    parser.add_argument("--parquet", type=str, required=True, help="train.parquet 路径")
    parser.add_argument("--hardness-json", type=str, required=True, help="train_hardness.json 路径")
    parser.add_argument("--output", type=str, default="train_filtered.parquet", help="输出路径")
    args = parser.parse_args()

    # Step 1: 加载 hardness 映射
    print("Loading hardness map...")
    hardness_map = load_hardness_map(args.hardness_json)

    # Step 2: 读取 parquet 并添加 hardness 列
    print("Reading parquet and adding hardness column...")
    df_train = pd.read_parquet(args.parquet)
    if not {"db_id", "question"}.issubset(df_train.columns):
        raise ValueError("Parquet must contain 'db_id' and 'question' columns.")

    df_with_hardness = add_hardness_column(df_train, hardness_map)

    # Step 3: 按比例筛选
    print("Filtering by hardness ratio...")
    keep_ratios = {"easy": 0.1, "medium": 0.2, "hard": 1.0, "extra": 1.0}
    df_filtered = filter_by_hardness_ratio(df_with_hardness, keep_ratios, seed=42)

    # Step 4: 保存（可选择是否保留 hardness 列）
    df_filtered.to_parquet(args.output, index=False)

    # 打印统计
    print("\n原始分布:")
    print(df_with_hardness["hardness"].value_counts().reindex(["easy","medium","hard","extra"], fill_value=0))
    print("\n筛选后分布:")
    print(df_filtered["hardness"].value_counts().reindex(["easy","medium","hard","extra"], fill_value=0))
    print(f"\n总计: {len(df_with_hardness)} → {len(df_filtered)}")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()