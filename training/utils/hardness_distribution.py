import json
from collections import defaultdict
from typing import Dict, Any, Optional



def evaluate_by_hardness(
    hardness_file: str,
    trajectory_file: str,
    known_totals: Optional[Dict[str, int]] = None
) -> Dict[str, Dict[str, Any]]:
    """
    根据官方难度标注，评估轨迹文件中各难度级别的准确率。

    Args:
        hardness_file (str): 官方难度标注 JSON 文件路径（如 hardness_results.json）
        trajectory_file (str): 轨迹结果 JSON 文件路径（如 test_rollouts_*.json）
        known_totals (Optional[Dict[str, int]]): 各难度的真实总数（用于分母）。若为 None，则使用实际匹配到的数量。

    Returns:
        Dict[str, Dict[str, Any]]: 按难度分类的评估结果，包含 correct, total, accuracy
    """
    # 默认已知总数（Spider dev set）
    if known_totals is None:
        known_totals = {
            "easy": 470,
            "medium": 857,
            "hard": 463,
            "extra": 357
        }

    # 加载难度标注
    with open(hardness_file, "r", encoding="utf-8") as f:
        spider = json.load(f)
    q2hardness = {item["question"]: item["hardness"] for item in spider}

    # 加载轨迹
    with open(trajectory_file, "r", encoding="utf-8") as f:
        traj = json.load(f)

    correct = defaultdict(int)
    matched_total = defaultdict(int)

    for idx, t in enumerate(traj):
        try:
            q = t.get("question")
            if not q or q not in q2hardness:
                continue
            h = q2hardness[q]
            matched_total[h] += 1
            if t.get("final_reward", 0) == 1.0:
                correct[h] += 1
        except Exception as e:
            print(f"Warning: skipping index {idx} due to error: {e}")

    # 构建结果
    results = {}
    for h in ["easy", "medium", "hard", "extra"]:
        c = correct[h]
        tot = known_totals[h]  # 使用真实总数作为分母（避免漏题导致分母偏小）
        acc = c / tot if tot > 0 else 0.0
        results[h] = {
            "correct": c,
            "total": tot,
            "matched_in_traj": matched_total[h],
            "accuracy": acc
        }

    return results


def print_results(results: Dict[str, Dict[str, Any]]) -> None:
    """打印格式化评估结果"""
    print("难度\t答对数\t总数\t准确率")
    for h in ["easy", "medium", "hard", "extra"]:
        r = results[h]
        print(f"{h}\t{r['correct']}\t{r['total']}\t{r['accuracy']:.2%}")


def main() -> None:
    """命令行入口"""
    import argparse

    parser = argparse.ArgumentParser(description="按难度评估 Spider 轨迹准确率")
    parser.add_argument(
        "--hardness-file",
        type=str,
        default="/path/to/LadderSQL/examples/my_spider/agent/hardness_results.json",
        help="官方难度标注文件路径"
    )
    parser.add_argument(
        "--trajectory-file",
        type=str,
        default="/path/to/LadderSQL/examples/my_spider/agent/agent_v1/results/20260204/test_rollouts_llm_truncated_20260205_1033.json",
        help="轨迹结果文件路径"
    )
    args = parser.parse_args()

    results = evaluate_by_hardness(
        hardness_file=args.hardness_file,
        trajectory_file=args.trajectory_file
    )
    print_results(results)


if __name__ == "__main__":
    main()