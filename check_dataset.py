import math
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
REQUIRED_COLUMNS = ["text", "valence", "arousal", "emotion"]
BIN_LABELS = ["-1.0~-0.6", "-0.6~-0.2", "-0.2~0.2", "0.2~0.6", "0.6~1.0"]
BIN_EDGES = [-1.0, -0.6, -0.2, 0.2, 0.6, 1.0]
COMPASS_LABELS = [
    "开心/满足",
    "兴奋/狂喜",
    "紧张/警觉",
    "愤怒/焦灼",
    "失望/厌烦",
    "悲伤/疲惫",
    "麻木/平静",
    "放松/安心",
]


def compass_label(valence, arousal):
    angle = math.degrees(math.atan2(arousal, valence)) % 360
    if angle >= 337.5 or angle < 22.5:
        return "开心/满足"
    if angle < 67.5:
        return "兴奋/狂喜"
    if angle < 112.5:
        return "紧张/警觉"
    if angle < 157.5:
        return "愤怒/焦灼"
    if angle < 202.5:
        return "失望/厌烦"
    if angle < 247.5:
        return "悲伤/疲惫"
    if angle < 292.5:
        return "麻木/平静"
    return "放松/安心"


def print_count_table(title, counts, total):
    print(f"\n{title}:")
    for label, count in counts.items():
        ratio = count / total if total else 0
        print(f"{label}: {count} ({ratio:.1%})")


def binned_counts(series):
    bins = pd.cut(series, bins=BIN_EDGES, labels=BIN_LABELS, include_lowest=True, right=True)
    return bins.value_counts(sort=False).reindex(BIN_LABELS, fill_value=0)


def check_dataset(path=DATA_PATH):
    data = pd.read_csv(path, encoding="utf-8-sig")
    total = len(data)
    print(f"数据文件: {path}")
    print(f"数据总条数: {total}")

    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        print(f"缺少必要列: {', '.join(missing_columns)}")
        return False

    numeric = data[["valence", "arousal"]].apply(pd.to_numeric, errors="coerce")
    valid_mask = numeric.notna().all(axis=1)
    range_mask = numeric.ge(-1).all(axis=1) & numeric.le(1).all(axis=1)
    valid_data = data.loc[valid_mask & range_mask].copy()
    valid_data[["valence", "arousal"]] = numeric.loc[valid_mask & range_mask]
    valid_total = len(valid_data)

    print("\nemotion 类别分布:")
    print(data["emotion"].fillna("<缺失>").value_counts().to_string())

    valence_counts = binned_counts(valid_data["valence"])
    arousal_counts = binned_counts(valid_data["arousal"])
    print_count_table("valence 分区统计", valence_counts.to_dict(), valid_total)
    print_count_table("arousal 分区统计", arousal_counts.to_dict(), valid_total)

    direction_counts = (
        valid_data.apply(lambda row: compass_label(row["valence"], row["arousal"]), axis=1)
        .value_counts()
        .reindex(COMPASS_LABELS, fill_value=0)
    )
    print_count_table("八方向象限统计", direction_counts.to_dict(), valid_total)

    print("\n自动诊断:")
    warnings = []
    if valid_total == 0:
        warnings.append("没有可用于罗素圆环分布分析的有效 valence/arousal 数据。")
    else:
        center_count = valid_data[
            valid_data["valence"].between(-0.2, 0.2, inclusive="both")
            & valid_data["arousal"].between(-0.2, 0.2, inclusive="both")
        ].shape[0]
        center_ratio = center_count / valid_total
        right_down_ratio = direction_counts["放松/安心"] / valid_total
        right_up_ratio = direction_counts["兴奋/狂喜"] / valid_total
        left_ratio = (valid_data["valence"] < 0).mean()

        if center_ratio < 0.15:
            warnings.append(f"中性样本不足：中心区域占比 {center_ratio:.1%}，低于 15%。")
        if right_down_ratio < 0.10:
            warnings.append(f"放松/安心样本不足：占比 {right_down_ratio:.1%}，低于 10%。")
        if right_up_ratio < 0.10:
            warnings.append(f"兴奋/正向高唤醒样本不足：占比 {right_up_ratio:.1%}，低于 10%。")
        if left_ratio > 0.50:
            warnings.append(f"负面样本占比过高：valence < 0 占比 {left_ratio:.1%}，高于 50%。")

    if warnings:
        for item in warnings:
            print(f"- {item}")
    else:
        print("- 未发现明显分布偏科。")

    invalid_numeric = int((~valid_mask).sum())
    outside_range = int((valid_mask & ~range_mask).sum())
    if invalid_numeric:
        print(f"- 注意：有 {invalid_numeric} 条 valence/arousal 非数字，未纳入分布统计。")
    if outside_range:
        print(f"- 注意：有 {outside_range} 条 valence/arousal 超出 -1 到 1，未纳入分布统计。")

    return not missing_columns and invalid_numeric == 0 and outside_range == 0


if __name__ == "__main__":
    check_dataset()
