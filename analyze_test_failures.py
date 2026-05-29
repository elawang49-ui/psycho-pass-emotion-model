from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
RESULT_PATH = BASE_DIR / "test_result.csv"
ANALYSIS_PATH = BASE_DIR / "test_failure_analysis.csv"

REQUIRED_COLUMNS = [
    "text",
    "case_type",
    "expected_valence_min",
    "expected_valence_max",
    "expected_arousal_min",
    "expected_arousal_max",
    "pred_valence",
    "pred_arousal",
    "pred_main_emotion",
    "valence_pass",
    "arousal_pass",
    "both_pass",
]


def load_results():
    if not RESULT_PATH.exists():
        raise FileNotFoundError(f"未找到评估结果文件: {RESULT_PATH}")

    data = pd.read_csv(RESULT_PATH, encoding="utf-8-sig")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"test_result.csv 缺少字段: {', '.join(missing_columns)}")

    return data


def as_bool(series):
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().eq("true")


def format_rate(value):
    return f"{value * 100:.2f}%"


def describe_bias(value, metric):
    if value > 0.05:
        return f"{metric} 偏高"
    if value < -0.05:
        return f"{metric} 偏低"
    return f"{metric} 基本接近"


def analyze():
    data = load_results()
    for column in ["valence_pass", "arousal_pass", "both_pass"]:
        data[column] = as_bool(data[column])

    data["expected_valence_center"] = (
        data["expected_valence_min"] + data["expected_valence_max"]
    ) / 2
    data["expected_arousal_center"] = (
        data["expected_arousal_min"] + data["expected_arousal_max"]
    ) / 2
    data["valence_diff"] = data["pred_valence"] - data["expected_valence_center"]
    data["arousal_diff"] = data["pred_arousal"] - data["expected_arousal_center"]

    failed = data[~(data["both_pass"] & data["valence_pass"] & data["arousal_pass"])].copy()

    grouped = (
        data.groupby("case_type", dropna=False)
        .agg(
            total=("text", "count"),
            both_pass_count=("both_pass", "sum"),
            both_pass_rate=("both_pass", "mean"),
            valence_pass_rate=("valence_pass", "mean"),
            arousal_pass_rate=("arousal_pass", "mean"),
            avg_valence_diff=("valence_diff", "mean"),
            avg_arousal_diff=("arousal_diff", "mean"),
        )
        .reset_index()
    )
    grouped["valence_error_direction"] = grouped["avg_valence_diff"].apply(
        lambda value: describe_bias(value, "valence")
    )
    grouped["arousal_error_direction"] = grouped["avg_arousal_diff"].apply(
        lambda value: describe_bias(value, "arousal")
    )
    grouped["major_error_direction"] = grouped.apply(
        lambda row: (
            row["valence_error_direction"]
            if abs(row["avg_valence_diff"]) >= abs(row["avg_arousal_diff"])
            else row["arousal_error_direction"]
        ),
        axis=1,
    )

    grouped.to_csv(ANALYSIS_PATH, index=False, encoding="utf-8-sig")
    return data, failed, grouped


def print_summary(failed, grouped):
    display_grouped = grouped.copy()
    for column in ["both_pass_rate", "valence_pass_rate", "arousal_pass_rate"]:
        display_grouped[column] = display_grouped[column].map(format_rate)

    print("按 case_type 分组统计:")
    print(display_grouped.to_string(index=False))

    failure_columns = [
        "text",
        "case_type",
        "expected_valence_min",
        "expected_valence_max",
        "pred_valence",
        "expected_arousal_min",
        "expected_arousal_max",
        "pred_arousal",
        "pred_main_emotion",
    ]
    print("\n失败样本前 50 条:")
    if failed.empty:
        print("无")
    else:
        print(failed[failure_columns].head(50).to_string(index=False))

    print(f"\n分析结果已保存: {ANALYSIS_PATH}")


def main():
    _, failed, grouped = analyze()
    print_summary(failed, grouped)


if __name__ == "__main__":
    main()
