from pathlib import Path

import pandas as pd

from emotion_core import predict_one


BASE_DIR = Path(__file__).resolve().parent
TEST_PATH = BASE_DIR / "test.csv"
RESULT_PATH = BASE_DIR / "test_result.csv"

REQUIRED_COLUMNS = [
    "text",
    "expected_valence_min",
    "expected_valence_max",
    "expected_arousal_min",
    "expected_arousal_max",
    "expected_emotion",
    "case_type",
]


def load_test_data():
    if not TEST_PATH.exists():
        raise FileNotFoundError(f"未找到测试文件: {TEST_PATH}")

    df = pd.read_csv(TEST_PATH, encoding="utf-8-sig")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in df.columns]
    if missing_columns:
        raise ValueError(f"test.csv 缺少字段: {', '.join(missing_columns)}")

    return df


def predict_row(text):
    result = predict_one(text)
    return {
        "pred_valence": result["pred_valence"],
        "pred_arousal": result["pred_arousal"],
        "pred_main_emotion": result["main_emotion"],
        "pred_intensity": result["emotion_intensity"],
    }


def format_rate(value):
    return f"{value * 100:.2f}%"


def evaluate():
    df = load_test_data()
    predictions = df["text"].astype(str).apply(predict_row)
    prediction_df = pd.DataFrame(predictions.tolist())
    result_df = pd.concat([df, prediction_df], axis=1)

    result_df["valence_pass"] = result_df["pred_valence"].between(
        result_df["expected_valence_min"], result_df["expected_valence_max"], inclusive="both"
    )
    result_df["arousal_pass"] = result_df["pred_arousal"].between(
        result_df["expected_arousal_min"], result_df["expected_arousal_max"], inclusive="both"
    )
    result_df["both_pass"] = result_df["valence_pass"] & result_df["arousal_pass"]

    result_df.to_csv(RESULT_PATH, index=False, encoding="utf-8-sig")
    return result_df


def print_summary(result_df):
    total = len(result_df)
    print(f"总样本数: {total}")
    print(f"valence 通过率: {format_rate(result_df['valence_pass'].mean())}")
    print(f"arousal 通过率: {format_rate(result_df['arousal_pass'].mean())}")
    print(f"两者都通过的通过率: {format_rate(result_df['both_pass'].mean())}")

    print("\n按 case_type 分组的通过率:")
    grouped = (
        result_df.groupby("case_type", dropna=False)
        .agg(
            sample_count=("text", "count"),
            valence_pass_rate=("valence_pass", "mean"),
            arousal_pass_rate=("arousal_pass", "mean"),
            both_pass_rate=("both_pass", "mean"),
        )
        .reset_index()
    )
    for column in ["valence_pass_rate", "arousal_pass_rate", "both_pass_rate"]:
        grouped[column] = grouped[column].map(format_rate)
    print(grouped.to_string(index=False))

    failed = result_df[~result_df["both_pass"]].head(30)
    print("\n失败样本前 30 条:")
    if failed.empty:
        print("无")
    else:
        display_columns = [
            "text",
            "case_type",
            "expected_emotion",
            "expected_valence_min",
            "expected_valence_max",
            "pred_valence",
            "valence_pass",
            "expected_arousal_min",
            "expected_arousal_max",
            "pred_arousal",
            "arousal_pass",
            "pred_main_emotion",
            "pred_intensity",
        ]
        print(failed[display_columns].to_string(index=False))

    print(f"\n结果已保存: {RESULT_PATH}")


def main():
    result_df = evaluate()
    print_summary(result_df)


if __name__ == "__main__":
    main()
