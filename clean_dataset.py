from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data.csv"
CLEAN_PATH = BASE_DIR / "data_clean.csv"
INVALID_PATH = BASE_DIR / "data_invalid_rows.csv"
REQUIRED_COLUMNS = ["text", "valence", "arousal", "emotion"]


def load_dataset():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"未找到数据文件: {DATA_PATH}")

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"data.csv 缺少字段: {', '.join(missing_columns)}")

    return data


def clean_dataset():
    raw_data = load_dataset()
    original_count = len(raw_data)

    data = raw_data.copy()
    data["valence"] = pd.to_numeric(data["valence"], errors="coerce")
    data["arousal"] = pd.to_numeric(data["arousal"], errors="coerce")

    invalid_numeric_mask = data[["valence", "arousal"]].isna().any(axis=1)
    invalid_rows = raw_data.loc[invalid_numeric_mask].copy()
    invalid_rows.to_csv(INVALID_PATH, index=False, encoding="utf-8-sig")

    clean_data = data.loc[~invalid_numeric_mask].copy()

    before_empty_text = len(clean_data)
    clean_data["text"] = clean_data["text"].astype(str).str.strip()
    clean_data = clean_data[clean_data["text"].ne("") & clean_data["text"].ne("nan")]
    removed_empty_text = before_empty_text - len(clean_data)

    before_missing_value = len(clean_data)
    clean_data = clean_data.dropna(subset=["valence", "arousal"])
    removed_missing_value = before_missing_value - len(clean_data)

    before_outside = len(clean_data)
    bounded = clean_data["valence"].between(-1, 1) & clean_data["arousal"].between(-1, 1)
    clean_data = clean_data.loc[bounded].copy()
    removed_outside = before_outside - len(clean_data)

    before_duplicates = len(clean_data)
    clean_data = clean_data.drop_duplicates()
    removed_duplicates = before_duplicates - len(clean_data)

    clean_data.to_csv(CLEAN_PATH, index=False, encoding="utf-8-sig")

    report = {
        "original_count": original_count,
        "invalid_numeric_count": len(invalid_rows),
        "removed_empty_text": removed_empty_text,
        "removed_missing_value": removed_missing_value,
        "removed_outside": removed_outside,
        "removed_duplicates": removed_duplicates,
        "final_count": len(clean_data),
        "emotion_distribution": clean_data["emotion"].value_counts().head(20),
        "negative_valence_ratio": (clean_data["valence"] < 0).mean(),
        "center_ratio": ((clean_data["valence"].abs() < 0.2) & (clean_data["arousal"].abs() < 0.2)).mean(),
        "low_arousal_negative_ratio": ((clean_data["valence"] < -0.2) & (clean_data["arousal"] < -0.2)).mean(),
    }
    return report


def print_report(report):
    print(f"原始总行数: {report['original_count']}")
    print(f"非数字 valence/arousal 行数: {report['invalid_numeric_count']}")
    print(f"删除空文本行数: {report['removed_empty_text']}")
    print(f"删除 valence/arousal 空值行数: {report['removed_missing_value']}")
    print(f"删除越界行数: {report['removed_outside']}")
    print(f"删除重复行数: {report['removed_duplicates']}")
    print(f"最终有效行数: {report['final_count']}")

    print("\nemotion 分布前 20:")
    print(report["emotion_distribution"].to_string())

    print(f"\nvalence < 0 的比例: {report['negative_valence_ratio'] * 100:.2f}%")
    print(f"中心区域比例: {report['center_ratio'] * 100:.2f}%")
    print(f"低唤醒负面比例: {report['low_arousal_negative_ratio'] * 100:.2f}%")
    print(f"\n已保存干净训练集: {CLEAN_PATH}")
    print(f"已保存非数字无效行: {INVALID_PATH}")


def main():
    report = clean_dataset()
    print_report(report)


if __name__ == "__main__":
    main()
