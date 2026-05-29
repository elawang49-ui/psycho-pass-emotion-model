from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputRegressor


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data_clean.csv"
MODEL_PATH = BASE_DIR / "emotion_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
REQUIRED_COLUMNS = ["text", "valence", "arousal", "emotion"]


def load_training_data():
    if not DATA_PATH.exists():
        raise FileNotFoundError("未找到 data_clean.csv，请先运行 clean_dataset.py 生成干净训练集。")

    data = pd.read_csv(DATA_PATH, encoding="utf-8-sig")
    missing_columns = [column for column in REQUIRED_COLUMNS if column not in data.columns]
    if missing_columns:
        raise ValueError(f"data_clean.csv 缺少字段: {', '.join(missing_columns)}")

    data = data.dropna(subset=["text", "valence", "arousal"]).copy()
    data["text"] = data["text"].astype(str).str.strip()
    data = data[data["text"] != ""]
    data[["valence", "arousal"]] = data[["valence", "arousal"]].apply(pd.to_numeric, errors="raise")

    bounded = data[["valence", "arousal"]].ge(-1) & data[["valence", "arousal"]].le(1)
    outside = ~bounded.all(axis=1)
    if outside.any():
        raise ValueError("data_clean.csv 中存在超出 -1 到 1 范围的 valence/arousal。")
    return data


def train_model():
    data = load_training_data()
    vectorizer = TfidfVectorizer(
        analyzer="char",
        ngram_range=(1, 3),
        min_df=1,
        max_features=8000,
    )
    features = vectorizer.fit_transform(data["text"])
    targets = data[["valence", "arousal"]]

    model = MultiOutputRegressor(
        RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    )
    model.fit(features, targets)

    joblib.dump(model, MODEL_PATH)
    joblib.dump(vectorizer, VECTORIZER_PATH)
    return data, model, vectorizer


if __name__ == "__main__":
    training_data, _, _ = train_model()
    print(f"训练完成，共使用 {len(training_data)} 条数据。")
    print(f"模型已保存到: {MODEL_PATH}")
    print(f"向量器已保存到: {VECTORIZER_PATH}")
