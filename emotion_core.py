import math
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "emotion_model.pkl"
VECTORIZER_PATH = BASE_DIR / "vectorizer.pkl"
PREDICTION_COLUMNS = ["pred_valence", "pred_arousal"]
ANALYSIS_COLUMNS = ["emotion_angle", "emotion_intensity", "intensity_level", "main_emotion", "direction_hint"]
OUTPUT_COLUMNS = PREDICTION_COLUMNS + ANALYSIS_COLUMNS

SHORT_TEXT_SCORES = {
    "开心": (0.65, 0.35),
    "很开心": (0.72, 0.45),
    "好开心": (0.78, 0.55),
    "开心死了": (0.82, 0.65),
    "高兴": (0.65, 0.35),
    "很高兴": (0.70, 0.42),
    "太高兴了": (0.78, 0.60),
    "喜欢": (0.65, 0.25),
    "我喜欢": (0.68, 0.30),
    "我喜欢你": (0.72, 0.35),
    "很喜欢": (0.70, 0.35),
    "好喜欢": (0.75, 0.45),
    "不错": (0.40, 0.05),
    "还不错": (0.38, 0.00),
    "满意": (0.50, 0.05),
    "很满意": (0.58, 0.10),
    "挺好的": (0.35, 0.00),
    "安心": (0.55, -0.35),
    "放心": (0.52, -0.30),
    "放松": (0.55, -0.45),
    "舒服": (0.50, -0.30),
    "稳了": (0.45, -0.20),
    "收到": (0.00, 0.00),
    "知道了": (0.00, 0.00),
    "看了": (0.00, 0.00),
    "好的": (0.08, -0.05),
    "嗯": (0.00, -0.05),
    "行": (0.00, 0.00),
    "烦": (-0.55, 0.45),
    "烦死了": (-0.75, 0.75),
    "累": (-0.45, -0.45),
    "好累": (-0.55, -0.55),
    "失望": (-0.65, -0.20),
    "无语": (-0.55, 0.25),
    "气死": (-0.85, 0.85),
    "气死我了": (-0.90, 0.90),
    "麻了": (-0.45, -0.45),
    "算了": (-0.30, -0.30),
}

SARCASM_TEXT_SCORES = {
    "挺好的，下次别做了": (-0.45, 0.25),
    "挺好的，下次别": (-0.42, 0.22),
    "我真的谢谢你啊": (-0.50, 0.40),
    "谢谢你啊": (-0.38, 0.25),
    "好好好，又是我的问题": (-0.60, 0.55),
    "又是我的问题": (-0.55, 0.45),
    "行，真有你的": (-0.55, 0.45),
    "真有你的": (-0.50, 0.40),
    "你可真会安排": (-0.55, 0.45),
    "太贴心了": (-0.40, 0.30),
    "真贴心啊": (-0.45, 0.35),
    "这服务真不错": (-0.40, 0.25),
    "真是大开眼界": (-0.50, 0.40),
    "笑死，太棒了": (-0.45, 0.45),
    "好棒棒哦": (-0.40, 0.35),
    "真不愧是你": (-0.42, 0.30),
    "我服了你了": (-0.55, 0.50),
    "你真厉害啊": (-0.35, 0.25),
}

SARCASM_LABEL = "阴阳怪气 / 轻微负向"


@lru_cache(maxsize=1)
def load_model():
    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    return model, vectorizer


def analyze_emotion(valence, arousal):
    valence = float(max(-1, min(1, valence)))
    arousal = float(max(-1, min(1, arousal)))
    angle = math.degrees(math.atan2(arousal, valence)) % 360
    intensity = min(1.0, math.sqrt(valence**2 + arousal**2))

    if valence > 0.15:
        direction_hint = "轻微正向"
    elif valence < -0.15:
        direction_hint = "轻微负向"
    else:
        direction_hint = "接近中性"

    if intensity < 0.25:
        main_emotion = "低波动 / 轻微偏向"
    elif angle >= 337.5 or angle < 22.5:
        main_emotion = "开心 / 满足"
    elif angle < 67.5:
        main_emotion = "兴奋 / 狂喜"
    elif angle < 112.5:
        main_emotion = "紧张 / 警觉"
    elif angle < 157.5:
        main_emotion = "愤怒 / 焦灼"
    elif angle < 202.5:
        main_emotion = "失望 / 厌烦"
    elif angle < 247.5:
        main_emotion = "悲伤 / 疲惫"
    elif angle < 292.5:
        main_emotion = "麻木 / 平静"
    else:
        main_emotion = "放松 / 安心"

    if intensity < 0.35:
        intensity_level = "轻微"
    elif intensity < 0.55:
        intensity_level = "中等"
    elif intensity < 0.80:
        intensity_level = "强烈"
    else:
        intensity_level = "极强"

    return {
        "emotion_angle": angle,
        "emotion_intensity": intensity,
        "intensity_level": intensity_level,
        "main_emotion": main_emotion,
        "direction_hint": direction_hint,
    }


def short_text_fallback(text):
    clean_text = str(text).strip()
    if len(clean_text) <= 6 and clean_text in SHORT_TEXT_SCORES:
        valence, arousal = SHORT_TEXT_SCORES[clean_text]
        return {"pred_valence": valence, "pred_arousal": arousal}
    return None


def sarcasm_fallback(text):
    clean_text = str(text).strip()
    if clean_text in SARCASM_TEXT_SCORES:
        valence, arousal = SARCASM_TEXT_SCORES[clean_text]
        return {
            "pred_valence": valence,
            "pred_arousal": arousal,
            "special_label": SARCASM_LABEL,
        }
    return None


def predict_texts(texts):
    clean_texts = [str(text).strip() for text in texts]
    result = pd.DataFrame(index=range(len(clean_texts)), columns=PREDICTION_COLUMNS + ["special_label"])
    model_texts = []
    model_indices = []

    for index, text in enumerate(clean_texts):
        fallback = sarcasm_fallback(text) or short_text_fallback(text)
        if fallback is None:
            model_indices.append(index)
            model_texts.append(text)
        else:
            result.loc[index, "pred_valence"] = fallback["pred_valence"]
            result.loc[index, "pred_arousal"] = fallback["pred_arousal"]
            result.loc[index, "special_label"] = fallback.get("special_label")

    if model_texts:
        model, vectorizer = load_model()
        features = vectorizer.transform(model_texts)
        predictions = model.predict(features)
        for index, prediction in zip(model_indices, predictions):
            result.loc[index, "pred_valence"] = prediction[0]
            result.loc[index, "pred_arousal"] = prediction[1]

    result["pred_valence"] = pd.to_numeric(result["pred_valence"]).clip(-1, 1)
    result["pred_arousal"] = pd.to_numeric(result["pred_arousal"]).clip(-1, 1)
    return result


def predict_one(text):
    special_label = None
    fallback = sarcasm_fallback(text)
    if fallback is not None:
        prediction = pd.Series(fallback)
        special_label = fallback["special_label"]
    else:
        prediction = predict_texts([text]).iloc[0]

    analysis = analyze_emotion(prediction["pred_valence"], prediction["pred_arousal"])
    if special_label:
        analysis["main_emotion"] = special_label

    return {
        "text": str(text).strip(),
        "pred_valence": float(prediction["pred_valence"]),
        "pred_arousal": float(prediction["pred_arousal"]),
        **analysis,
    }
