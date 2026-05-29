from emotion_core import predict_one


def main():
    sentences = [
        "开心",
        "很开心",
        "好开心",
        "开心死了",
        "喜欢",
        "我喜欢你",
        "安心",
        "放心",
        "收到",
        "我看了一下",
        "挺好的，下次别做了",
        "我真的谢谢你啊",
        "好好好，又是我的问题",
        "行，真有你的",
        "真有你的",
        "你可真会安排",
        "气死我了",
        "好累",
    ]

    print("text,valence,arousal,main_emotion,intensity,intensity_level")
    for text in sentences:
        result = predict_one(text)
        print(
            f"{text},"
            f"{result['pred_valence']:.4f},"
            f"{result['pred_arousal']:.4f},"
            f"{result['main_emotion']},"
            f"{result['emotion_intensity']:.4f},"
            f"{result['intensity_level']}"
        )


if __name__ == "__main__":
    main()
