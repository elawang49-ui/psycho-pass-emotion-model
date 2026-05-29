import math
from html import escape
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from emotion_core import (
    OUTPUT_COLUMNS,
    PREDICTION_COLUMNS,
    analyze_emotion,
    predict_one,
    predict_texts,
)

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False


def add_emotion_analysis(predictions):
    analysis = predictions.apply(
        lambda row: analyze_emotion(row["pred_valence"], row["pred_arousal"]), axis=1
    )
    interpreted = pd.DataFrame(analysis.tolist())
    if "special_label" in predictions.columns:
        labels = predictions["special_label"].reset_index(drop=True)
        interpreted.loc[labels.notna(), "main_emotion"] = labels[labels.notna()]
    display_predictions = predictions[PREDICTION_COLUMNS].reset_index(drop=True)
    return pd.concat([display_predictions, interpreted], axis=1)


def predict_dataframe(df):
    output = df.copy()
    for column in OUTPUT_COLUMNS:
        output[column] = pd.NA

    valid_rows = output["text"].notna() & output["text"].astype(str).str.strip().ne("")
    if valid_rows.any():
        predictions = predict_texts(output.loc[valid_rows, "text"].astype(str).tolist())
        interpreted = add_emotion_analysis(predictions)
        output.loc[valid_rows, OUTPUT_COLUMNS] = interpreted[OUTPUT_COLUMNS].to_numpy()
    return output, valid_rows


def read_csv_safely(file):
    encodings = ["utf-8-sig", "utf-8", "gbk", "gb18030"]
    for enc in encodings:
        try:
            file.seek(0)
            return pd.read_csv(file, encoding=enc)
        except UnicodeDecodeError:
            continue
    st.error("CSV 编码无法识别，请另存为 UTF-8 CSV 或上传 xlsx 文件。")
    return None


def read_uploaded_table(uploaded_file):
    suffix = Path(uploaded_file.name).suffix.lower()
    if suffix == ".csv":
        return read_csv_safely(uploaded_file)
    if suffix in {".xlsx", ".xls"}:
        return pd.read_excel(uploaded_file)
    st.error("仅支持 CSV 或 Excel 文件。")
    return None


def draw_emotion_compass(points, highlight_point=None, title=None):
    labels = [
        ("开心/满足", 0),
        ("兴奋/狂喜", 45),
        ("紧张/警觉", 90),
        ("愤怒/焦灼", 135),
        ("失望/厌烦", 180),
        ("悲伤/疲惫", 225),
        ("麻木/平静", 270),
        ("放松/安心", 315),
    ]
    plot_points = pd.DataFrame(points, columns=PREDICTION_COLUMNS)
    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    fig.patch.set_facecolor("#fbfcfe")
    ax.set_facecolor("#fbfcfe")

    ax.add_patch(plt.Circle((0, 0), 1, fill=False, color="#334155", linewidth=1.7))
    ax.axhline(0, color="#cbd5e1", linewidth=1.1)
    ax.axvline(0, color="#cbd5e1", linewidth=1.1)

    for label, degrees in labels:
        radians = math.radians(degrees)
        ax.text(
            1.18 * math.cos(radians),
            1.18 * math.sin(radians),
            label,
            ha="center",
            va="center",
            fontsize=10,
            color="#334155",
        )

    if not plot_points.empty:
        ax.scatter(
            plot_points["pred_valence"],
            plot_points["pred_arousal"],
            color="#e11d48",
            alpha=0.42 if len(plot_points) > 1 else 0.92,
            s=45 if len(plot_points) > 1 else 145,
            zorder=3,
            label="预测点" if len(plot_points) > 1 else None,
        )

    if highlight_point is not None:
        x = float(highlight_point["pred_valence"])
        y = float(highlight_point["pred_arousal"])
        ax.annotate(
            "",
            xy=(x, y),
            xytext=(0, 0),
            arrowprops={"arrowstyle": "->", "color": "#0f766e", "lw": 2.4},
            zorder=4,
        )
        ax.scatter(
            [x],
            [y],
            color="#0f766e",
            edgecolor="white",
            linewidth=1.5,
            s=185,
            zorder=5,
            label="平均向量" if len(plot_points) > 1 else "当前状态",
        )
        ax.legend(loc="lower right", frameon=False, fontsize=9)

    ax.set_xlim(-1.38, 1.38)
    ax.set_ylim(-1.38, 1.38)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("Valence（愉悦度）")
    ax.set_ylabel("Arousal（唤醒度）")
    ax.set_title(title or "心理测量者：八方向情绪罗盘", fontsize=14, pad=16)
    ax.grid(alpha=0.15, linestyle="--")
    for spine in ax.spines.values():
        spine.set_visible(False)
    return fig


def draw_time_trend(df, value_columns, title, ylabel):
    timeline = df.dropna(subset=["timestamp"]).sort_values("timestamp")
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for column, label, color in value_columns:
        ax.plot(timeline["timestamp"], timeline[column], marker="o", linewidth=1.7, label=label, color=color)
    ax.set_title(title, fontsize=12)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("时间")
    ax.grid(alpha=0.18, linestyle="--")
    if len(value_columns) > 1:
        ax.legend(frameon=False)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig


def show_state_card(title, analysis, valence, arousal):
    st.markdown(f"#### {title}")
    values = [
        ("当前主情绪", analysis["main_emotion"]),
        ("轻微方向", analysis["direction_hint"]),
        ("强度等级", analysis["intensity_level"]),
        ("情绪强度", f"{analysis['emotion_intensity']:.2f}"),
        ("valence（愉悦度）", f"{valence:.2f}"),
        ("arousal（唤醒度）", f"{arousal:.2f}"),
        ("方向角", f"{analysis['emotion_angle']:.1f}°"),
    ]
    cells = "".join(
        (
            '<div class="state-cell">'
            f'<div class="state-label">{escape(label)}</div>'
            f'<div class="state-value">{escape(value)}</div>'
            "</div>"
        )
        for label, value in values
    )
    st.markdown(f'<div class="state-card">{cells}</div>', unsafe_allow_html=True)


def show_single_detection():
    text = st.text_area("输入中文文本", height=110, placeholder="请输入需要分析的中文句子")
    if st.button("开始检测", type="primary", disabled=not text.strip()):
        result = predict_one(text)
        show_state_card("当前状态卡片", result, result["pred_valence"], result["pred_arousal"])
        point = pd.DataFrame([result])[PREDICTION_COLUMNS]
        st.pyplot(draw_emotion_compass(point, result))
        st.dataframe(pd.DataFrame([result]), use_container_width=True)


def show_batch_detection():
    st.caption("适用于普通评论表：只需要包含 text 列，不需要 timestamp。")
    uploaded_file = st.file_uploader(
        "上传普通评论 CSV / Excel", type=["csv", "xlsx", "xls"], key="batch_detection_upload"
    )
    if uploaded_file is None:
        return

    source = read_uploaded_table(uploaded_file)
    if source is None:
        return

    st.markdown("#### 原始数据预览")
    st.dataframe(source.head(20), use_container_width=True)

    if "text" not in source.columns:
        st.error("批量情绪检测文件至少需要包含 text 列。")
        return

    output, valid_rows = predict_dataframe(source)
    if not valid_rows.all():
        st.warning("空文本行不会参与预测，输出中的预测与解释字段将为空。")

    valid_output = output.loc[valid_rows, OUTPUT_COLUMNS]
    if not valid_output.empty:
        avg_valence = float(pd.to_numeric(valid_output["pred_valence"]).mean())
        avg_arousal = float(pd.to_numeric(valid_output["pred_arousal"]).mean())
        average_point = {"pred_valence": avg_valence, "pred_arousal": avg_arousal}
        average_analysis = analyze_emotion(avg_valence, avg_arousal)
        show_state_card("整体平均情绪状态卡片", average_analysis, avg_valence, avg_arousal)
        st.pyplot(draw_emotion_compass(valid_output[PREDICTION_COLUMNS], average_point))

    st.markdown("#### 预测结果表格")
    st.dataframe(output, use_container_width=True)
    st.download_button(
        "下载预测结果 CSV",
        data=output.to_csv(index=False).encode("utf-8-sig"),
        file_name="batch_emotion_predictions.csv",
        mime="text/csv",
    )


def show_time_trajectory():
    st.caption("适用于情绪变化表：必须包含 timestamp 和 text，可选 target / trigger_type。")
    uploaded_file = st.file_uploader(
        "上传时间轨迹 CSV / Excel", type=["csv", "xlsx", "xls"], key="time_trajectory_upload"
    )
    if uploaded_file is None:
        return

    source = read_uploaded_table(uploaded_file)
    if source is None:
        return

    st.markdown("#### 原始数据预览")
    st.dataframe(source.head(20), use_container_width=True)

    required = {"timestamp", "text"}
    missing = sorted(required - set(source.columns))
    if missing:
        st.error(f"时间轨迹分析文件缺少必要列：{', '.join(missing)}。")
        return

    working = source.copy()
    working["timestamp"] = pd.to_datetime(working["timestamp"], errors="coerce")
    invalid_time = working["timestamp"].isna()
    if invalid_time.any():
        st.warning(f"有 {int(invalid_time.sum())} 行 timestamp 无法解析，已从分析中排除。")
        working = working.loc[~invalid_time].copy()
    if working.empty:
        st.error("没有可解析 timestamp 的有效记录。")
        return

    if "target" not in working.columns:
        working["target"] = "未标注"
    else:
        working["target"] = working["target"].fillna("未标注").astype(str).str.strip().replace("", "未标注")

    if "trigger_type" not in working.columns:
        working["trigger_type"] = "未标注"
    else:
        working["trigger_type"] = (
            working["trigger_type"].fillna("未标注").astype(str).str.strip().replace("", "未标注")
        )

    working = working.sort_values("timestamp").reset_index(drop=True)
    output, valid_rows = predict_dataframe(working)
    if not valid_rows.all():
        st.warning("空文本行不会参与预测，趋势图和峰值事件只使用有效文本。")
    output = output.loc[valid_rows].copy()
    if output.empty:
        st.error("没有可预测的有效文本。")
        return

    for column in PREDICTION_COLUMNS + ["emotion_intensity", "emotion_angle"]:
        output[column] = pd.to_numeric(output[column], errors="coerce")

    st.markdown("#### 时间排序后的预测结果表格")
    st.dataframe(output, use_container_width=True)

    trend_col, intensity_col = st.columns(2)
    with trend_col:
        st.pyplot(
            draw_time_trend(
                output,
                [("pred_valence", "valence（愉悦度）", "#0f766e"), ("pred_arousal", "arousal（唤醒度）", "#e11d48")],
                "Valence（愉悦度）/ Arousal（唤醒度）时间趋势",
                "坐标值",
            )
        )
    with intensity_col:
        st.pyplot(
            draw_time_trend(
                output,
                [("emotion_intensity", "intensity", "#7c3aed")],
                "情绪强度时间趋势",
                "强度",
            )
        )

    st.markdown("#### 峰值事件")
    peak_events = output.loc[
        output["emotion_intensity"] >= 0.75,
        [
            "timestamp",
            "text",
            "target",
            "trigger_type",
            "main_emotion",
            "pred_valence",
            "pred_arousal",
            "emotion_intensity",
            "intensity_level",
        ],
    ]
    if peak_events.empty:
        st.caption("当前没有强度达到 0.75 的峰值事件。")
    else:
        st.dataframe(peak_events, use_container_width=True)

    st.markdown("#### 按对象分组的平均情绪强度")
    target_summary = (
        output.groupby("target", dropna=False)
        .agg(
            records=("text", "count"),
            avg_intensity=("emotion_intensity", "mean"),
            avg_valence=("pred_valence", "mean"),
            avg_arousal=("pred_arousal", "mean"),
        )
        .sort_values("avg_intensity", ascending=False)
        .reset_index()
    )
    st.dataframe(target_summary, use_container_width=True)

    st.download_button(
        "下载轨迹分析结果 CSV",
        data=output.to_csv(index=False).encode("utf-8-sig"),
        file_name="time_trajectory_analysis.csv",
        mime="text/csv",
    )


st.set_page_config(page_title="心理测量者 v2", layout="wide")
st.markdown(
    """
    <style>
    .state-card {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 0;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        overflow: hidden;
        margin: 0.25rem 0 1rem;
        background: #ffffff;
    }
    .state-cell {
        padding: 0.8rem 0.95rem;
        border-right: 1px solid #f1f5f9;
        border-bottom: 1px solid #f1f5f9;
        min-height: 4.25rem;
    }
    .state-label {
        color: #64748b;
        font-size: 0.82rem;
        margin-bottom: 0.25rem;
    }
    .state-value {
        color: #0f172a;
        font-size: 1.28rem;
        line-height: 1.3;
        font-weight: 600;
        word-break: break-word;
    }
    @media (max-width: 640px) {
        .state-card { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .state-cell { padding: 0.7rem 0.75rem; }
        .state-value { font-size: 1.08rem; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)
st.title("心理测量者 v2：八方向情绪罗盘")

try:
    predict_one("收到")
except FileNotFoundError:
    st.error("未找到 emotion_model.pkl 或 vectorizer.pkl，请先运行 train_simple.py。")
    st.stop()
except Exception as exc:
    st.error(f"本地模型加载失败：{exc}")
    st.stop()

single_tab, batch_tab, time_tab = st.tabs(["单句检测", "批量情绪检测", "时间轨迹分析"])

with single_tab:
    show_single_detection()

with batch_tab:
    show_batch_detection()

with time_tab:
    show_time_trajectory()
