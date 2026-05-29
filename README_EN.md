# PSYCHO-PASS Emotion Compass

PSYCHO-PASS Emotion Compass is an experimental Chinese short-text emotion detection and public-opinion emotion monitoring project based on the Russell Circumplex Model of Affect.

Given a Chinese text input, the system predicts `valence` (pleasantness), `arousal` (activation), emotional direction, emotional intensity, and timestamped records. It is designed to help observe emotional trends in comment sections, user feedback, bullet comments, social text, and similar short-text scenarios.

The current version focuses on Chinese emotion recognition and emotion compass visualization. Future versions may explore relationship-risk analysis, emotional timelines, sarcasm recognition, and operations-oriented monitoring.

## Project Positioning

This project is not a psychological diagnosis tool, medical tool, or relationship decision-making tool.

It is closer to a lightweight prototype for Chinese emotion recognition, public-opinion emotion monitoring, and text-analysis experimentation.

The current version focuses on Chinese short-text emotion quantification and visualization. Relationship-risk analysis is only a possible future extension.

## Russell Circumplex Model

The Russell Circumplex Model represents affective states in a two-dimensional space:

- `valence`: pleasantness, ranging from `-1` to `1`
- `arousal`: activation, ranging from `-1` to `1`

The system maps the two-dimensional output into an eight-direction emotion compass:

- Happiness / Satisfaction
- Excitement / Euphoria
- Tension / Alertness
- Anger / Agitation
- Disappointment / Annoyance
- Sadness / Fatigue
- Numbness / Calmness
- Relaxation / Reassurance

For low-intensity points near the center of the circle, the system avoids forcing a strong emotion label.

## Features

- Single-sentence emotion prediction
- Batch CSV / Excel emotion prediction
- Time-trajectory analysis
- Eight-direction emotion compass visualization
- `pred_valence` and `pred_arousal` outputs
- Emotion angle, intensity, intensity level, and main emotion label
- Short-text fallback rules
- Sarcasm fallback rules
- CSV result download
- Test-set evaluation script
- Badcase analysis script
- Dataset cleaning and invalid-label isolation script

## Technical Approach

The final prediction system uses a lightweight local machine-learning approach.

Runtime inference does not call any online large-language-model API and does not depend on `transformers` or HuggingFace.

During dataset construction, online large-language models were used to assist in generating part of the anonymized Chinese short-text samples. The generated samples were then processed through manual rule constraints, format cleaning, privacy filtering, test-set evaluation, and badcase iteration. Online large models were only used as a data-generation aid and are not involved in final model inference.

Technology stack:

- Character-level TF-IDF ngrams
- RandomForestRegressor
- MultiOutputRegressor
- Streamlit
- pandas
- matplotlib
- scikit-learn

## Repository Structure

```text
emotion_project/
  app.py                     # Streamlit UI and visualization
  emotion_core.py            # Local prediction core
  train_simple.py            # Local TF-IDF + regression training
  clean_dataset.py           # Build data_clean.csv and data_invalid_rows.csv
  check_dataset.py           # Dataset distribution checks
  quick_test.py              # Quick smoke test
  evaluate_test.py           # Evaluation script
  analyze_test_failures.py   # Failure and bias analysis
  demo_data.csv              # Public demo data format
  demo_test.csv              # Public demo test format
  README.md
  README_EN.md
  eval_report.md
  requirements.txt
```

## Data Statement

The full experimental dataset contains roughly 6k samples, including self-built Chinese short-text examples and samples partially assisted by online large-language-model generation.

For privacy and data-ethics reasons, the full training dataset is not released. This repository only provides `demo_data.csv` to demonstrate the data format and local workflow.

Training-data format:

```text
text,valence,arousal,emotion
```

Test-data format:

```text
text,expected_valence_min,expected_valence_max,expected_arousal_min,expected_arousal_max,expected_emotion,case_type
```

## Current Evaluation Snapshot

The following numbers are a stage-level snapshot:

- `data_clean.csv`: 6661 valid rows
- `data_invalid_rows.csv`: 207 invalid rows
- Original `data.csv`: 6877 rows
- Removed duplicate rows: 9
- Overall both-pass rate: 48.33%
- Valence pass rate: 61.67%
- Arousal pass rate: 67.50%

Important note: the pass rate is based on expected `valence / arousal` intervals in a self-built test set. It is not equivalent to standard accuracy in general-purpose emotion classification tasks. The result is only used to observe stage-level behavior and iteration directions across different `case_type` groups.

Both-pass rate by case type:

- Neutral daily statements: 86.67%
- Low-arousal negative: 46.67%
- Relaxation / Reassurance / Relief: 53.33%
- Explicit positive short texts: 33.33%
- Explicit negative short texts: 26.67%
- Mixed complex emotions: 26.67%
- Sarcasm / Ironic tone: 73.33%
- High-arousal positive: 40.00%

## Local Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

Clean data:

```bash
python clean_dataset.py
```

Train model:

```bash
python train_simple.py
```

Run quick test:

```bash
python quick_test.py
```

Start the Streamlit app:

```bash
streamlit run app.py
```

## Usage Restrictions

This project is provided for personal learning, academic research, portfolio display, and non-commercial experimentation only.

Without explicit written permission from the author, this project or derivative versions may not be used for commercial products, enterprise internal production systems, paid consulting, recruitment screening, psychological diagnosis, medical diagnosis, judicial decision-making, credit evaluation, or other high-risk/profit-oriented scenarios.

If you would like to use this project for commercial cooperation, enterprise tools, content-operations systems, or other profit-oriented scenarios, please contact the author for authorization first.

## Privacy and Data Ethics

This project is intended for short-text emotion quantification experiments. It should not be used for medical diagnosis, psychological diagnosis, recruitment screening, credit evaluation, judicial judgment, or other high-risk decision-making scenarios.

The full training data is not publicly released to reduce privacy, misuse, and distribution-risk concerns.

## Author Note

This is an early-stage personal experimental project.

It is not recommended for serious decision-making scenarios in its current form.

If you find it useful, stars, issues, discussions, and job offers are all welcome.

Latest updates: WeChat public account 「贫道有点闲」
