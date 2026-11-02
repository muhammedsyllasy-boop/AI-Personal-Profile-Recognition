# AI Personal Profile Recognition

A Streamlit + Machine Learning educational prototype that predicts a broad professional domain from a synthetic personal skill profile.

## Features
- Generates 20,000 synthetic profiles.
- Trains and compares Random Forest, Extra Trees, Gradient Boosting, and Logistic Regression.
- Selects and saves the best-performing pipeline.
- Streamlit dashboard, profile prediction, model experiments, algorithm explanations, random simulations, and downloadable reports.
- CSV dataset download.
- Free local AI Photo Analyzer using the BLIP image-captioning model (no OpenAI API key or paid API credits).

## Project structure
```text
AI_Personal_Profile_Recognition/
├── app.py
├── generate_dataset.py
├── train_model.py
├── requirements.txt
├── README.md
├── data/
└── models/
```

## Setup (Windows PowerShell)
```powershell
cd path\to\AI_Personal_Profile_Recognition
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python generate_dataset.py
python train_model.py
streamlit run app.py
```

If PowerShell blocks activation, you can skip activation and run:
```powershell
py -m pip install -r requirements.txt
py generate_dataset.py
py train_model.py
py -m streamlit run app.py
```

## First launch
The app automatically creates the dataset and trains models if the generated files are missing. This can take a few minutes on the first run.

## Important note
The dataset is synthetic and the model is for demonstration and coursework only. It is not a scientifically validated career test and should not be used for hiring, admissions, or other high-impact decisions.


## AI Photo Analyzer (online vision feature)

The Streamlit app includes an **AI Photo Analyzer** page. It can describe visible clothing,
accessories, pose, activity, setting, and objects in an uploaded image.

### Configure
1. Install dependencies: `python -m pip install -r requirements.txt`
2. Start the app: `python -m streamlit run app.py`
3. Open **AI Photo Analyzer**, upload a JPG/PNG/WebP image, enter your OpenAI API key,
   select a vision model, and click **Analyze photo**.

The API key is entered into the password field at runtime and is not saved by this app.
You can also configure a key using Streamlit secrets or environment variables in a future
deployment setup. Online analysis requires internet access and an OpenAI API account with
API access/billing; ChatGPT subscriptions and API billing are separate.

### Responsible-use and privacy limits
The feature describes observable image content only. It does not identify a person or infer
their name, address, social accounts, sensitive traits, health, personality, or credentials.
Only upload images you have permission to use. The image is sent to the selected AI service
for analysis when you click the button.


## Free AI Photo Analyzer
The AI Photo Analyzer uses `Salesforce/blip-image-captioning-base` through Hugging Face Transformers. It runs locally on your computer. The first run requires an internet connection to download the model files (roughly 1 GB); later runs use the cached model. No OpenAI API key or API credits are required. CPU inference is supported but may be slower.

The analyzer generates basic captions, not verified facts. It must not be used to identify people or infer sensitive/private characteristics.
