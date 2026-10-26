# 🥗 Food Verse AI – Multimodal Food Intelligence Agent

A Streamlit app that combines computer vision, video analysis and LLM reasoning to give
real-time insights on food, freshness, kitchen hygiene, cooking and nutrition.

**Live demo:** _add Streamlit Cloud link here_
**Demo video:** _add link here_

## Problem
- Calorie tracking depends on manual guesswork.
- Freshness is judged by eye, with no consistent scoring.
- Restaurant hygiene checks are periodic, not continuous.
- Recipes are static and ignore what the user actually has.
- Diet advice is rarely personalized to the person's body and goals.

## Solution
One modular platform where a multimodal LLM (OpenAI GPT-4o family) handles vision and
reasoning, and Python handles the calculations and structure.

## Features
- **Food & Calories**
  - Photo or camera input
  - Per-item calories, protein, carbs, fat with confidence level
  - Macro chart and a running daily meal log
- **Freshness Detection**
  - Verdict (Fresh / Ripe / Overripe / Spoiled) and a 0-100 score
  - Visible signs, shelf-life estimate, storage tip
- **Hygiene Monitoring**
  - Image or short video (4 frames sampled with OpenCV)
  - Violations with severity and fix recommendations
  - Hygiene score and downloadable report
- **Smart Cooking Assistant**
  - Type ingredients or detect them from a photo
  - Diet and time limit filters
  - Streaming chat for follow-ups (make it spicier, swap an ingredient)
- **Personalized Nutrition Plan**
  - BMI, BMR, TDEE and calorie/macro targets calculated in Python (Mifflin-St Jeor)
  - LLM builds a one-day meal plan that matches those targets
  - Target vs plan macro chart

## Architecture
- **UI:** Streamlit (tabs, chat, charts, camera input)
- **Vision + language:** OpenAI chat completions with image input
- **Structured output:** JSON mode with a fixed schema per module, parsed into tables
- **Video:** OpenCV extracts evenly spaced frames, sent as a multi-image prompt
- **Grounding:** nutrition targets are computed in code, then given to the LLM
- **State:** `st.session_state` for meal log, chat history and results

## Project structure
- `app.py`: Streamlit UI and OpenAI calls
- `logic.py`: pure helpers (nutrition maths, image prep, video frame sampling, JSON cleanup)
- `tests/test_logic.py`: automated tests for `logic.py`
- `.streamlit/secrets.toml.example`: template for your API key

## Run locally
- `git clone <your-repo-url>` and open the folder
- `pip install -r requirements.txt`
- Copy `.streamlit/secrets.toml.example` to `.streamlit/secrets.toml` and add your key
- `streamlit run app.py`

## Run the tests
- `python tests/test_logic.py`
- Covers nutrition formulas, JSON parsing, image preparation, video frame extraction and bad-input handling

## Reliability features
- Images are auto-rotated, resized to 1024 px and re-encoded as JPEG before sending (lower cost, fewer errors)
- Model output is validated: missing or non-numeric fields fall back to safe defaults
- Friendly messages for invalid key, no credit, network failure and unreadable answers
- Calorie target never goes below a safe minimum (1500 kcal men / 1200 kcal women)
- Optional `APP_PASSWORD` secret to protect your API credit on a public deployment

## Deploy on Streamlit Community Cloud
- Push the project to a public GitHub repo (make sure `secrets.toml` is NOT pushed)
- Go to share.streamlit.io and sign in with GitHub
- Click **New app**, select the repo, branch `main`, main file `app.py`
- Open **Advanced settings → Secrets** and paste: `OPENAI_API_KEY = "sk-..."`
- Optional but recommended: also add `APP_PASSWORD = "your-code"` so only people with the code can use the app
- Click **Deploy** and copy the app URL into the project page

## Limitations
- All outputs are AI estimates and can be wrong; not medical advice.
- Hygiene check is not an official inspection and works on sampled frames, not live streams.
- Calorie estimates depend on image quality and portion visibility.
- Uses a paid API, so usage costs money.

## Future scope
- Fine-tuned freshness classifier for higher accuracy
- Real-time webcam stream analysis
- User login and saved history in a database
- Barcode scanning and a local food database for exact nutrition values
- Multilingual support

## Screenshots
_Add one screenshot per tab here._
