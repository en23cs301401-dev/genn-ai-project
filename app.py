"""Food Verse AI - Multimodal Food Intelligence Agent (Streamlit)."""
import hmac
import json
import os

import pandas as pd
import streamlit as st
from openai import (
    APIConnectionError,
    AuthenticationError,
    BadRequestError,
    OpenAI,
    RateLimitError,
)

from logic import (
    extract_frames,
    nutrition_targets,
    parse_json,
    prepare_image,
    rows_to_df,
    safe_int,
    to_data_url,
    totals,
)

st.set_page_config(page_title="Food Verse AI", page_icon="🥗", layout="wide")

SYSTEM_JSON = (
    "You are a careful food-intelligence assistant. "
    "Reply with a single valid JSON object only, matching the requested schema. "
    "Never invent details you cannot see; use lower confidence when unsure."
)


# ----------------------------------------------------------------------------
# Secrets, access control, API helpers
# ----------------------------------------------------------------------------
def secret(name):
    try:
        return st.secrets.get(name)
    except Exception:  # no secrets file present
        return None


def check_access():
    """Optional access code: set APP_PASSWORD in secrets to protect your API credit."""
    code = secret("APP_PASSWORD")
    if not code or st.session_state.get("authed"):
        return
    st.title("🥗 Food Verse AI")
    entered = st.text_input("Access code", type="password")
    if entered and hmac.compare_digest(entered, str(code)):
        st.session_state.authed = True
        st.rerun()
    if entered:
        st.error("Wrong access code.")
    st.stop()


def get_client():
    key = secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY") or st.session_state.get("manual_key")
    if not key:
        st.warning("Add your OpenAI API key in the sidebar (or in Streamlit secrets) to continue.")
        st.stop()
    return OpenAI(api_key=key, timeout=60)


def model_name():
    return st.session_state.get("model", "gpt-4o-mini")


def ask_json(prompt, images=None):
    """Send text (+ optional images) to the model and return the parsed JSON reply."""
    content = [{"type": "text", "text": prompt}]
    for url in images or []:
        content.append({"type": "image_url", "image_url": {"url": url}})
    response = get_client().chat.completions.create(
        model=model_name(),
        messages=[
            {"role": "system", "content": SYSTEM_JSON},
            {"role": "user", "content": content},
        ],
        response_format={"type": "json_object"},
        temperature=0.2,
        max_tokens=1200,
    )
    result = parse_json(response.choices[0].message.content)
    if not isinstance(result, dict):
        raise json.JSONDecodeError("not an object", "", 0)
    return result


def stream_chat(messages):
    stream = get_client().chat.completions.create(
        model=model_name(), messages=messages, stream=True, temperature=0.7
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            yield chunk.choices[0].delta.content


def run(fn):
    """Run an action and show friendly errors instead of crashing the app."""
    try:
        return fn()
    except AuthenticationError:
        st.error("The OpenAI API key is invalid. Check the key and try again.")
    except RateLimitError:
        st.error("Rate limit or no credit left on this OpenAI account. Check billing and retry.")
    except APIConnectionError:
        st.error("Could not reach OpenAI. Check your internet connection and retry.")
    except BadRequestError as e:
        st.error(f"The request was rejected (the image may be unsupported): {e}")
    except json.JSONDecodeError:
        st.error("The model returned an unreadable answer. Please try again.")
    except Exception as e:  # noqa: BLE001
        st.error(f"Something went wrong: {e}")
    return None


@st.cache_data(show_spinner=False)
def cached_frames(video_bytes, n):
    return extract_frames(video_bytes, n)


def image_input(label, key):
    """Upload or camera input. Returns a ready-to-send data URL, or None."""
    mode = st.radio("Input method", ["Upload", "Camera"], horizontal=True, key=f"{key}_mode")
    if mode == "Upload":
        file = st.file_uploader(label, type=["jpg", "jpeg", "png", "webp"], key=f"{key}_up")
    else:
        file = st.camera_input(label, key=f"{key}_cam")
    if file is None:
        return None
    try:
        data_url = to_data_url(prepare_image(file.getvalue()))
    except Exception:  # noqa: BLE001
        st.error("Could not read this image. Please use a JPG, PNG or WEBP file.")
        return None
    st.image(file)
    return data_url


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
check_access()

with st.sidebar:
    st.title("🥗 Food Verse AI")
    st.caption("Multimodal Food Intelligence Agent")
    st.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], key="model")
    if not (secret("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")):
        st.text_input("OpenAI API key", type="password", key="manual_key")
    st.divider()
    st.caption(
        "⚠️ Results are AI estimates. This is not medical advice or an official "
        "food-safety inspection."
    )

st.title("Food Verse AI")
st.write("Vision + language AI for food recognition, freshness, hygiene, cooking and nutrition.")

tab_food, tab_fresh, tab_hyg, tab_cook, tab_plan = st.tabs(
    ["🍛 Food & Calories", "🍎 Freshness", "🧼 Hygiene", "👨‍🍳 Cooking Assistant", "📋 Nutrition Plan"]
)

# ----------------------------------------------------------------------------
# 1. Food recognition + calories
# ----------------------------------------------------------------------------
with tab_food:
    st.subheader("Food recognition and calorie estimation")
    left, right = st.columns(2)
    with left:
        img = image_input("Food photo", "food")
    with right:
        if img and st.button("Analyze food", type="primary", key="food_go"):
            prompt = (
                "Identify every food item in the image. Return JSON: "
                '{"is_food": bool, "items": [{"name": str, "portion": str, "calories": int, '
                '"protein_g": number, "carbs_g": number, "fat_g": number}], '
                '"confidence": "low|medium|high", "health_note": str}. '
                "Estimate for the visible portion."
            )
            with st.spinner("Analyzing..."):
                st.session_state.food_result = run(lambda: ask_json(prompt, [img]))

        res = st.session_state.get("food_result")
        if res:
            df = rows_to_df(res.get("items"))
            if not res.get("is_food", True) or df.empty:
                st.info("No food detected in this image.")
            else:
                st.dataframe(df, hide_index=True)
                total = totals(df)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Calories", f"{total['calories']:.0f}")
                c2.metric("Protein (g)", f"{total['protein_g']:.0f}")
                c3.metric("Carbs (g)", f"{total['carbs_g']:.0f}")
                c4.metric("Fat (g)", f"{total['fat_g']:.0f}")
                st.bar_chart(
                    pd.Series(
                        {"Protein": total["protein_g"], "Carbs": total["carbs_g"], "Fat": total["fat_g"]}
                    )
                )
                st.caption(f"Confidence: {res.get('confidence', 'n/a')}")
                if res.get("health_note"):
                    st.info(res["health_note"])
                if st.button("➕ Add to today's meal log", key="food_log"):
                    names = df["name"].astype(str).tolist() if "name" in df.columns else ["Meal"]
                    st.session_state.setdefault("meal_log", []).append({"meal": ", ".join(names), **total})
                    st.success("Added.")

    log = st.session_state.get("meal_log", [])
    if log:
        st.divider()
        st.subheader("Today's meal log")
        log_df = pd.DataFrame(log)
        st.dataframe(log_df, hide_index=True)
        st.metric("Total calories today", f"{log_df['calories'].sum():.0f} kcal")
        if st.button("Clear log", key="food_clear"):
            st.session_state.meal_log = []
            st.rerun()

# ----------------------------------------------------------------------------
# 2. Freshness detection
# ----------------------------------------------------------------------------
with tab_fresh:
    st.subheader("Freshness detection")
    left, right = st.columns(2)
    with left:
        img = image_input("Fruit or vegetable photo", "fresh")
    with right:
        if img and st.button("Check freshness", type="primary", key="fresh_go"):
            prompt = (
                "Assess the freshness of the produce in the image. Return JSON: "
                '{"item": str, "verdict": "Fresh|Ripe|Overripe|Spoiled", '
                '"freshness_score": int (0-100), "visible_signs": [str], '
                '"shelf_life_days": int, "storage_tip": str, "safe_to_eat": bool}'
            )
            with st.spinner("Inspecting..."):
                st.session_state.fresh_result = run(lambda: ask_json(prompt, [img]))

        res = st.session_state.get("fresh_result")
        if res:
            st.markdown(f"### {res.get('item', 'Item')} - {res.get('verdict', '?')}")
            score = safe_int(res.get("freshness_score"))
            st.progress(score / 100, text=f"Freshness score: {score}/100")
            c1, c2 = st.columns(2)
            c1.metric("Estimated shelf life", f"{res.get('shelf_life_days', '?')} days")
            c2.metric("Safe to eat", "Yes" if res.get("safe_to_eat") else "No")
            st.write("**Visible signs**")
            for s in res.get("visible_signs", []) or []:
                st.write(f"- {s}")
            if res.get("storage_tip"):
                st.info(f"💡 {res['storage_tip']}")

# ----------------------------------------------------------------------------
# 3. Hygiene monitoring (image or video)
# ----------------------------------------------------------------------------
with tab_hyg:
    st.subheader("Restaurant hygiene monitoring")
    source = st.radio("Source", ["Image", "Video"], horizontal=True, key="hyg_source")
    left, right = st.columns(2)
    images = []
    with left:
        if source == "Image":
            img = image_input("Kitchen photo", "hyg")
            if img:
                images = [img]
        else:
            video = st.file_uploader("Short kitchen video (mp4 recommended)", type=["mp4", "mov", "avi"], key="hyg_vid")
            if video:
                frames = cached_frames(video.getvalue(), 4)
                if frames:
                    st.caption(f"Analyzing {len(frames)} frames sampled from the video")
                    for col, fr in zip(st.columns(len(frames)), frames):
                        col.image(fr)
                    images = [to_data_url(prepare_image(fr)) for fr in frames]
                else:
                    st.error("Could not read frames from this video. Try an mp4 file.")
    with right:
        if images and st.button("Run hygiene check", type="primary", key="hyg_go"):
            prompt = (
                "You are a food-safety inspector. Review the kitchen image(s) and return JSON: "
                '{"violations": [{"issue": str, "severity": "Low|Medium|High", "recommendation": str}], '
                '"positives": [str], "hygiene_score": int (0-100), "summary": str}. '
                "Only report what is visible: gloves, hairnets, aprons, surface cleanliness, "
                "food covering, cross-contamination, pests, waste handling."
            )
            with st.spinner("Inspecting..."):
                st.session_state.hyg_result = run(lambda: ask_json(prompt, images))

        res = st.session_state.get("hyg_result")
        if res:
            score = safe_int(res.get("hygiene_score"))
            st.progress(score / 100, text=f"Hygiene score: {score}/100")
            st.write(res.get("summary", ""))
            violations = [v for v in (res.get("violations") or []) if isinstance(v, dict)]
            if violations:
                st.dataframe(pd.DataFrame(violations), hide_index=True)
            else:
                st.success("No visible violations found.")
            for p in res.get("positives", []) or []:
                st.write(f"✅ {p}")

            report = [f"Hygiene report - score {score}/100", "", str(res.get("summary", "")), "", "Violations:"]
            for v in violations:
                report.append(f"- [{v.get('severity')}] {v.get('issue')} -> {v.get('recommendation')}")
            st.download_button("⬇️ Download report", "\n".join(report), "hygiene_report.txt")

# ----------------------------------------------------------------------------
# 4. Smart cooking assistant (streaming chat)
# ----------------------------------------------------------------------------
with tab_cook:
    st.subheader("Smart cooking assistant")
    left, right = st.columns(2)
    with left:
        ingredients = st.text_area("Ingredients you have", placeholder="e.g. paneer, onion, tomato, rice")
        with st.expander("Or detect ingredients from a photo"):
            fridge = image_input("Fridge / ingredients photo", "fridge")
        diet = st.selectbox("Diet", ["Vegetarian", "Non-vegetarian", "Vegan", "Eggetarian"], key="cook_diet")
        minutes = st.slider("Max cooking time (minutes)", 10, 90, 30, step=5)
        if st.button("Generate recipe", type="primary", key="cook_go"):
            items = ingredients
            if fridge:
                with st.spinner("Detecting ingredients..."):
                    found = run(
                        lambda: ask_json('List the food ingredients visible. Return JSON: {"ingredients": [str]}', [fridge])
                    )
                if found:
                    detected = [str(x) for x in (found.get("ingredients") or [])]
                    items = ", ".join(x for x in [items.strip(), *detected] if x)
            if not items.strip():
                st.warning("Enter or photograph some ingredients first.")
            else:
                st.session_state.chat = [
                    {
                        "role": "system",
                        "content": (
                            "You are a friendly chef. Give recipes with: title, time, ingredients "
                            "(note what is missing), numbered steps, and an approximate calorie count "
                            "per serving. Keep answers concise and use markdown."
                        ),
                    },
                    {
                        "role": "user",
                        "content": f"Ingredients: {items}. Diet: {diet}. Max time: {minutes} minutes. Suggest one recipe.",
                    },
                ]
                st.session_state.pending = True

    with right:
        chat = st.session_state.get("chat", [])
        for m in chat:
            if m["role"] != "system":
                with st.chat_message(m["role"]):
                    st.markdown(m["content"])
        if st.session_state.get("pending"):
            st.session_state.pending = False
            with st.chat_message("assistant"):
                reply = run(lambda: st.write_stream(stream_chat(chat)))
            if reply:
                chat.append({"role": "assistant", "content": reply})
        if chat:
            with st.form("cook_form", clear_on_submit=True):
                follow = st.text_input("Ask a follow-up (e.g. make it spicier, swap paneer)")
                sent = st.form_submit_button("Send")
            if sent and follow.strip():
                chat.append({"role": "user", "content": follow.strip()})
                st.session_state.pending = True
                st.rerun()

# ----------------------------------------------------------------------------
# 5. Personalized nutrition plan (calculation + LLM)
# ----------------------------------------------------------------------------
with tab_plan:
    st.subheader("Personalized nutrition plan")
    st.caption("Calculations use the Mifflin-St Jeor equation and are meant for adults (18+).")
    c1, c2, c3 = st.columns(3)
    age = c1.number_input("Age", 18, 90, 22)
    sex = c1.selectbox("Sex", ["Male", "Female"])
    height = c2.number_input("Height (cm)", 120, 220, 170)
    weight = c2.number_input("Weight (kg)", 30.0, 200.0, 65.0)
    activity = c3.selectbox("Activity level", ["Sedentary", "Light", "Moderate", "Very active"])
    goal = c3.selectbox("Goal", ["Weight loss", "Maintain weight", "Muscle gain"])
    d1, d2 = st.columns(2)
    diet = d1.selectbox("Diet preference", ["Vegetarian", "Non-vegetarian", "Vegan", "Eggetarian"], key="plan_diet")
    cuisine = d2.selectbox("Cuisine", ["Indian", "Mediterranean", "Continental", "Asian", "Any"])
    avoid = st.text_input("Allergies / foods to avoid (optional)")

    if st.button("Generate plan", type="primary", key="plan_go"):
        t = nutrition_targets(age, sex, height, weight, activity, goal)
        st.session_state.plan_targets = t
        st.session_state.plan_result = None
        prompt = (
            f"Create a one-day meal plan for a {age}-year-old {sex.lower()}, {diet}, "
            f"{cuisine} cuisine, goal: {goal}. Avoid: {avoid or 'nothing specific'}. "
            f"Daily target: about {t['Target kcal']} kcal, {t['Protein g']} g protein, "
            f"{t['Carbs g']} g carbs, {t['Fat g']} g fat. Return JSON: "
            '{"meals": [{"meal": "Breakfast|Lunch|Snack|Dinner", "dish": str, "description": str, '
            '"calories": int, "protein_g": number, "carbs_g": number, "fat_g": number}], '
            '"tips": [str]}'
        )
        with st.spinner("Building your plan..."):
            st.session_state.plan_result = run(lambda: ask_json(prompt))

    t = st.session_state.get("plan_targets")
    if t:
        for col, (k, v) in zip(st.columns(len(t)), t.items()):
            col.metric(k, v)
    plan = st.session_state.get("plan_result")
    if t and plan:
        pdf = rows_to_df(plan.get("meals"))
        if pdf.empty:
            st.warning("The model returned no meals. Please try again.")
        else:
            st.dataframe(pdf, hide_index=True)
            tot = totals(pdf)
            st.write(f"**Plan total:** {tot['calories']:.0f} kcal (target {t['Target kcal']} kcal)")
            st.bar_chart(
                pd.DataFrame(
                    {
                        "Target": [t["Protein g"], t["Carbs g"], t["Fat g"]],
                        "Plan": [tot["protein_g"], tot["carbs_g"], tot["fat_g"]],
                    },
                    index=["Protein", "Carbs", "Fat"],
                )
            )
            for tip in plan.get("tips", []) or []:
                st.write(f"💡 {tip}")
