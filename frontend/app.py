import os
import requests
import streamlit as st

BACKEND_URL = "http://127.0.0.1:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(page_title="Football Possession Analyzer", layout="centered")

st.title("⚽ Football Possession Analyzer")
st.write("Upload a football video to analyze team possession.")

uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    if st.button("Analyze Video"):
        with st.spinner("Analyzing video... this may take a while"):
            files = {"file": (uploaded_file.name, uploaded_file.getvalue())}
            try:
                response = requests.post(f"{BACKEND_URL}/analyze", files=files, timeout=3600)
            except requests.exceptions.RequestException as e:
                st.error(f"Could not reach backend: {e}")
                st.stop()

        if response.status_code != 200:
            st.error(f"Analysis failed: {response.text}")
        else:
            data = response.json()
            st.session_state["result"] = data

if "result" in st.session_state:
    data = st.session_state["result"]

    st.subheader("Analyzed Video")
    video_path = os.path.join(OUTPUTS_DIR, data["video"])
    if os.path.exists(video_path):
        st.video(video_path)
    else:
        st.warning("Processed video not found on disk.")

    st.subheader("Possession")

    st.write(f"Team A — {data['team_a_possession']}%")
    st.progress(data["team_a_possession"] / 100)

    st.write(f"Team B — {data['team_b_possession']}%")
    st.progress(data["team_b_possession"] / 100)

    st.markdown(f"**Players detected:** {data['players_detected']}")
    st.markdown(f"**Current possession:** {data['current_possession']}")
