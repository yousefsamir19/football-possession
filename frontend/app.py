import os
import requests
import streamlit as st

# ---------- Mode switch ----------
# "local"      -> reads the output video directly from disk (backend + frontend on same machine)
# "deployment" -> fetches the output video from the backend over HTTP (backend + frontend on different machines/hosts)
MODE = "local"

BACKEND_URL = "http://127.0.0.1:8000"
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")

st.set_page_config(page_title="Football Possession Analyzer", page_icon="⚽", layout="centered")

st.title("⚽ Football Possession Analyzer")
st.caption("Upload a match clip and see who ran the game.")

with st.container(border=True):
    uploaded_file = st.file_uploader("Choose a video", type=["mp4", "avi", "mov"])
    analyze_clicked = st.button("Analyze Video", type="primary", use_container_width=True)

if uploaded_file is not None and analyze_clicked:
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
        st.session_state["result"] = response.json()


def get_video_bytes(filename):
    """Return the analyzed video as bytes, using disk or HTTP depending on MODE."""
    if MODE == "local":
        video_path = os.path.join(OUTPUTS_DIR, filename)
        if not os.path.exists(video_path):
            st.error(f"Processed video not found at: {video_path}")
            st.caption("Check that the backend saved the output video to the 'outputs' folder with the filename it returned.")
            return None
        with open(video_path, "rb") as f:
            return f.read()

    # deployment mode: fetch from backend, e.g. GET /download/{filename}
    try:
        video_resp = requests.get(f"{BACKEND_URL}/download/{filename}", timeout=120)
    except requests.exceptions.RequestException as e:
        st.error(f"Could not fetch video from backend: {e}")
        return None

    if video_resp.status_code != 200:
        st.error(f"Could not download video: {video_resp.text}")
        return None

    return video_resp.content


if "result" in st.session_state:
    data = st.session_state["result"]
    a_pct = data["team_a_possession"]
    b_pct = data["team_b_possession"]

    with st.container(border=True):
        st.subheader("Analyzed Video")
        video_bytes = get_video_bytes(data["video"])
        if video_bytes is not None:
            st.video(video_bytes)
            st.download_button(
                label="⬇️ Download analyzed video",
                data=video_bytes,
                file_name=data["video"],
                mime="video/mp4",
                use_container_width=True,
            )

    with st.container(border=True):
        st.subheader("Possession")

        st.write("🟢 Team A")
        st.progress(a_pct / 100, text=f"{a_pct}%")

        st.write("🔵 Team B")
        st.progress(b_pct / 100, text=f"{b_pct}%")

    dominant = "Team A" if a_pct >= b_pct else "Team B"
    st.success(f"🏆 Dominant team: {dominant}")