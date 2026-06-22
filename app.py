"""
HandSense AI - Learned Non-Use Detection
Digital Aiding 4 Aging Hackathon 2026 | AI Vibe Coding
"""

import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import requests

import mediapipe as mp
from mediapipe.tasks import python as mp_tasks
from mediapipe.tasks.python import vision as mp_vision

# --- Model ---
MODEL_PATH = "/tmp/hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


@st.cache_resource
def ensure_model():
    if not os.path.exists(MODEL_PATH):
        try:
            resp = requests.get(MODEL_URL, timeout=120, stream=True)
            resp.raise_for_status()
            with open(MODEL_PATH, "wb") as f:
                for chunk in resp.iter_content(chunk_size=65536):
                    f.write(chunk)
        except Exception as exc:
            raise RuntimeError(f"ดาวน์โหลดโมเดลล้มเหลว: {exc}") from exc
    return MODEL_PATH


# --- Metric Functions ---

def calc_speed(positions):
    if len(positions) < 2:
        return 0.0
    dists = [np.linalg.norm(np.array(positions[i]) - np.array(positions[i - 1]))
             for i in range(1, len(positions))]
    return float(np.mean(dists))


def calc_smoothness(positions):
    if len(positions) < 5:
        return 0.0
    p = np.array(positions)
    vel = np.diff(p, axis=0)
    acc = np.diff(vel, axis=0)
    jerk = np.diff(acc, axis=0)
    mean_jerk = np.mean(np.linalg.norm(jerk, axis=1))
    return float(1.0 / (mean_jerk + 1e-6))


def calc_range_of_motion(positions):
    if len(positions) < 2:
        return 0.0
    p = np.array(positions)
    rom = np.max(p, axis=0) - np.min(p, axis=0)
    return float(np.linalg.norm(rom))


def calc_finger_spread(spreads):
    return float(np.mean(spreads)) if spreads else 0.0


# --- Video Analysis ---

def analyze_video(video_path, progress_bar=None):
    ensure_model()  # download if needed
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    data = {
        "Left":  {"positions": [], "spreads": [], "frame_count": 0},
        "Right": {"positions": [], "spreads": [], "frame_count": 0},
    }

    options = mp_vision.HandLandmarkerOptions(
        base_options=mp_tasks.BaseOptions(model_asset_path=MODEL_PATH),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.55,
        min_tracking_confidence=0.55,
    )

    frame_idx = 0
    with mp_vision.HandLandmarker.create_from_options(options) as landmarker:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_idx += 1
            if progress_bar and total_frames > 0:
                progress_bar.progress(min(frame_idx / total_frames, 1.0))

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
            timestamp_ms = int(frame_idx * 1000 / fps)

            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks and result.handedness:
                for lm_list, handedness_list in zip(result.hand_landmarks, result.handedness):
                    side = handedness_list[0].category_name  # "Left" or "Right"
                    wrist = lm_list[0]
                    data[side]["positions"].append((wrist.x, wrist.y))
                    data[side]["frame_count"] += 1
                    thumb_tip = lm_list[4]
                    pinky_mcp = lm_list[20]
                    spread = np.sqrt(
                        (thumb_tip.x - pinky_mcp.x) ** 2 +
                        (thumb_tip.y - pinky_mcp.y) ** 2
                    )
                    data[side]["spreads"].append(spread)

    cap.release()
    return data


def compute_metrics(side_data):
    pos = side_data["positions"]
    spr = side_data["spreads"]
    if len(pos) < 5:
        return None
    speed      = calc_speed(pos)
    smoothness = calc_smoothness(pos)
    rom        = calc_range_of_motion(pos)
    spread     = calc_finger_spread(spr)
    smooth_norm = min(smoothness * speed, speed * 3)
    score = speed * 0.40 + smooth_norm * 0.40 + rom * 0.20
    return {
        "speed": speed, "smoothness": smoothness, "rom": rom,
        "spread": spread, "frames": side_data["frame_count"], "score": score,
    }


def diagnose(left_m, right_m):
    if left_m is None and right_m is None:
        return None, "ไม่พบมือในวิดีโอ"
    if left_m is None:
        return "left", "มือซ้ายไม่ถูกใช้งาน — อาจมีสัญญาณ Learned Non-Use"
    if right_m is None:
        return "right", "มือขวาไม่ถูกใช้งาน — อาจมีสัญญาณ Learned Non-Use"
    ratio = left_m["score"] / (right_m["score"] + 1e-9)
    THRESHOLD = 0.70
    if ratio < THRESHOLD:
        pct = round((1 - ratio) * 100, 1)
        return "left", "มือซ้ายคะแนนต่ำกว่ามือขวา " + str(pct) + "% - แนะนำประเมิน Learned Non-Use ที่มือซ้าย"
    elif ratio > (1 / THRESHOLD):
        pct = round((ratio - 1) * 100, 1)
        return "right", "มือขวาคะแนนต่ำกว่ามือซ้าย " + str(pct) + "% - แนะนำประเมิน Learned Non-Use ที่มือขวา"
    else:
        return None, "การเคลื่อนไหวทั้งสองมือสมดุล — ไม่พบสัญญาณ Learned Non-Use"


# --- Streamlit UI ---

def main():
    st.set_page_config(page_title="HandSense AI", page_icon="🖐️", layout="wide")
    st.title("🖐️ HandSense AI")
    st.subheader("ระบบตรวจจับ Learned Non-Use ในผู้สูงอายุผ่านการวิเคราะห์วิดีโอ")
    st.caption("Digital Aiding 4 Aging Hackathon 2026 - AI Vibe Coding - RMUTL NAN")
    st.divider()

    with st.expander("วิธีใช้งาน"):
        st.markdown(
            "1. ถ่ายวิดีโอให้เห็นมือทั้งสองข้าง 30 วินาที - 2 นาที\n"
            "2. อัปโหลดวิดีโอ (.mp4 / .avi / .mov)\n"
            "3. กด วิเคราะห์วิดีโอ\n"
            "4. ดูผลเปรียบเทียบมือซ้าย-ขวาและรายงาน"
        )

    uploaded = st.file_uploader("อัปโหลดวิดีโอ", type=["mp4", "avi", "mov", "mkv"])
    if uploaded is None:
        st.info("กรุณาอัปโหลดวิดีโอเพื่อเริ่มการวิเคราะห์")
        st.stop()

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
        tmp.write(uploaded.read())
        tmp_path = tmp.name

    st.video(tmp_path)
    st.divider()

    if st.button("วิเคราะห์วิดีโอ", type="primary", use_container_width=True):
        try:
            with st.spinner("กำลังประมวลผล..."):
                prog = st.progress(0.0, text="กำลังอ่านวิดีโอ...")
                raw = analyze_video(tmp_path, progress_bar=prog)
                prog.empty()
        except Exception as exc:
            st.error(f"เกิดข้อผิดพลาด: {exc}")
            st.stop()

        lm = compute_metrics(raw["Left"])
        rm = compute_metrics(raw["Right"])
        suspect, msg = diagnose(lm, rm)

        st.divider()
        st.subheader("ผลการวิเคราะห์")
        if suspect is None and "สมดุล" in msg:
            st.success(msg)
        elif suspect is not None:
            st.warning(msg)
        else:
            st.error(msg)

        st.divider()
        col_l, col_r = st.columns(2)

        def show_metrics(col, label, m, highlight=False):
            with col:
                prefix = "🔴 " if highlight else ""
                st.markdown("### " + prefix + "มือ" + label)
                if m is None:
                    st.error("ไม่พบข้อมูลมือข้างนี้")
                    return
                st.metric("Speed", "{:.4f}".format(m["speed"]))
                st.metric("Smoothness", "{:.2f}".format(m["smoothness"]))
                st.metric("ROM", "{:.4f}".format(m["rom"]))
                st.metric("คะแนนรวม", "{:.4f}".format(m["score"]))
                st.caption("Frames: " + str(m["frames"]))

        show_metrics(col_l, "ซ้าย", lm, highlight=(suspect == "left"))
        show_metrics(col_r, "ขวา",  rm, highlight=(suspect == "right"))

        if lm and rm:
            st.divider()
            st.subheader("เปรียบเทียบคะแนนรวม")
            import pandas as pd
            chart_data = pd.DataFrame({
                "มือ": ["มือซ้าย", "มือขวา"],
                "คะแนน": [lm["score"], rm["score"]],
            }).set_index("มือ")
            st.bar_chart(chart_data)

        st.divider()
        st.subheader("สรุปรายงาน")

        ls  = "{:.4f}".format(lm["speed"])      if lm else "N/A"
        lsm = "{:.4f}".format(lm["smoothness"]) if lm else "N/A"
        lr  = "{:.4f}".format(lm["rom"])         if lm else "N/A"
        lsc = "{:.4f}".format(lm["score"])       if lm else "N/A"
        rs  = "{:.4f}".format(rm["speed"])       if rm else "N/A"
        rsm = "{:.4f}".format(rm["smoothness"])  if rm else "N/A"
        rr  = "{:.4f}".format(rm["rom"])         if rm else "N/A"
        rsc = "{:.4f}".format(rm["score"])       if rm else "N/A"

        report = (
            "**HandSense AI Report**\n\n"
            "| รายการ | มือซ้าย | มือขวา |\n"
            "|--------|---------|--------|\n"
            "| Speed | " + ls + " | " + rs + " |\n"
            "| Smoothness | " + lsm + " | " + rsm + " |\n"
            "| ROM | " + lr + " | " + rr + " |\n"
            "| **คะแนนรวม** | **" + lsc + "** | **" + rsc + "** |\n\n"
            "**สรุป:** " + msg + "\n\n"
            "*วิเคราะห์โดย HandSense AI - Digital Aiding 4 Aging Hackathon 2026*"
        )
        st.markdown(report)

    try:
        os.unlink(tmp_path)
    except Exception:
        pass


if __name__ == "__main__":
    main()
