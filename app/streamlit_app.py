"""Streamlit UI for context-aware PPE compliance and risk assessment."""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import av
import cv2
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

from src.detection.detector import ModelUnavailableError, PPEDetector
from src.pipeline import FrameResult, analyze_image
from src.risk_scoring.risk_calculator import load_weights
from src.rule_engine.rule_engine import load_scenarios
from src.utils.visualization import annotate_image

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
LOGGER = logging.getLogger(__name__)


def _severity_color(severity: str) -> str:
    if severity in {"High Risk", "Critical Risk"}:
        return "#dc2626"
    return "#f59e0b"


def _load_image_from_upload(uploaded_file) -> Image.Image:
    uploaded_file.seek(0)
    return Image.open(uploaded_file).convert("RGB")


def _sample_video_frames(uploaded_file, max_frames: int) -> list[tuple[str, Image.Image]]:
    suffix = Path(uploaded_file.name).suffix or ".mp4"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_file.write(uploaded_file.getbuffer())
        temp_path = Path(temp_file.name)

    capture = cv2.VideoCapture(str(temp_path))
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    step = max(1, frame_count // max_frames)
    frames: list[tuple[str, Image.Image]] = []
    index = 0
    while capture.isOpened() and len(frames) < max_frames:
        ok, frame = capture.read()
        if not ok:
            break
        if index % step == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append((f"{uploaded_file.name} frame {index}", Image.fromarray(rgb)))
        index += 1
    capture.release()
    temp_path.unlink(missing_ok=True)
    return frames


def _get_detector(confidence_threshold: float = 0.4):
    return PPEDetector(conf_threshold=confidence_threshold)


class LivePPEProcessor(VideoProcessorBase):
    def __init__(
        self,
        detector,
        scenario: dict,
        weights: dict[str, float],
        overlap_threshold: float,
    ) -> None:
        self.detector = detector
        self.scenario = scenario
        self.weights = weights
        self.overlap_threshold = overlap_threshold
        self.latest_result: FrameResult | None = None
        self.latest_frame_name = "Live webcam"

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image_bgr = frame.to_ndarray(format="bgr24")
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        result = analyze_image(
            pil_image,
            self.detector,
            self.scenario,
            self.weights,
            self.overlap_threshold,
        )
        self.latest_result = result

        annotated = annotate_image(
            image=pil_image,
            detections=result.detections,
            assignments=result.verification.assignments,
            required_items=set(self.scenario.get("required", [])),
            ignored_items=result.compliance.ignored,
        )
        annotated_bgr = cv2.cvtColor(np.array(annotated), cv2.COLOR_RGB2BGR)
        warning_color = _severity_color(result.risk.severity)

        overlay_height = 72
        cv2.rectangle(annotated_bgr, (0, 0), (annotated_bgr.shape[1], overlay_height), (20, 20, 20), -1)
        cv2.rectangle(
            annotated_bgr,
            (0, 0),
            (annotated_bgr.shape[1], overlay_height),
            tuple(int(warning_color.lstrip("#")[index : index + 2], 16) for index in (0, 2, 4)),
            3,
        )
        cv2.putText(
            annotated_bgr,
            f"Risk: {result.risk.severity}  ({result.risk.normalized_score:.1f}/100)",
            (18, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.72,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        missing_text = ", ".join(item.replace("_", " ").title() for item in sorted(result.compliance.required_missing))
        if not missing_text:
            missing_text = "Missing PPE: None"
        else:
            missing_text = f"Missing PPE: {missing_text}"
        cv2.putText(
            annotated_bgr,
            missing_text[:90],
            (18, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        return av.VideoFrame.from_ndarray(annotated_bgr, format="bgr24")


def _analyze_named_image(
    name: str,
    image: Image.Image,
    detector,
    scenario: dict,
    weights: dict[str, float],
    overlap_threshold: float,
) -> tuple[str, Image.Image, FrameResult]:
    result = analyze_image(image, detector, scenario, weights, overlap_threshold)
    return name, image, result


def _render_frame_result(name: str, image: Image.Image, result: FrameResult, scenario: dict) -> None:
    compliance = result.compliance
    required_items = set(scenario.get("required", []))
    missing_items = sorted(compliance.required_missing)
    warning_color = "#f59e0b" if result.risk.severity in {"Safe", "Low Risk", "Moderate Risk"} else "#dc2626"

    st.subheader(name)
    banner_color = "#18804a" if compliance.is_compliant else "#be3030"
    st.markdown(
        f"""
        <div style="background:{banner_color}; color:white; padding:0.75rem 1rem; font-weight:700;">
          {'COMPLIANT' if compliance.is_compliant else 'NON-COMPLIANT'}
        </div>
        """,
        unsafe_allow_html=True,
    )

    annotated = annotate_image(
        image=image,
        detections=result.detections,
        assignments=result.verification.assignments,
        required_items=required_items,
        ignored_items=compliance.ignored,
    )

    left, right = st.columns([1.25, 1])
    with left:
        st.image(annotated, caption="Annotated detections", use_container_width=True)
    with right:
        st.markdown(
            f"""
            <div style="padding:0.9rem 1rem; border-radius:0.75rem; background:#f4f6f8; margin-bottom:0.75rem;">
              <div style="font-size:0.9rem; font-weight:700; color:#1f2937;">Expected Risk</div>
              <div style="font-size:1.8rem; font-weight:800; color:{warning_color};">
                {result.risk.severity}
              </div>
              <div style="color:#374151;">{result.risk.normalized_score:.1f}/100</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown(
            f"<div style='padding:0.45rem 0.7rem; border-left:4px solid {warning_color}; color:{warning_color}; font-weight:700;'>"
            f"WARNING: Most important safety issue for this scene"
            f"</div>",
            unsafe_allow_html=True,
        )
        st.write("Missing PPE for this scene:")
        if missing_items:
            st.error(", ".join(item.replace("_", " ").title() for item in missing_items))
        else:
            st.success("None")
        st.write("Most important warning:")
        st.warning(result.risk.recommendation)
        if compliance.ignored:
            with st.expander("Detected but not required for this scene"):
                st.write(", ".join(sorted(item.replace("_", " ").title() for item in compliance.ignored)))


def _process_inputs(
    inputs: list[tuple[str, Image.Image]],
    detector,
    scenario: dict,
    weights: dict[str, float],
    overlap_threshold: float,
) -> list[tuple[str, Image.Image, FrameResult]]:
    results: list[tuple[str, Image.Image, FrameResult]] = []
    with st.spinner("Processing PPE compliance..."):
        for name, image in inputs:
            results.append(
                _analyze_named_image(
                    name=name,
                    image=image,
                    detector=detector,
                    scenario=scenario,
                    weights=weights,
                    overlap_threshold=overlap_threshold,
                )
            )
    return results


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="PPE Kit Detector", layout="wide")
    st.title("PPE Kit Detector")
    st.caption("Select a scene, add an image, take a webcam snapshot, or upload a video, then review the trained model output.")

    scenarios = load_scenarios()
    weights = load_weights()

    scenario_key = st.selectbox(
        "Scene",
        options=list(scenarios.keys()),
        format_func=lambda key: scenarios[key]["display_name"],
    )
    scenario = scenarios[scenario_key]

    try:
        detector = _get_detector()
    except ModelUnavailableError as exc:
        st.error(str(exc))
        return

    st.info(
        f"Required PPE for {scenario['display_name']}: "
        + ", ".join(item.replace("_", " ").title() for item in scenario.get("required", []))
    )

    input_mode = st.radio("Input type", ["Image", "Webcam", "Video"], horizontal=True)

    if input_mode == "Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"], key="image")
        if st.button("Detect PPE in image", type="primary", key="image_run"):
            if uploaded_image is None:
                st.warning("Upload an image first.")
            else:
                image = _load_image_from_upload(uploaded_image)
                results = _process_inputs(
                    inputs=[(uploaded_image.name, image)],
                    detector=detector,
                    scenario=scenario,
                    weights=weights,
                    overlap_threshold=0.18,
                )
                _render_frame_result(*results[0], scenario=scenario)

    elif input_mode == "Webcam":
        st.write("Live webcam PPE detection runs only in this tab.")
        st.caption("The video stream is processed continuously and the risk warning is drawn on the live frame.")
        webrtc_ctx = webrtc_streamer(
            key="ppe-live-webcam",
            mode=WebRtcMode.SENDRECV,
            video_processor_factory=lambda: LivePPEProcessor(
                detector=detector,
                scenario=scenario,
                weights=weights,
                overlap_threshold=0.18,
            ),
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        if webrtc_ctx.video_processor and webrtc_ctx.video_processor.latest_result:
            live_result = webrtc_ctx.video_processor.latest_result
            severity_color = _severity_color(live_result.risk.severity)
            st.markdown(
                f"""
                <div style="padding:0.8rem 1rem; border-radius:0.75rem; border-left:6px solid {severity_color}; background:#f8fafc; margin-top:1rem;">
                  <div style="font-weight:700; color:{severity_color};">Live Risk: {live_result.risk.severity}</div>
                  <div>Score: {live_result.risk.normalized_score:.1f}/100</div>
                  <div>Missing PPE: {', '.join(item.replace('_', ' ').title() for item in sorted(live_result.compliance.required_missing)) or 'None'}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    else:
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "mov", "avi"], key="video")
        if st.button("Detect PPE in video", type="primary", key="video_run"):
            if uploaded_video is None:
                st.warning("Upload a video first.")
            else:
                frames = _sample_video_frames(uploaded_video, max_frames=6)
                if not frames:
                    st.warning("No frames could be read from the video.")
                else:
                    results = _process_inputs(
                        inputs=frames,
                        detector=detector,
                        scenario=scenario,
                        weights=weights,
                        overlap_threshold=0.18,
                    )
                    if len(results) > 1:
                        worst = max(results, key=lambda item: item[2].risk.normalized_score)
                        col1, col2 = st.columns(2)
                        col1.metric("Frames checked", len(results))
                        col2.metric(
                            "Worst frame risk",
                            worst[2].risk.severity,
                            f"{worst[2].risk.normalized_score:.1f}/100",
                        )
                        selected_name = st.selectbox(
                            "Review frame",
                            [item[0] for item in results],
                            key="video_frame",
                        )
                        selected = next(item for item in results if item[0] == selected_name)
                        _render_frame_result(*selected, scenario=scenario)
                    else:
                        _render_frame_result(*results[0], scenario=scenario)


if __name__ == "__main__":
    main()
