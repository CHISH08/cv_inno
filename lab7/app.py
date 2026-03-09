from __future__ import annotations

import os
import tempfile
import time
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Any

import cv2
import gradio as gr
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModelForDepthEstimation

DEPTH_MODEL_ID = os.getenv("DEPTH_MODEL_ID", "depth-anything/Depth-Anything-V2-Small-hf")

MAX_VIDEO_FRAMES = int(os.getenv("MAX_VIDEO_FRAMES", "240"))

IMAGE_MAX_SIDE = int(os.getenv("IMAGE_MAX_SIDE", "640"))
VIDEO_MAX_SIDE = int(os.getenv("VIDEO_MAX_SIDE", "320"))
VIDEO_FPS_LIMIT = float(os.getenv("VIDEO_FPS_LIMIT", "2.5"))
STREAM_EVERY = float(os.getenv("STREAM_EVERY", "0.35"))

IMAGE_INFER_SIZE = int(os.getenv("IMAGE_INFER_SIZE", "384"))
VIDEO_INFER_SIZE = int(os.getenv("VIDEO_INFER_SIZE", "224"))
WEBCAM_INFER_SIZE = int(os.getenv("WEBCAM_INFER_SIZE", "192"))

VIDEO_BATCH_SIZE = int(os.getenv("VIDEO_BATCH_SIZE", "4"))

ENABLE_TORCH_COMPILE = os.getenv("ENABLE_TORCH_COMPILE", "0") == "1"
PERF_LOG = os.getenv("PERF_LOG", "0") == "1"
PERF_LOG_EVERY = max(1, int(os.getenv("PERF_LOG_EVERY", "30")))

WEBCAM_STREAM_LOCK = Lock()

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass


class DepthEngine:
    def __init__(self) -> None:
        self.has_cuda = torch.cuda.is_available()
        self.device = torch.device("cuda:0" if self.has_cuda else "cpu")
        self.dtype = torch.float16 if self.has_cuda else torch.float32

        print("cuda:", self.has_cuda, "device:", self.device, "dtype:", self.dtype)
        if self.has_cuda:
            print("gpu:", torch.cuda.get_device_name(0))

        self.processor = AutoImageProcessor.from_pretrained(DEPTH_MODEL_ID, use_fast=True)
        self.model = AutoModelForDepthEstimation.from_pretrained(
            DEPTH_MODEL_ID,
            torch_dtype=self.dtype,
        ).to(self.device)
        self.model.eval()

        mean = self._ensure_rgb_triplet(getattr(self.processor, "image_mean", None))
        std = self._ensure_rgb_triplet(getattr(self.processor, "image_std", None))
        self.webcam_mean = torch.tensor(mean, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)
        self.webcam_std = torch.tensor(std, device=self.device, dtype=self.dtype).view(1, 3, 1, 1)

        self.perf_log = PERF_LOG
        self.perf_log_every = PERF_LOG_EVERY
        self._perf_lock = Lock()
        self._perf_frames = 0
        self._perf_totals = {
            "preprocess": 0.0,
            "model": 0.0,
            "postprocess": 0.0,
            "total": 0.0,
        }

        if ENABLE_TORCH_COMPILE and self.has_cuda and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model, mode="reduce-overhead", fullgraph=False)
                print("torch.compile: enabled")
            except Exception as e:
                print("torch.compile disabled:", repr(e))

    @staticmethod
    def _ensure_rgb_triplet(value: Any) -> list[float]:
        if value is None:
            return [0.5, 0.5, 0.5]
        if isinstance(value, (float, int)):
            v = float(value)
            return [v, v, v]

        values = [float(x) for x in value]
        if not values:
            return [0.5, 0.5, 0.5]
        if len(values) == 1:
            return [values[0], values[0], values[0]]
        return values[:3]

    def _maybe_sync_for_perf(self) -> None:
        if self.perf_log and self.has_cuda:
            torch.cuda.synchronize()

    def _record_live_perf(
        self,
        preprocess_s: float,
        model_s: float,
        postprocess_s: float,
        total_s: float,
    ) -> None:
        if not self.perf_log:
            return

        with self._perf_lock:
            self._perf_frames += 1
            self._perf_totals["preprocess"] += preprocess_s
            self._perf_totals["model"] += model_s
            self._perf_totals["postprocess"] += postprocess_s
            self._perf_totals["total"] += total_s

            if self._perf_frames % self.perf_log_every != 0:
                return

            frames = float(self._perf_frames)
            avg_pre_ms = 1000.0 * self._perf_totals["preprocess"] / frames
            avg_model_ms = 1000.0 * self._perf_totals["model"] / frames
            avg_post_ms = 1000.0 * self._perf_totals["postprocess"] / frames
            avg_total_ms = 1000.0 * self._perf_totals["total"] / frames
            avg_total_s = self._perf_totals["total"] / frames
            fps = (1.0 / avg_total_s) if avg_total_s > 0 else 0.0

            print(
                "[perf][live] "
                f"frames={self._perf_frames} "
                f"avg_ms(pre={avg_pre_ms:.1f}, model={avg_model_ms:.1f}, "
                f"post={avg_post_ms:.1f}, total={avg_total_ms:.1f}) "
                f"fps={fps:.2f}"
            )

    def _prepare_batch(self, images_rgb: list[np.ndarray], infer_size: int) -> torch.Tensor:
        inputs = self.processor(
            images=images_rgb,
            return_tensors="pt",
            do_resize=True,
            size={"height": infer_size, "width": infer_size},
            keep_aspect_ratio=False,
        )
        pixel_values = inputs["pixel_values"].to(
            self.device,
            dtype=self.dtype,
            non_blocking=True,
        )
        return pixel_values

    def _prepare_webcam_fast(self, image_rgb: np.ndarray, infer_size: int) -> torch.Tensor:
        resized = cv2.resize(image_rgb, (infer_size, infer_size), interpolation=cv2.INTER_AREA)
        resized = np.ascontiguousarray(resized)

        pixel_values = torch.from_numpy(resized).permute(2, 0, 1).unsqueeze(0)
        pixel_values = pixel_values.to(self.device, dtype=self.dtype, non_blocking=True)
        pixel_values = pixel_values / 255.0
        pixel_values = (pixel_values - self.webcam_mean) / self.webcam_std
        return pixel_values

    def infer_batch_same_size(
        self,
        images_rgb: list[np.ndarray],
        infer_size: int,
        output_h: int,
        output_w: int,
    ) -> list[np.ndarray]:
        if not images_rgb:
            return []

        pixel_values = self._prepare_batch(images_rgb, infer_size)

        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            pred = outputs.predicted_depth
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=(output_h, output_w),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)

        depth_batch = pred.float().cpu().numpy()
        return [depth_batch[i] for i in range(depth_batch.shape[0])]

    def infer_webcam_fast(self, image_rgb: np.ndarray, infer_size: int) -> np.ndarray:
        t0 = time.perf_counter()
        pixel_values = self._prepare_webcam_fast(image_rgb, infer_size)
        self._maybe_sync_for_perf()
        t1 = time.perf_counter()

        with torch.inference_mode():
            outputs = self.model(pixel_values=pixel_values)
            pred = outputs.predicted_depth
            pred = F.interpolate(
                pred.unsqueeze(1),
                size=(infer_size, infer_size),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)
        self._maybe_sync_for_perf()
        t2 = time.perf_counter()

        depth = pred[0].float().cpu().numpy()
        depth_colored = colorize_depth(depth)
        t3 = time.perf_counter()

        self._record_live_perf(
            preprocess_s=t1 - t0,
            model_s=t2 - t1,
            postprocess_s=t3 - t2,
            total_s=t3 - t0,
        )
        return depth_colored

    def infer_one(
        self,
        image_rgb: np.ndarray,
        infer_size: int,
        output_h: int,
        output_w: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        depth = self.infer_batch_same_size([image_rgb], infer_size, output_h, output_w)[0]
        depth_colored = colorize_depth(depth)
        return depth, depth_colored


@lru_cache(maxsize=1)
def get_engine() -> DepthEngine:
    return DepthEngine()


def warmup_models() -> None:
    engine = get_engine()

    dummy_webcam = np.zeros((WEBCAM_INFER_SIZE, WEBCAM_INFER_SIZE, 3), dtype=np.uint8)
    dummy_image = np.zeros((IMAGE_INFER_SIZE, IMAGE_INFER_SIZE, 3), dtype=np.uint8)
    dummy_video_batch = [
        np.zeros((VIDEO_INFER_SIZE, VIDEO_INFER_SIZE, 3), dtype=np.uint8)
        for _ in range(max(1, VIDEO_BATCH_SIZE))
    ]

    _ = engine.infer_webcam_fast(dummy_webcam, WEBCAM_INFER_SIZE)
    _ = engine.infer_one(
        dummy_image,
        IMAGE_INFER_SIZE,
        IMAGE_INFER_SIZE,
        IMAGE_INFER_SIZE,
    )
    _ = engine.infer_batch_same_size(
        dummy_video_batch,
        VIDEO_INFER_SIZE,
        VIDEO_INFER_SIZE,
        VIDEO_INFER_SIZE,
    )

    if engine.has_cuda:
        torch.cuda.synchronize()


def extract_video_path(video_input: Any) -> str | None:
    if video_input is None:
        return None
    if isinstance(video_input, str):
        return video_input
    if isinstance(video_input, dict):
        for key in ("path", "name", "video"):
            value = video_input.get(key)
            if isinstance(value, str):
                return value
    return None


def extract_image_path(image_input: Any) -> str | None:
    if image_input is None:
        return None
    if isinstance(image_input, str):
        return image_input
    if isinstance(image_input, dict):
        for key in ("path", "name", "image"):
            value = image_input.get(key)
            if isinstance(value, str):
                return value
    return None


def load_image_rgb(image_input: Any) -> np.ndarray:
    if image_input is None:
        raise gr.Error("Image is empty.")

    if isinstance(image_input, np.ndarray):
        arr = np.asarray(image_input, dtype=np.uint8)
        if arr.ndim == 2:
            arr = cv2.cvtColor(arr, cv2.COLOR_GRAY2RGB)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = cv2.cvtColor(arr, cv2.COLOR_RGBA2RGB)
        return arr

    image_path = extract_image_path(image_input)
    if image_path and Path(image_path).exists():
        image_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
        if image_bgr is None:
            raise gr.Error(f"Failed to load image: {image_path}")
        return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)

    raise gr.Error("Unsupported image input format.")


def resize_keep_aspect(image_rgb: np.ndarray, max_side: int) -> np.ndarray:
    h, w = image_rgb.shape[:2]
    current_max = max(h, w)
    if current_max <= max_side:
        return image_rgb

    scale = max_side / float(current_max)
    new_w = max(1, int(round(w * scale)))
    new_h = max(1, int(round(h * scale)))
    return cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_AREA)


def normalize_depth(depth: np.ndarray) -> np.ndarray:
    d_min = float(np.min(depth))
    d_max = float(np.max(depth))
    if d_max - d_min < 1e-8:
        return np.zeros_like(depth, dtype=np.float32)
    return ((depth - d_min) / (d_max - d_min)).astype(np.float32)


def colorize_depth(depth: np.ndarray) -> np.ndarray:
    depth_norm = normalize_depth(depth)
    depth_u8 = np.clip(depth_norm * 255.0, 0, 255).astype(np.uint8)
    depth_colored_bgr = cv2.applyColorMap(depth_u8, cv2.COLORMAP_INFERNO)
    return cv2.cvtColor(depth_colored_bgr, cv2.COLOR_BGR2RGB)


def estimate_depth_map(image_rgb: np.ndarray, infer_size: int) -> tuple[np.ndarray, np.ndarray]:
    h, w = image_rgb.shape[:2]
    engine = get_engine()
    return engine.infer_one(image_rgb, infer_size=infer_size, output_h=h, output_w=w)


def analyze_image(image_input: Any) -> tuple[np.ndarray, str]:
    image_rgb = load_image_rgb(image_input)
    image_rgb = resize_keep_aspect(image_rgb, IMAGE_MAX_SIDE)
    _, depth_colored = estimate_depth_map(image_rgb, IMAGE_INFER_SIZE)
    return depth_colored, f"Relative depth map is ready. Output size: {image_rgb.shape[1]}x{image_rgb.shape[0]}."


def make_video_writer(output_path: str, fps: float, width: int, height: int) -> cv2.VideoWriter:
    return cv2.VideoWriter(
        output_path,
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )


def flush_video_batch(
    writer: cv2.VideoWriter,
    engine: DepthEngine,
    frame_batch_rgb: list[np.ndarray],
    infer_size: int,
) -> int:
    if not frame_batch_rgb:
        return 0

    h, w = frame_batch_rgb[0].shape[:2]
    depth_batch = engine.infer_batch_same_size(
        frame_batch_rgb,
        infer_size=infer_size,
        output_h=h,
        output_w=w,
    )

    for depth in depth_batch:
        depth_colored = colorize_depth(depth)
        writer.write(cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR))

    return len(depth_batch)


def process_video(video_input: Any, progress=gr.Progress()) -> tuple[str | None, str]:
    video_path = extract_video_path(video_input)
    if not video_path:
        raise gr.Error("Upload a valid video first.")
    if not Path(video_path).exists():
        raise gr.Error(f"Video file not found: {video_path}")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise gr.Error("Failed to open video.")

    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if src_fps <= 0:
        src_fps = 25.0

    fps = min(src_fps, VIDEO_FPS_LIMIT) if VIDEO_FPS_LIMIT > 0 else src_fps

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        total_frames = MAX_VIDEO_FRAMES

    scale = 1.0
    if max(frame_width, frame_height) > VIDEO_MAX_SIDE:
        scale = VIDEO_MAX_SIDE / float(max(frame_width, frame_height))

    out_w = max(1, int(round(frame_width * scale)))
    out_h = max(1, int(round(frame_height * scale)))

    out_dir = Path(tempfile.mkdtemp(prefix="depth_video_"))
    depth_path = str(out_dir / f"depth_{int(time.time())}.mp4")

    depth_writer = make_video_writer(depth_path, fps, out_w, out_h)
    if not depth_writer.isOpened():
        cap.release()
        raise gr.Error("Failed to create depth output video.")

    engine = get_engine()

    processed = 0
    written = 0
    truncated = False
    sampled_every = max(1, int(round(src_fps / fps))) if fps > 0 else 1
    frame_batch_rgb: list[np.ndarray] = []

    try:
        frame_idx = 0
        while True:
            ok, frame_bgr = cap.read()
            if not ok:
                break

            if frame_idx % sampled_every != 0:
                frame_idx += 1
                continue
            frame_idx += 1

            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if scale != 1.0:
                frame_rgb = cv2.resize(frame_rgb, (out_w, out_h), interpolation=cv2.INTER_AREA)

            frame_batch_rgb.append(frame_rgb)
            processed += 1

            if len(frame_batch_rgb) >= max(1, VIDEO_BATCH_SIZE):
                written += flush_video_batch(
                    writer=depth_writer,
                    engine=engine,
                    frame_batch_rgb=frame_batch_rgb,
                    infer_size=VIDEO_INFER_SIZE,
                )
                frame_batch_rgb.clear()

                progress(
                    min(processed / max(total_frames, 1), 1.0),
                    desc=f"Processed frames: {processed}",
                )

            if processed >= MAX_VIDEO_FRAMES:
                truncated = True
                break

        if frame_batch_rgb:
            written += flush_video_batch(
                writer=depth_writer,
                engine=engine,
                frame_batch_rgb=frame_batch_rgb,
                infer_size=VIDEO_INFER_SIZE,
            )
            frame_batch_rgb.clear()

    finally:
        cap.release()
        depth_writer.release()

    if written == 0:
        raise gr.Error("No readable frames in the input video.")

    status = f"Done. Processed {processed} frame(s)."
    if truncated:
        status += f" Stopped at MAX_VIDEO_FRAMES={MAX_VIDEO_FRAMES}."
    if scale != 1.0:
        status += f" Output resized to {out_w}x{out_h}."
    if fps < src_fps:
        status += f" FPS limited to {fps:.1f}."
    if VIDEO_BATCH_SIZE > 1:
        status += f" Batch size: {VIDEO_BATCH_SIZE}."

    return depth_path, status


def stream_webcam(frame: np.ndarray) -> tuple[Any, Any]:
    if frame is None:
        return gr.skip(), gr.skip()

    if not WEBCAM_STREAM_LOCK.acquire(blocking=False):
        return gr.skip(), gr.skip()

    try:
        image_rgb = np.asarray(frame, dtype=np.uint8)
        if image_rgb.ndim == 2:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_GRAY2RGB)
        elif image_rgb.ndim == 3 and image_rgb.shape[2] == 4:
            image_rgb = cv2.cvtColor(image_rgb, cv2.COLOR_RGBA2RGB)

        engine = get_engine()
        depth_colored = engine.infer_webcam_fast(image_rgb, WEBCAM_INFER_SIZE)
        return depth_colored, "Live relative depth is running."
    finally:
        WEBCAM_STREAM_LOCK.release()


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Relative Depth App") as demo:
        gr.Markdown(
            """
            # Relative Depth App
            - Image → relative depth map
            - Video → relative depth video
            - Webcam → live relative depth

            Optimized for maximum speed on the current Depth Anything V2 Small model.
            """
        )

        with gr.Tabs():
            with gr.TabItem("Image"):
                with gr.Row():
                    image_input = gr.Image(
                        label="Input image",
                        sources=["upload", "webcam"],
                        type="filepath",
                    )
                    depth_output = gr.Image(label="Relative depth map", type="numpy")
                image_status = gr.Markdown("Upload an image to begin.")
                analyze_btn = gr.Button("Analyze image", variant="primary")

            with gr.TabItem("Video"):
                with gr.Row():
                    video_input = gr.Video(label="Input video", sources=["upload"])
                    video_depth_output = gr.Video(label="Relative depth video")
                video_btn = gr.Button("Process video", variant="primary")
                video_status = gr.Markdown("Upload a video and click `Process video`.")

            with gr.TabItem("Webcam"):
                gr.Markdown("Allow camera access. This tab computes live relative depth only.")
                with gr.Row():
                    webcam_input = gr.Image(
                        label="Webcam input",
                        sources=["webcam"],
                        type="numpy",
                        streaming=True,
                    )
                    webcam_depth = gr.Image(label="Live relative depth", type="numpy")
                webcam_status = gr.Markdown("Waiting for webcam frames...")

        analyze_btn.click(
            fn=analyze_image,
            inputs=image_input,
            outputs=[depth_output, image_status],
        )

        video_btn.click(
            fn=process_video,
            inputs=video_input,
            outputs=[video_depth_output, video_status],
        )

        webcam_input.stream(
            fn=stream_webcam,
            inputs=webcam_input,
            outputs=[webcam_depth, webcam_status],
            stream_every=STREAM_EVERY,
            show_progress="hidden",
            queue=False,
            trigger_mode="always_last",
            concurrency_limit=1,
        )

    return demo


app = build_app()

if __name__ == "__main__":
    warmup_models()
    app.queue(max_size=1).launch(
        server_name="0.0.0.0",
        server_port=int(os.getenv("PORT", "7860")),
    )
