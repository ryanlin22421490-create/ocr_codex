import base64
import argparse
import re
import signal
import sys
import threading
import time
from queue import Full, Queue
from pathlib import Path

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

SCRIPT_DIR = Path(__file__).resolve().parent
IMX519_DIR = SCRIPT_DIR / "IMX519"
if IMX519_DIR.exists():
    sys.path.insert(0, str(IMX519_DIR))

from JetsonCamera import Camera
from Focuser import Focuser
import daynight
from ContinuousAutofocus_turbo import (FocusController, RoiMeter,
                                       add_af_arguments, check_af_args)

# =========================
# Ollama 設定
# =========================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen3-vl:2b-instruct"
OLLAMA_NUM_CTX = 2048

# =========================
# GPU / YOLO
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = None
half = device == "cuda"

FPS_OPTIONS = (9, 15, 17, 60, 120)
SENSOR_MODES = {
    9: (4656, 3496),
    15: (3840, 2160),
    17: (3840, 2160),
    60: (1920, 1080),
    120: (1280, 720),
}
EXPOSURE_MIN_MS = 0.1
EXPOSURE_MAX_RATIO = 0.95
EXPOSURE_SENSOR_MAX_MS = 683.709
EXPOSURE_STEP_MS = 1.0
EXPOSURE_COARSE_STEP_MS = 10.0
FOCUS_STEP = 50
AF_REOPEN_WARMUP = 1.0   # seconds for auto exposure to settle after the camera reopens

KEY_LEFT = (2424832, 65361, 81)
KEY_UP = (2490368, 65362, 82)
KEY_RIGHT = (2555904, 65363, 83)
KEY_DOWN = (2621440, 65364, 84)
# Keys that reopen the camera (or change exposure): they would skew the startup
# day / night measurement, so pressing one during it cancels the detection.
REOPEN_KEY_CODES = {ord(c) for c in "-_=+{}[]aA"}

# =========================
# 執行參數
# =========================
GRACE_FRAMES = 15
# ByteTrack drops a lost ID after its 30-frame track buffer and never reuses it,
# so per-track state unseen this long can be forgotten (~18 s at 17 FPS).
FORGET_FRAMES = 300
CLEANUP_INTERVAL_FRAMES = 30
OCR_FRAME_INTERVAL = 5
OCR_INTERVAL_SECONDS = 1.0
OCR_QUEUE_SIZE = 2
# 看板讀不清楚時，模型會把提示詞的「1 至 4 位數字」照抄成 1234；視同 NONE。
# 單次確認沒有第二次結果可以糾正，不過濾的話 1234 會直接被確認且不再重讀。
INVALID_RESULTS = {"1234"}

# 小於 32 px 的圖片會讓 qwen3-vl runner 回 500；ROI 不放大，這種 ROI 直接不送 OCR。
# ROI 從高解析度來源影格裁切，可能超過 1000 px 寬，一律縮到 MAX_OCR_SIDE，
# 否則影像 token 會超過 OLLAMA_NUM_CTX（4K 來源的 ROI 約 2000 個 token）。
OLLAMA_MIN_IMAGE_SIDE = 32
MAX_OCR_SIDE = 512
MIN_SOURCE_ROI_WIDTH = OLLAMA_MIN_IMAGE_SIDE
MIN_SOURCE_ROI_HEIGHT = OLLAMA_MIN_IMAGE_SIDE

# 公車框長寬比（處理畫面座標，寬 / 高）。排隊進站時被前車擋住的後車，框只剩露出的細條，
# 看板只拍到一半就被辨識（10/22 實測：132 只露出「13」，被確認成 1221）。
# 單次確認沒有第二次結果可以糾正，太窄的框先不送 OCR，等車開到前面、框變寬再辨識。
# 10/22 錄影：單獨一台的框中位數 1.10，被前車擋住的後車中位數 0.78；
# 下限 0.8 擋掉約 56% 的後車框，單獨車輛只擋掉約 4%。上限 1.8 過濾側面角度。
# 10/28 錄影（鏡頭和公車道幾乎同一直線）：108 單獨正面駛來、看板完整時只有 0.54～0.79，
# 被 0.8 誤擋約 1.9 秒；被 12 號擋住時 0.42～0.51（0.51 那幾幀看板已完整，放行無妨），下限改為 0.5。
# 長寬比會隨架設角度改變，換位置要重新量。
ASPECT_RATIO_MIN = 0.5
ASPECT_RATIO_MAX = 1.8

# =========================
# MP4 錄影（--record）
# =========================
RECORD_DEFAULT_MAX_FPS = 30
RECORD_QUEUE_SIZE = 16      # annotated frames waiting for the encoder (~2.7 MB each at 720p)
RECORD_MAX_STALL = 2.0      # seconds a frozen camera (e.g. reopening) is shown in the video

# =========================
# 共用狀態
# =========================
confirmed_buses = {}
last_seen_frame = {}
last_ocr_time = {}
ocr_pending_ids = set()
ocr_queue = Queue(maxsize=OCR_QUEUE_SIZE)
lock = threading.Lock()
ollama_session = requests.Session()
ollama_session.trust_env = False


def encode_frame(frame):
    """將 OpenCV BGR 影像編碼成 Ollama 可接受的 JPEG Base64。"""
    ok, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 95])
    if not ok:
        raise ValueError("無法將 ROI 編碼為 JPEG")
    return base64.b64encode(buffer).decode("utf-8")


def warmup_ollama():
    """預載入模型並保持在 VRAM，減少第一個 OCR 任務的等待時間。"""
    print("Ollama warming up...")
    # 暖機只走小尺寸全黑圖片，不使用 OCR 放大或影像增強。
    dummy = np.zeros((OLLAMA_MIN_IMAGE_SIDE, OLLAMA_MIN_IMAGE_SIDE, 3), dtype=np.uint8)
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": "warmup"},
            {"role": "user", "content": "OK", "images": [encode_frame(dummy)]},
        ],
        "stream": False,
        "keep_alive": -1,
        "options": {"num_ctx": OLLAMA_NUM_CTX, "num_predict": 1, "temperature": 0},
    }

    try:
        response = ollama_session.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        print("Ollama ready (kept alive in VRAM)")
    except requests.RequestException as error:
        print("warmup failed:", error)


# =========================
# ROI 影像增強（僅處理亮度通道）
# =========================
def build_gamma_lut(gamma: float) -> np.ndarray:
    """建立 gamma LUT。gamma > 1 變暗，gamma < 1 變亮。"""
    if gamma <= 0:
        raise ValueError("gamma 必須大於 0")
    table = np.array(
        [((value / 255.0) ** gamma) * 255 for value in range(256)],
        dtype=np.uint8,
    )
    return table


def apply_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    """依 gamma 套用亮度校正；方向與 build_gamma_lut 的說明一致。"""
    return cv2.LUT(image, build_gamma_lut(gamma))


def overexposed_ratio(image: np.ndarray, threshold: int = 250) -> float:
    """計算近乎死白的亮度像素比例。"""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    return float(np.count_nonzero(luma >= threshold) / luma.size)


def auto_gamma(image: np.ndarray, target_mean: float = 128.0) -> float:
    """估算把平均亮度調向 target_mean 的 gamma（>1 變暗，<1 變亮）。"""
    luma = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    mean_brightness = float(np.clip(luma.mean(), 1.0, 254.0))
    target = float(np.clip(target_mean, 1.0, 254.0))
    gamma = np.log(target / 255.0) / np.log(mean_brightness / 255.0)
    return float(np.clip(gamma, 0.55, 2.5))


def unsharp_mask(image: np.ndarray, sigma: float = 0.8,
                 strength: float = 0.65) -> np.ndarray:
    """低強度銳化；內部高斯模糊只用來計算銳化差值。"""
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    return cv2.addWeighted(image, 1 + strength, blurred, -strength, 0)


def enhance_roi(roi_img: np.ndarray) -> np.ndarray:
    """針對 LED 看板做條件式增強，不改變 BGR 色彩通道。

    流程：依 LAB-L 曝光狀態選擇 gamma 校正或 CLAHE -> 輕度銳化。
    不使用雙邊濾波或高光壓縮。正常曝光且對比足夠的影像不會被強制
    套用 gamma，避免損失數字與背景之間的細節。
    """
    if roi_img is None or roi_img.size == 0:
        raise ValueError("ROI 不可為空")

    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)

    mean_luma = float(luma.mean())
    contrast = float(luma.std())
    white_ratio = overexposed_ratio(luma)
    clahe_used = False
    gamma_used = None

    if white_ratio >= 0.08 or mean_luma >= 165:
        strategy = "overexposed"
        gamma_used = auto_gamma(luma, target_mean=120.0)
        luma = apply_gamma(luma, gamma_used)
    elif mean_luma <= 115:
        strategy = "dark"
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        luma = clahe.apply(luma)
        gamma_used = auto_gamma(luma, target_mean=142.0)
        luma = apply_gamma(luma, gamma_used)
        clahe_used = True
    elif contrast <= 35:
        strategy = "low-contrast"
        clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
        luma = clahe.apply(luma)
        clahe_used = True
    else:
        strategy = "normal"

    luma = unsharp_mask(luma)
    enhanced = cv2.cvtColor(cv2.merge((luma, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    gamma_text = f"{gamma_used:.2f}" if gamma_used is not None else "None"
    print(
        f"[enhance_roi] {strategy}, mean={mean_luma:.1f}, "
        f"contrast={contrast:.1f}, white={white_ratio:.1%}, "
        f"gamma={gamma_text}, CLAHE={clahe_used}"
    )
    return enhanced


def shrink_for_ocr(roi_img: np.ndarray) -> np.ndarray:
    """只縮不放：最長邊超過 MAX_OCR_SIDE 才等比例縮小（控制影像 token），其餘原尺寸送出。"""
    height, width = roi_img.shape[:2]
    if max(width, height) <= MAX_OCR_SIDE:
        return roi_img

    scale = MAX_OCR_SIDE / max(width, height)
    # 看板 ROI 長寬比約 2～6，縮到 512 後短邊仍遠大於 32；max() 只是防極端比例觸發 500
    output_width = max(OLLAMA_MIN_IMAGE_SIDE, round(width * scale))
    output_height = max(OLLAMA_MIN_IMAGE_SIDE, round(height * scale))
    print(
        f"[shrink_for_ocr] {width}x{height} -> "
        f"{output_width}x{output_height}, scale={scale:.2f}"
    )
    return cv2.resize(
        roi_img,
        (output_width, output_height),
        interpolation=cv2.INTER_AREA,
    )


# =========================
# 看板 ROI 擷取與驗證
# =========================
def extract_sign_roi(frame: np.ndarray, x1: int, y1: int,
                     x2: int, y2: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """使用固定比例擷取車體上方 LED 看板區域。"""
    frame_height, frame_width = frame.shape[:2]
    x1, x2 = max(0, x1), min(frame_width, x2)
    y1, y2 = max(0, y1), min(frame_height, y2)
    width, height = x2 - x1, y2 - y1
    if width <= 0 or height <= 0:
        return frame[0:0, 0:0], (x1, y1, x1, y1)

    roi_box = (
        x1 + int(width * 0.06),
        y1,
        x2 - int(width * 0.06),
        y1 + int(height * 0.3),
    )
    rx1, ry1, rx2, ry2 = roi_box
    return frame[ry1:ry2, rx1:rx2], roi_box


def crop_source_roi(source_frame: np.ndarray,
                    roi_box: tuple[int, int, int, int],
                    frame_size: tuple[int, int]) -> np.ndarray:
    """把處理畫面（frame_size = 寬, 高）座標的 ROI 換算到高解析度來源影格再裁切。"""
    source_height, source_width = source_frame.shape[:2]
    scale_x = source_width / frame_size[0]
    scale_y = source_height / frame_size[1]
    rx1, ry1, rx2, ry2 = roi_box
    sx1, sx2 = (min(max(round(x * scale_x), 0), source_width) for x in (rx1, rx2))
    sy1, sy2 = (min(max(round(y * scale_y), 0), source_height) for y in (ry1, ry2))
    return source_frame[sy1:sy2, sx1:sx2]


def is_valid_roi(roi_img: np.ndarray) -> bool:
    """任一邊小於 32 px 就不送 OCR：ROI 不放大，送出會讓 qwen3-vl runner 回 500。"""
    if roi_img is None or roi_img.size == 0:
        return False
    height, width = roi_img.shape[:2]
    return width >= MIN_SOURCE_ROI_WIDTH and height >= MIN_SOURCE_ROI_HEIGHT


def normalize_ocr_result(result: str) -> str | None:
    """只接受 1 至 4 位數字，避免模型說明文字進入投票確認。"""
    compact = re.sub(r"\s+", "", result)
    if compact.upper() == "NONE" or compact in INVALID_RESULTS:
        return None
    return compact if re.fullmatch(r"\d{1,4}", compact) else None


# =========================
# OCR worker（受限佇列，避免多台車同時建立大量執行緒）
# =========================
def process_ocr(track_id: int, roi_img: np.ndarray):
    roi_height, roi_width = roi_img.shape[:2]
    prepared_roi = enhance_roi(shrink_for_ocr(roi_img))
    sent_height, sent_width = prepared_roi.shape[:2]
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {
                "role": "system",
                "content": "你是公車 LED 號碼辨識系統。只能回傳 1 至 4 位數字；無法辨識時只回傳 NONE。",
            },
            {
                "role": "user",
                "content": "辨識圖片中的公車路線號碼。",
                "images": [encode_frame(prepared_roi)],
            },
        ],
        "stream": False,
        "keep_alive": -1,
        # 模型預設 temperature=1 會隨機取樣（暖機設的 0 不會延續到這裡）；單次確認沒有第二次結果可以糾正，固定為 0 避免隨機讀出錯誤號碼。
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0},
    }
    started = time.monotonic()
    response = ollama_session.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    raw_result = response.json()["message"]["content"].strip()
    result = normalize_ocr_result(raw_result)
    print(f"[{time.strftime('%H:%M:%S')}] OCR [ID:{track_id}] = {raw_result!r} "
          f"| source ROI {roi_width}x{roi_height} -> sent {sent_width}x{sent_height} "
          f"| {time.monotonic() - started:.2f} s")

    if result is None:
        return

    # 一次有效結果就確認；確認後主迴圈不再為這個 ID 排 OCR。
    with lock:
        confirmed_buses[track_id] = result
    print(f"[{time.strftime('%H:%M:%S')}] CONFIRMED [ID:{track_id}] {result}")


def ocr_worker():
    while True:
        track_id, roi_img = ocr_queue.get()
        try:
            process_ocr(track_id, roi_img)
        except (ValueError, cv2.error, requests.RequestException) as error:
            print(f"OCR error [ID:{track_id}]:", error)
        except Exception as error:
            print(f"Unexpected OCR error [ID:{track_id}]:", error)
        finally:
            with lock:
                ocr_pending_ids.discard(track_id)
            ocr_queue.task_done()


# =========================
# YOLO detection
# =========================
def detect_bus(frame, conf: float):
    results = model.track(
        frame,
        imgsz=(768, 1280),
        device=device,
        classes=[5],
        conf=conf,
        half=half,
        persist=True,
        verbose=False,
        tracker="bytetrack.yaml",
    )

    buses = []
    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            if box.id is None:
                continue
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            track_id = int(box.id[0])
            buses.append((x1, y1, x2, y2, confidence, track_id))
    return buses


def schedule_ocr(track_id: int, roi_img: np.ndarray, now: float) -> bool:
    """在冷卻時間、重複任務與佇列容量都允許時排入 OCR。"""
    with lock:
        if track_id in ocr_pending_ids:
            return False
        if now - last_ocr_time.get(track_id, 0.0) < OCR_INTERVAL_SECONDS:
            return False
        try:
            ocr_queue.put_nowait((track_id, roi_img.copy()))
        except Full:
            return False
        ocr_pending_ids.add(track_id)
        last_ocr_time[track_id] = now
        return True


def get_visible_results(frame_id: int) -> list[str]:
    """取得目前出現或仍在寬限期內、已確認的公車號碼。"""
    with lock:
        visible_ids = [
            track_id
            for track_id, last_frame in last_seen_frame.items()
            if frame_id - last_frame <= GRACE_FRAMES and track_id in confirmed_buses
        ]
        return [confirmed_buses[track_id] for track_id in sorted(visible_ids)]


def forget_stale_tracks(frame_id: int) -> None:
    """清除超過 FORGET_FRAMES 沒出現的 ID，讓共用字典不會隨執行時間無限增長。

    last_seen_frame 記錄每一個出現過的 ID，其他字典的 key 都是它的子集，掃它即可。
    OCR 處理中的 ID 先保留：worker 只寫入處理中的 ID，結束後下一輪再一起刪除。
    """
    with lock:
        stale = [
            track_id
            for track_id, last_frame in last_seen_frame.items()
            if frame_id - last_frame > FORGET_FRAMES and track_id not in ocr_pending_ids
        ]
        for track_id in stale:
            del last_seen_frame[track_id]
            confirmed_buses.pop(track_id, None)
            last_ocr_time.pop(track_id, None)
    if stale:
        print(f"Forgot {len(stale)} stale track(s): {stale}")


class ContinuousAF:
    """Turbo continuous AF (ContinuousAutofocus_turbo) driven by the main-loop frames.

    Manual focus keys switch it off so it does not fight the user; F / Enter
    switches it back on and refocuses immediately.
    """

    def __init__(self, focuser, args):
        self.focuser = focuser
        self.args = args
        self.meter = RoiMeter(args)
        self.enabled = not args.no_af
        self.roi = None
        self.restart(warmup=2.0)

    def restart(self, warmup):
        """Fresh search state at the current lens position (camera reopened, AF re-enabled)."""
        self.controller = FocusController(
            lambda p: self.focuser.set(Focuser.OPT_FOCUS, p),
            self.focuser.get(Focuser.OPT_FOCUS), self.args, warmup=warmup)
        self.meter.reset()

    def enable(self):
        self.enabled = True
        self.restart(warmup=0.0)

    def disable(self):
        self.enabled = False

    @property
    def scanning(self):
        """Lens is moving through search positions: frames are deliberately defocused."""
        return self.enabled and self.controller.active

    def update(self, frame, now):
        """Must see the clean frame, before any box or text is drawn on it."""
        if not self.enabled:
            return
        m = self.meter.measure(frame, now)
        self.roi = m.roi
        self.controller.update(now, m.sharpness, m.motion, m.valid,
                               1.0 if self.args.no_exposure_comp else m.brightness)


def parse_args():
    parser = argparse.ArgumentParser(
        description="V12 YOLO + Ollama OCR using Jetson IMX519 camera "
                    "(confirmed after one OCR result)."
    )
    parser.add_argument("--width", type=int, default=1280,
                        help="Preview width. Default: 1280")
    parser.add_argument("--height", type=int, default=720,
                        help="Preview height. Default: 720")
    parser.add_argument("--source-width", type=int, default=3840,
                        help="Camera output width; the OCR ROI is cropped from this frame. "
                             "Default: 3840")
    parser.add_argument("--source-height", type=int, default=2160,
                        help="Camera output height. Default: 2160")
    parser.add_argument("--framerate", type=int, choices=FPS_OPTIONS, default=17,
                        help="IMX519 FPS. Supported: 9, 15, 17, 60, 120")
    # 鏡頭倒裝，實際執行一律用 2（旋轉 180 度）
    parser.add_argument("--flip", type=int, choices=range(8), default=2, metavar="0-7",
                        help="Rotate / mirror the camera image in hardware (nvvidconv flip-method): "
                             "0 none, 2 rotate 180 (camera mounted upside down), "
                             "4 mirror left-right, 6 mirror top-bottom. Default: 2")
    parser.add_argument("--exposure", type=float, default=None,
                        help="Fixed exposure time in ms; default is auto exposure "
                             f"({daynight.REFERENCE_EXPOSURE_MS:g} ms with day / night detection)")
    parser.add_argument("--gain", default="1 1",
                        help='Fixed gain range, e.g. "1 1"; use "" for auto gain. '
                             "Only used with --no-auto-daynight")
    daynight.add_arguments(parser)
    parser.add_argument("-i", "--i2c-bus", type=int, default=10,
                        help="Focuser I2C bus. Default: 10")
    parser.add_argument("--model", default=None,
                        help="YOLO model path; default searches this folder and its parent")
    parser.add_argument("--device", default=None,
                        help="Inference device, e.g. 0, cpu; default auto")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO bus confidence threshold, 0 to 1. Default: 0.5. "
                             "Only used with --no-auto-daynight")
    parser.add_argument("--no-af", action="store_true",
                        help="Start in manual focus; F / Enter enables continuous AF")
    parser.add_argument("--record", nargs="?", const="", default=None, metavar="FILE",
                        help="Save the window's video as MP4; without FILE: "
                             "record_<date>_<time>.mp4 in the current directory")
    parser.add_argument("--record-fps", type=float, default=None,
                        help="MP4 frame rate; default: camera FPS, at most "
                             f"{RECORD_DEFAULT_MAX_FPS}")
    add_af_arguments(parser)
    # 對焦 ROI 預設為畫面 20%，從中央往左 0.1、往上 0.1（中心在 (0.4, 0.4)，1280x720 為 x 384～640、y 216～360）；
    # 共用模組 ContinuousAutofocus_turbo 的預設（中央 30%）不變
    parser.set_defaults(roi_size=0.2, roi_x=0.0, roi_y=0.0)
    args = parser.parse_args()
    check_af_args(parser, args)
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")
    daynight.check_args(args)
    if args.record_fps is not None and args.record_fps <= 0:
        parser.error("--record-fps must be > 0")
    if args.source_width < args.width or args.source_height < args.height:
        parser.error("--source-width/--source-height must be at least --width/--height")
    # 長寬比不同時 ROI 換算仍正確，但預覽畫面會被拉伸，YOLO 看到的車子會變形
    if abs(args.source_width / args.source_height - args.width / args.height) > 0.01:
        print("Warning: source and preview aspect ratios differ; "
              "the preview frame will be stretched")
    return args


def max_exposure_ms(framerate: int) -> float:
    if framerate <= 0:
        raise ValueError("framerate must be greater than zero")
    return min(
        (1000.0 / framerate) * EXPOSURE_MAX_RATIO,
        EXPOSURE_SENSOR_MAX_MS,
    )


def clamp_exposure_ms(exposure_ms: float | None, framerate: int) -> float | None:
    if exposure_ms is None:
        return None
    return min(max(float(exposure_ms), EXPOSURE_MIN_MS), max_exposure_ms(framerate))


def resolve_model_path(requested: str | None) -> str:
    if requested:
        candidate = Path(requested).expanduser()
        if candidate.exists() or candidate.is_absolute():
            return str(candidate)

    for candidate in (SCRIPT_DIR / "yolov8n.pt", SCRIPT_DIR.parent / "yolov8n.pt"):
        if candidate.exists():
            return str(candidate)
    return requested or "yolov8n.pt"


def reopen_camera_with_exposure(camera, cam_state, focuser=None):
    exposure_ms = cam_state["exposure_ms"]
    exposure_ns = None if exposure_ms is None else int(exposure_ms * 1_000_000)
    camera.reconfigure(
        framerate=cam_state["framerate"],
        exposure_time_ns=exposure_ns,
        gain_range=cam_state["gain_range"],
        ae_lock=exposure_ms is not None,
        capture_width=cam_state["capture_width"],
        capture_height=cam_state["capture_height"],
    )
    if focuser is not None:
        focuser.reinit()


def change_framerate(camera, cam_state, focuser, direction):
    """Return True when the camera was reopened at a new frame rate."""
    current_index = FPS_OPTIONS.index(cam_state["framerate"])
    new_index = max(0, min(len(FPS_OPTIONS) - 1, current_index + direction))
    new_fps = FPS_OPTIONS[new_index]
    if new_fps == cam_state["framerate"]:
        return False

    capture_width, capture_height = SENSOR_MODES[new_fps]
    cam_state["framerate"] = new_fps
    cam_state["capture_width"] = capture_width
    cam_state["capture_height"] = capture_height
    cam_state["exposure_ms"] = clamp_exposure_ms(
        cam_state["exposure_ms"], new_fps
    )
    reopen_camera_with_exposure(camera, cam_state, focuser)
    return True


def adjust_exposure(camera, cam_state, focuser, delta_ms):
    """Step the fixed exposure. Leaving auto exposure starts from 0 ms (going up)
    or from the frame-rate limit (going down)."""
    limit = max_exposure_ms(cam_state["framerate"])
    current = cam_state["exposure_ms"]
    if current is None:
        current = 0.0 if delta_ms > 0 else limit
    cam_state["exposure_ms"] = min(max(current + delta_ms, EXPOSURE_MIN_MS), limit)
    reopen_camera_with_exposure(camera, cam_state, focuser)


def draw_control_overlay(frame, cam_state, focuser, af):
    focus = focuser.get(Focuser.OPT_FOCUS)
    if af.enabled:
        controller = af.controller
        reference = "--" if controller.reference is None else "{:.0f}".format(controller.reference)
        autofocus_text = "AF: {} | scans={} aborts={} ref={}".format(
            controller.state, controller.attempts, controller.aborts, reference)
    else:
        autofocus_text = "AF: MANUAL (F/Enter = continuous AF)"

    exposure_ms = cam_state["exposure_ms"]
    exposure_text = "Exposure: Auto"
    if exposure_ms is not None:
        exposure_text = "Exposure: {:.2f} ms (~1/{:.0f}s)".format(
            exposure_ms, 1000.0 / exposure_ms
        )

    lines = [
        "Focus: {} | {}".format(focus, autofocus_text),
        exposure_text,
        daynight.mode_text(cam_state),
        "Camera: {} FPS ({}x{})".format(
            cam_state["framerate"],
            cam_state["capture_width"],
            cam_state["capture_height"],
        ),
        "[/] exposure 1 ms | {/} 10 ms | A auto exposure | -/= FPS",
        "W/S or Up/Down manual focus | R reset focus | Enter/F continuous AF",
        "C snapshot | Q/Esc exit",
    ]
    if af.enabled and af.roi is not None:
        x, y, roi_width, roi_height = af.roi
        cv2.rectangle(frame, (x, y), (x + roi_width, y + roi_height), (255, 255, 0), 1)
    for index, text in enumerate(lines):
        cv2.putText(frame, text, (20, 85 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(frame, text, (20, 85 + index * 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)


def handle_camera_key(key, camera, cam_state, focuser, af, frame):
    key8 = key & 0xFF
    reopened = False
    if key8 in (ord("-"), ord("_")):
        reopened = change_framerate(camera, cam_state, focuser, -1)
    elif key8 in (ord("="), ord("+")):
        reopened = change_framerate(camera, cam_state, focuser, 1)
    elif key8 == ord("}"):
        adjust_exposure(camera, cam_state, focuser, EXPOSURE_COARSE_STEP_MS)
        reopened = True
    elif key8 == ord("{"):
        adjust_exposure(camera, cam_state, focuser, -EXPOSURE_COARSE_STEP_MS)
        reopened = True
    elif key8 == ord("]") or key in KEY_RIGHT:
        adjust_exposure(camera, cam_state, focuser, EXPOSURE_STEP_MS)
        reopened = True
    elif key8 == ord("[") or key in KEY_LEFT:
        adjust_exposure(camera, cam_state, focuser, -EXPOSURE_STEP_MS)
        reopened = True
    elif key8 in (ord("w"), ord("W")) or key in KEY_UP:
        af.disable()
        focuser.set(Focuser.OPT_FOCUS, focuser.get(Focuser.OPT_FOCUS) + FOCUS_STEP)
    elif key8 in (ord("s"), ord("S")) or key in KEY_DOWN:
        af.disable()
        focuser.set(Focuser.OPT_FOCUS, focuser.get(Focuser.OPT_FOCUS) - FOCUS_STEP)
    elif key8 in (ord("f"), ord("F"), 10, 13) or key == 65421:
        af.enable()
    elif key8 in (ord("r"), ord("R")):
        af.disable()
        focuser.reset(Focuser.OPT_FOCUS)
    elif key8 in (ord("a"), ord("A")):
        cam_state["exposure_ms"] = None
        reopen_camera_with_exposure(camera, cam_state, focuser)
        reopened = True
    elif key8 in (ord("c"), ord("C")):
        cv2.imwrite("v12_image.jpg", frame)
        print("Saved snapshot: v12_image.jpg")
    if reopened:
        # New pipeline, new exposure: the old search state and motion frame are stale.
        af.restart(AF_REOPEN_WARMUP)


def open_video_writer(path, fps, size):
    """MP4 writer: OpenCV's FFmpeg backend (MPEG-4), else a GStreamer x264 pipeline."""
    writer = cv2.VideoWriter(str(path), cv2.CAP_FFMPEG,
                             cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if writer.isOpened():
        return writer, "FFmpeg mp4v"
    writer.release()
    pipeline = (
        "appsrc ! videoconvert ! video/x-raw,format=I420 ! "
        "x264enc speed-preset=ultrafast tune=zerolatency ! h264parse ! "
        f'mp4mux ! filesink location="{path}"'
    )
    writer = cv2.VideoWriter(pipeline, cv2.CAP_GSTREAMER, 0, fps, size)
    if writer.isOpened():
        return writer, "GStreamer x264"
    raise RuntimeError(f"Cannot open an MP4 writer for {path} "
                       "(OpenCV has neither FFmpeg nor GStreamer x264enc)")


class VideoRecorder:
    """Encodes the displayed frames to MP4 on a background thread.

    The file has a fixed frame rate but frames arrive at YOLO's uneven pace, so
    each frame is held or dropped by its capture time: the video plays back in
    real time. A full queue (encoder too slow) drops frames rather than slowing
    recognition down.
    """

    def __init__(self, path, fps, size):
        self.path = Path(path).resolve()
        self.fps = fps
        self.writer, self.backend = open_video_writer(self.path, fps, size)
        self.queue = Queue(maxsize=RECORD_QUEUE_SIZE)
        self.start_time = None
        self.last_image = None
        self.written = 0
        self.dropped = 0
        self.thread = threading.Thread(target=self._run, daemon=True, name="mp4-recorder")
        self.thread.start()

    @property
    def seconds(self):
        return self.written / self.fps

    def add(self, image, timestamp):
        try:
            self.queue.put_nowait((image, timestamp))
        except Full:
            self.dropped += 1

    def _run(self):
        max_hold = max(1, round(self.fps * RECORD_MAX_STALL))
        while True:
            item = self.queue.get()
            if item is None:
                return
            image, timestamp = item
            if self.start_time is None:
                self.start_time = timestamp
            slot = int((timestamp - self.start_time) * self.fps)
            if slot - self.written > max_hold:
                # Long stall: keep only RECORD_MAX_STALL seconds of it.
                self.start_time += (slot - self.written - max_hold) / self.fps
                slot = self.written + max_hold
            # Hold the previous frame until this one's time, then show this one.
            while self.written < slot:
                self.writer.write(self.last_image)
                self.written += 1
            if self.written == slot:
                self.writer.write(image)
                self.written += 1
            self.last_image = image

    def close(self):
        """Encode what is queued and finalise the file (an MP4 is unreadable without it)."""
        self.queue.put(None)
        self.thread.join()
        self.writer.release()


def main():
    global model, half, device

    # `kill <pid>` and a dropped SSH session still run the finally block, so an
    # MP4 being recorded is finalised. SIGHUP is left alone when nohup ignores it.
    signal.signal(signal.SIGTERM, lambda signum, stack: sys.exit(0))
    if hasattr(signal, "SIGHUP") and signal.getsignal(signal.SIGHUP) == signal.SIG_DFL:
        signal.signal(signal.SIGHUP, lambda signum, stack: sys.exit(0))

    args = parse_args()
    model_path = resolve_model_path(args.model)
    inference_device = args.device or device
    capture_width, capture_height = SENSOR_MODES[args.framerate]
    exposure_ms = clamp_exposure_ms(args.exposure, args.framerate)
    exposure_ns = int(exposure_ms * 1_000_000) if exposure_ms is not None else None
    cam_state = {
        "framerate": args.framerate,
        "exposure_ms": exposure_ms,
        "capture_width": capture_width,
        "capture_height": capture_height,
        # gain_range / conf / mode / probe: day profile + startup probe, or --gain / --conf
        **daynight.initial_state(args, exposure_ms),
    }
    gain_range = cam_state["gain_range"]

    print(
        "Opening IMX519: source {}x{} (OCR ROI), preview {}x{} (YOLO) "
        "@ {} FPS, sensor mode {}x{}".format(
            args.source_width, args.source_height, args.width, args.height,
            args.framerate, capture_width, capture_height,
        )
    )
    camera = Camera(
        width=args.source_width,
        height=args.source_height,
        framerate=args.framerate,
        exposure_time_ns=exposure_ns,
        gain_range=gain_range,
        ae_lock=exposure_ms is not None,
        capture_width=capture_width,
        capture_height=capture_height,
        flip_method=args.flip,
    )

    focuser = Focuser(args.i2c_bus)
    af = ContinuousAF(focuser, args)

    def reopen_for_profile():
        """Reopen with cam_state's gain for a day / night profile (raises RuntimeError on failure)."""
        reopen_camera_with_exposure(camera, cam_state, focuser)
        # New pipeline, new gain: the old search state and motion frame are stale.
        af.restart(AF_REOPEN_WARMUP)

    model = YOLO(model_path)
    device = inference_device
    half = inference_device == "cuda"

    warmup_ollama()
    threading.Thread(target=ocr_worker, daemon=True, name="ollama-ocr-worker").start()

    frame_id = 0
    processing_size = (args.width, args.height)

    recorder = None
    try:
        if args.record is not None:
            record_path = args.record or time.strftime("record_%Y%m%d_%H%M%S.mp4")
            record_fps = args.record_fps or min(args.framerate, RECORD_DEFAULT_MAX_FPS)
            recorder = VideoRecorder(record_path, record_fps, (args.width, args.height))
            print(f"Recording {recorder.path} ({recorder.backend}, {record_fps:g} FPS)")

        while True:
            try:
                source_frame = camera.getFrame(timeout=2.0)
            except Exception as error:
                print("等待 IMX519 影像失敗:", error)
                continue
            if source_frame is None:
                continue

            frame_id += 1
            frame_time = time.monotonic()
            # YOLO / AF / 視窗用縮小的處理畫面；看板 ROI 之後從 source_frame 裁切
            frame = cv2.resize(source_frame, processing_size, interpolation=cv2.INTER_AREA)
            af.update(frame, time.monotonic())
            # 開機日夜判斷：量乾淨的處理畫面（畫框、文字之前），收滿 10 秒才套用結果
            daynight.startup_step(cam_state, frame, frame_time, reopen_for_profile)
            # Draw only on a copy: `frame` stays clean for AF and YOLO.
            display = frame.copy()
            buses = detect_bus(frame, cam_state["conf"])
            current_ids = {bus[5] for bus in buses}

            with lock:
                for track_id in current_ids:
                    last_seen_frame[track_id] = frame_id
            if frame_id % CLEANUP_INTERVAL_FRAMES == 0:
                forget_stale_tracks(frame_id)

            for x1, y1, x2, y2, confidence, track_id in buses:
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    continue

                # roi_box 是處理畫面座標（畫框用）；送 OCR 的影像從高解析度來源裁切
                _, roi_box = extract_sign_roi(frame, x1, y1, x2, y2)
                roi = crop_source_roi(source_frame, roi_box, processing_size)
                rx1, ry1, rx2, ry2 = roi_box
                cv2.rectangle(display, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(display, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

                with lock:
                    confirmed_result = confirmed_buses.get(track_id)

                if confirmed_result:
                    cv2.putText(
                        display,
                        f"BUS ID:{track_id} -> {confirmed_result}",
                        (x1, max(25, y1 - 10)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2,
                    )
                    continue

                label = f"BUS ID:{track_id} [OCR]"
                valid_roi = is_valid_roi(roi)
                aspect_ratio = width / height
                if not valid_roi:
                    label = f"BUS ID:{track_id} [ROI TOO SMALL]"
                elif aspect_ratio < ASPECT_RATIO_MIN:
                    # 框太窄：多半是被前車擋住的後車，看板只露出一部分
                    label = f"BUS ID:{track_id} [NARROW]"
                elif aspect_ratio > ASPECT_RATIO_MAX:
                    label = f"BUS ID:{track_id} [ANGLE FILTER]"
                elif af.scanning:
                    # Lens is moving through search positions; OCR resumes once focused.
                    label = f"BUS ID:{track_id} [AF SCAN]"
                elif frame_id % OCR_FRAME_INTERVAL == 0:
                    schedule_ocr(track_id, roi, time.monotonic())

                cv2.putText(
                    display,
                    label,
                    (x1, max(25, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 255, 255),
                    2,
                )

            final_results = get_visible_results(frame_id)
            display_text = ",".join(final_results) if final_results else "NONE"
            color = (255, 0, 0) if final_results else (0, 165, 255)
            cv2.putText(
                display,
                f"BUS NUMBER: {display_text} | IMX519 {cam_state['framerate']} FPS",
                (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                color,
                2,
            )
            draw_control_overlay(display, cam_state, focuser, af)
            if recorder is not None:
                recorder.add(display, frame_time)
            cv2.imshow("Bus Detection", display)

            key = cv2.waitKeyEx(10)
            if key == -1:
                continue
            if key == 27 or (key & 0xFF) in (ord("q"), ord("Q")):
                break
            if (key & 0xFF) in REOPEN_KEY_CODES or key in KEY_LEFT or key in KEY_RIGHT:
                # 手動改曝光 / FPS 會讓量到的亮度失準：以使用者的設定為準，維持白天參數
                daynight.cancel_startup(cam_state, reopen_for_profile)
            handle_camera_key(key, camera, cam_state, focuser, af, display)
    finally:
        if recorder is not None:
            recorder.close()
        camera.close()
        cv2.destroyAllWindows()
        if recorder is not None:
            print(f"Saved video: {recorder.path} ({recorder.seconds:.1f} s, "
                  f"{recorder.dropped} frames dropped)")


if __name__ == "__main__":
    main()

