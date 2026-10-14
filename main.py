import base64
import math
import re
import threading
import time
from queue import Full, Queue

import cv2
import numpy as np
import requests
import torch
from ultralytics import YOLO

# =========================
# Ollama 設定
# =========================
OLLAMA_URL = "http://localhost/api/chat"
MODEL_NAME = "qwen3-vl:4B-instruct"
# 每次 OCR 只用約 100～320 個 token（提示詞 + 一張 ROI；實測最大 317，為 512x512 的 ROI），2048 已足夠；
# context 越小 KV cache 越小，Jetson 的 CPU/GPU 共用記憶體較不易吃緊。
# context 只在模型載入時決定：模型已載入時，請求帶較小的 num_ctx 不會重新載入（Ollama 0.17.7 實測 ps 不變）。
# 要確實套用新值，先在伺服器執行 ollama stop qwen3-vl:4B-instruct 卸載模型，再啟動本程式。
OLLAMA_NUM_CTX = 2048

# =========================
# GPU / YOLO
# =========================
device = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", device)

model = YOLO("yolov8n.pt")
half = device == "cuda"
VIDEO_PATH = "C:/Users/Mitnick/Desktop/FCU/VS_Code/ocr_codex/bus_image/bus_2.mp4"
# 偵測、OCR 與顯示畫面固定為橫式 1280x720 或直式 720x1280，依影片方向自動選擇；
# 長寬比不是 16:9 的影片（例如 4:3）等比例縮放後補黑邊，不會變形。
LANDSCAPE_SIZE = (1280, 720)
PORTRAIT_SIZE = (720, 1280)
YOLO_STRIDE = 32
WINDOW_NAME = "Bus Detection"

# =========================
# 執行參數
# =========================
GRACE_FRAMES = 15
OCR_FRAME_INTERVAL = 5
OCR_INTERVAL_SECONDS = 1.0
OCR_QUEUE_SIZE = 2
CONFIRMATIONS_REQUIRED = 2
# 看板讀不清楚時，模型會把提示詞的「1 至 4 位數字」照抄成 1234；視同 NONE。
# bus_21 實測：過曝看板依序回傳 1234 -> NONE -> 1234，因 NONE 不重置計數而被確認成 1234。
INVALID_RESULTS = {"1234"}
# 追蹤 ID 超過此影格數未出現就清除其狀態，避免長時間執行時字典持續成長。
# ByteTrack 丟失 30 格後即移除該 ID 且不再重用，因此必須大於 30 與 GRACE_FRAMES。
STALE_TRACK_FRAMES = 300
PRUNE_INTERVAL_FRAMES = 100

# 任一邊小於 32 px 的圖片會讓 qwen3-vl runner 當掉（回 500）；小 ROI 放大後送出，大 ROI 縮到 MAX_OCR_SIDE。
OLLAMA_MIN_IMAGE_SIDE = 32
TARGET_OCR_MIN_SIDE = 96
MAX_OCR_SIDE = 512
MIN_SOURCE_ROI_WIDTH = 4
MIN_SOURCE_ROI_HEIGHT = 4
MIN_SOURCE_ROI_AREA = 48

# =========================
# 共用狀態
# =========================
confirmed_buses = {}
last_seen_frame = {}
last_ocr_time = {}
ocr_history = {}
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
    """在 LAB 的 L 通道依序執行 CLAHE、Gamma 與輕度銳化。

    Gamma 係數由原始 L 通道估算，目標均值 128、範圍 0.55 至 2.5；
    估算後先執行 CLAHE，再套用 Gamma。a、b 通道保留供色彩重建。
    不使用雙邊濾波、高光壓縮或獨立高斯平滑。
    """
    if roi_img is None or roi_img.size == 0:
        raise ValueError("ROI 不可為空")

    lab = cv2.cvtColor(roi_img, cv2.COLOR_BGR2LAB)
    luma, channel_a, channel_b = cv2.split(lab)

    mean_luma = float(luma.mean())
    contrast = float(luma.std())
    white_ratio = overexposed_ratio(luma)
    gamma_used = auto_gamma(luma, target_mean=128.0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    luma = clahe.apply(luma)
    luma = apply_gamma(luma, gamma_used)
    luma = unsharp_mask(luma)
    enhanced = cv2.cvtColor(cv2.merge((luma, channel_a, channel_b)), cv2.COLOR_LAB2BGR)
    print(
        f"[enhance_roi] clahe-gamma, mean={mean_luma:.1f}, "
        f"contrast={contrast:.1f}, white={white_ratio:.1%}, "
        f"gamma={gamma_used:.2f}, CLAHE=True, target=128"
    )
    return enhanced


def resize_for_ocr(roi_img: np.ndarray) -> np.ndarray:
    """等比例縮放 ROI：小圖放大（任一邊 < 32 px 會讓 Ollama runner 當掉），大圖縮到 MAX_OCR_SIDE。

    Ollama 會再把影像縮放成 32 的倍數，未滿 65536 像素的一律放大到約 65536 像素，
    但大圖不會縮小；不先縮小時，1280x720 的 ROI 會用掉約 880 個影像 token。
    """
    height, width = roi_img.shape[:2]
    scale = max(
        1.0,
        OLLAMA_MIN_IMAGE_SIDE / width,
        OLLAMA_MIN_IMAGE_SIDE / height,
        TARGET_OCR_MIN_SIDE / width,
        TARGET_OCR_MIN_SIDE / height,
    )

    max_scaled_side = max(width, height) * scale
    if max_scaled_side > MAX_OCR_SIDE:
        scale = MAX_OCR_SIDE / max(width, height)

    if scale == 1.0:
        return roi_img

    output_width = max(OLLAMA_MIN_IMAGE_SIDE, round(width * scale))
    output_height = max(OLLAMA_MIN_IMAGE_SIDE, round(height * scale))
    print(
        f"[resize_for_ocr] {width}x{height} -> "
        f"{output_width}x{output_height}, scale={scale:.2f}"
    )
    return cv2.resize(
        roi_img,
        (output_width, output_height),
        interpolation=cv2.INTER_CUBIC if scale > 1.0 else cv2.INTER_AREA,
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


def is_valid_roi(roi_img: np.ndarray) -> bool:
    """只過濾空圖或極端小圖；其餘小 ROI 交給 resize_for_ocr 放大。"""
    if roi_img is None or roi_img.size == 0:
        return False
    height, width = roi_img.shape[:2]
    return (
        width >= MIN_SOURCE_ROI_WIDTH
        and height >= MIN_SOURCE_ROI_HEIGHT
        and width * height >= MIN_SOURCE_ROI_AREA
    )


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
    prepared_roi = enhance_roi(resize_for_ocr(roi_img))
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
        # 模型預設 temperature=1 會隨機取樣；固定為 0 讓同一張圖得到相同答案，較快湊到連續確認。
        "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": 0},
    }
    response = ollama_session.post(OLLAMA_URL, json=payload, timeout=60)
    response.raise_for_status()
    raw_result = response.json()["message"]["content"].strip()
    result = normalize_ocr_result(raw_result)
    print(f"OCR [ID:{track_id}] = {raw_result!r}")

    if result is None:
        return

    with lock:
        history = ocr_history.get(track_id)
        if history and history["last"] == result:
            history["count"] += 1
        else:
            history = {"last": result, "count": 1}
            ocr_history[track_id] = history

        print(f"ID:{track_id} {result} ({history['count']}/{CONFIRMATIONS_REQUIRED})")
        if history["count"] >= CONFIRMATIONS_REQUIRED:
            confirmed_buses[track_id] = result
            print(f"CONFIRMED [ID:{track_id}] {result}")


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
def letterbox_params(height: int, width: int) -> tuple[int, int, int, int, float, int, int]:
    """原始影格 -> 顯示畫面的轉換：(目標寬, 目標高, 縮放後寬, 縮放後高, 縮放比例, 左黑邊, 上黑邊)。

    prepare_display 與 crop_source_roi 共用，確保兩邊的座標換算一致。
    """
    target_width, target_height = PORTRAIT_SIZE if height > width else LANDSCAPE_SIZE
    scale = min(target_width / width, target_height / height)
    new_width, new_height = round(width * scale), round(height * scale)
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    return target_width, target_height, new_width, new_height, scale, offset_x, offset_y


def prepare_display(frame: np.ndarray) -> np.ndarray:
    """依影片方向輸出 1280x720 或 720x1280；長寬比不同時等比例縮放並置中補黑邊，不拉伸。

    一律回傳新陣列：之後在顯示畫面上畫框，不會污染用來裁切 ROI 的原始影格。
    """
    height, width = frame.shape[:2]
    target_width, target_height, new_width, new_height, scale, x, y = letterbox_params(height, width)
    if (new_width, new_height) == (width, height):
        resized = frame.copy()
    else:
        resized = cv2.resize(
            frame,
            (new_width, new_height),
            interpolation=cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR,
        )
    if (new_width, new_height) == (target_width, target_height):
        return resized

    canvas = np.zeros((target_height, target_width, 3), dtype=frame.dtype)
    canvas[y:y + new_height, x:x + new_width] = resized
    return canvas


def crop_source_roi(source_frame: np.ndarray,
                    roi_box: tuple[int, int, int, int]) -> np.ndarray:
    """把顯示畫面座標的 ROI 換算回原始影格再裁切。

    高解析度影片（例如 2160x3840）縮成 720x1280 後只剩 1/3 解析度；從原始影格裁切可保留遠處看板的細節，
    過大的 ROI 之後再由 resize_for_ocr 縮到 MAX_OCR_SIDE。
    """
    height, width = source_frame.shape[:2]
    *_, scale, offset_x, offset_y = letterbox_params(height, width)
    rx1, ry1, rx2, ry2 = roi_box
    sx1, sx2 = (min(max(round((x - offset_x) / scale), 0), width) for x in (rx1, rx2))
    sy1, sy2 = (min(max(round((y - offset_y) / scale), 0), height) for y in (ry1, ry2))
    return source_frame[sy1:sy2, sx1:sx2]


def detect_bus(frame):
    # imgsz 依畫面方向取 stride 倍數，避免直式畫面被塞進橫式輸入而縮小
    height, width = frame.shape[:2]
    imgsz = (
        math.ceil(height / YOLO_STRIDE) * YOLO_STRIDE,
        math.ceil(width / YOLO_STRIDE) * YOLO_STRIDE,
    )
    results = model.track(
        frame,
        imgsz=imgsz,
        device=device,
        classes=[5],
        conf=0.2,
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
        numbers = [confirmed_buses[track_id] for track_id in sorted(visible_ids)]
        # 同一台車可能因 ID 切換或長車身被拆成前後兩框而有多個 ID，顯示時去除重複號碼
        return list(dict.fromkeys(numbers))


def prune_stale_tracks(frame_id: int):
    """清除長時間未出現的追蹤 ID；仍在 OCR 中的 ID 保留，避免 worker 寫回已刪除的狀態。"""
    with lock:
        stale_ids = [
            track_id
            for track_id, last_frame in last_seen_frame.items()
            if frame_id - last_frame > STALE_TRACK_FRAMES and track_id not in ocr_pending_ids
        ]
        for track_id in stale_ids:
            for state in (confirmed_buses, last_seen_frame, last_ocr_time, ocr_history):
                state.pop(track_id, None)


def main():
    warmup_ollama()
    threading.Thread(target=ocr_worker, daemon=True, name="ollama-ocr-worker").start()

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"無法開啟影片: {VIDEO_PATH}")

    frame_id = 0
    aspect_ratio_min = 0.0
    aspect_ratio_max = 1.8
    # Win32 版 OpenCV 不支援 WINDOW_KEEPRATIO，WINDOW_NORMAL 會把影像拉伸到視窗大小；
    # 用 AUTOSIZE 讓視窗等於畫面大小（1280x720 或 720x1280），並放在螢幕左上角。
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    cv2.moveWindow(WINDOW_NAME, 0, 0)

    while True:
        ret, source_frame = cap.read()
        if not ret:
            break

        frame_id += 1
        # 偵測與顯示用 1280x720 / 720x1280 的畫面；ROI 從未縮小、未畫框的原始影格裁切
        frame = prepare_display(source_frame)
        if frame_id == 1:
            print(f"Source size: {source_frame.shape[1]}x{source_frame.shape[0]}, "
                  f"display size: {frame.shape[1]}x{frame.shape[0]}")
        buses = detect_bus(frame)

        with lock:
            # 記錄所有 ID（不只已確認的），才能判斷哪些 ID 已消失並清除
            for bus in buses:
                last_seen_frame[bus[5]] = frame_id

        for x1, y1, x2, y2, confidence, track_id in buses:
            width, height = x2 - x1, y2 - y1
            if width <= 0 or height <= 0:
                continue

            # extract_sign_roi 只用來算顯示座標的看板框，影像改從原始影格裁切
            _, roi_box = extract_sign_roi(frame, x1, y1, x2, y2)
            roi = crop_source_roi(source_frame, roi_box)
            rx1, ry1, rx2, ry2 = roi_box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(frame, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)

            with lock:
                confirmed_result = confirmed_buses.get(track_id)

            if confirmed_result:
                cv2.putText(
                    frame,
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
            elif not aspect_ratio_min <= aspect_ratio <= aspect_ratio_max:
                label = f"BUS ID:{track_id} [ANGLE FILTER]"
            elif frame_id % OCR_FRAME_INTERVAL == 0:
                roi_height, roi_width = roi.shape[:2]
                if roi_width < TARGET_OCR_MIN_SIDE or roi_height < TARGET_OCR_MIN_SIDE:
                    label = f"BUS ID:{track_id} [ROI UPSCALE]"
                schedule_ocr(track_id, roi, time.monotonic())

            cv2.putText(
                frame,
                label,
                (x1, max(25, y1 - 10)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 255),
                2,
            )

        if frame_id % PRUNE_INTERVAL_FRAMES == 0:
            prune_stale_tracks(frame_id)

        final_results = get_visible_results(frame_id)
        display_text = ",".join(final_results) if final_results else "NONE"
        color = (255, 0, 0) if final_results else (0, 165, 255)
        cv2.putText(
            frame,
            f"BUS NUMBER: {display_text}",
            (20, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            color,
            2,
        )
        cv2.imshow(WINDOW_NAME, frame)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
