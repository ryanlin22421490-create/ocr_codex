"""用錄影檔模擬公車號碼語音播報，並分析前後排序。

把之前錄的影片丟進來，程式會照 nohead_onetime_main.py 的流程跑一次：
YOLO + ByteTrack 追蹤、看板 ROI、[NARROW] / [ANGLE FILTER] / [ROI TOO SMALL] 過濾、
一次確認的 OCR，再交給 announcer.py 決定什麼時候、怎麼播報。
時間一律用影片時間（第幾幀 / FPS），所以結果可以重現。

用法（在 IMX519 資料夾或專案根目錄都可以）：
    python3 IMX519/announce_sim.py bus_image/bus_2.mp4
    python3 IMX519/announce_sim.py bus_image/*.mp4 --save-video
    python3 IMX519/announce_sim.py bus_image                    # 資料夾裡的所有影片
    python3 IMX519/announce_sim.py video.mp4 --ollama-url http://192.168.0.110:11434/api/chat
    python3 IMX519/announce_sim.py video.mp4 --ocr track-id     # 不接 Ollama，只測排序與播報時機

輸出（--out-dir，預設 announce_sim_out/<影片名稱>/）：
    log.txt               完整紀錄（OCR、確認、播報、略過 / 丟棄的原因）
    announcements.csv     每次播報與略過 / 丟棄
    <影片名稱>.srt         播報字幕，時間對齊原影片；和影片放在同一個資料夾，VLC 會自動載入
    order_frames.csv      每一個同時有 2 輛以上公車的幀：框下緣排序與面積、框高排序
    order_disagree/       框下緣排序和面積排序不一致的畫面（人工確認哪個才對）
    <影片名稱>.mp4         --save-video：標註後的影片（框、前後名次、播報橫幅）
    summary.txt           統計
ocr_cache.json 放在 --out-dir：同一張 ROI 只問 Ollama 一次，之後改播報參數重跑不用等 OCR。

與實際執行的差異：
- OCR 回應時間：預設用實際送 Ollama 量到的往返時間，在影片時間軸上延後確認
  （一條 OCR 執行緒、佇列 2 筆，和 nohead 相同）。--ocr-latency 可改成固定秒數。
- 沒有自動對焦，所以沒有 [AF SCAN]；也沒有日夜判斷，YOLO 門檻用 --conf。
- 錄影若是 nohead --record 錄的，畫面上已經畫了框和文字，偵測與 OCR 可能和原始畫面略有差異。
- 不會發出聲音：播報內容寫在紀錄和字幕；播報長度用字數估計。
"""
import argparse
import csv
import hashlib
import json
import math
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import nohead_onetime_main as live   # 沿用即時程式的常數與函式（不會開相機）
import announcer
from announcer import AnnouncerCore, Box, Event, order_by_area, order_by_position
from ultralytics import YOLO

DEFAULT_FPS = 17.0
FAKE_OCR_LATENCY = 1.0
PROGRESS_INTERVAL = 5.0        # 秒（實際時間）：印出處理進度的間隔
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov", ".mkv"}
ORDER_LABELS = {"前方": "FRONT", "後方": "REAR"}


# =========================
# 參數
# =========================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Simulate bus-number voice announcements on recorded videos "
                    "(same detection / OCR rules as nohead_onetime_main.py).")
    parser.add_argument("videos", nargs="+", help="Video files")
    parser.add_argument("--out-dir", default="announce_sim_out",
                        help="Output folder; one sub-folder per video. Default: announce_sim_out")
    parser.add_argument("--model", default=None,
                        help="YOLO model path; default searches this folder and its parent")
    parser.add_argument("--device", default=None, help="Inference device, e.g. 0, cpu")
    parser.add_argument("--conf", type=float, default=0.5,
                        help="YOLO bus confidence (day 0.5, night 0.3). Default: 0.5")
    parser.add_argument("--width", type=int, default=1280,
                        help="Processing frame long side (videos are never enlarged). Default: 1280")
    parser.add_argument("--fps", type=float, default=None,
                        help="Override the video FPS (use when the file reports a wrong value)")
    parser.add_argument("--ocr", choices=("ollama", "track-id"), default="ollama",
                        help="ollama: real OCR; track-id: pretend every bus reads as its track ID "
                             "(no server, tests ordering and timing only). Default: ollama")
    parser.add_argument("--ollama-url", default=live.OLLAMA_URL,
                        help=f"Ollama chat API. Default: {live.OLLAMA_URL}")
    parser.add_argument("--ocr-latency", type=float, default=None,
                        help="Fixed OCR round-trip seconds on the video timeline; default: measured "
                             f"(ollama) or {FAKE_OCR_LATENCY:g} s (track-id)")
    parser.add_argument("--no-cache", action="store_true", help="Do not use ocr_cache.json")
    parser.add_argument("--save-video", action="store_true",
                        help="Save the annotated video next to the subtitles")
    parser.add_argument("--max-order-images", type=int, default=200,
                        help="Max order-disagreement images per video. Default: 200")
    parser.add_argument("--order-image-interval", type=float, default=1.0,
                        help="Seconds of video between two saved disagreement images. Default: 1")
    group = parser.add_argument_group("announcement rules (announcer.py)")
    group.add_argument("--announce-window", type=float, default=announcer.DEFAULT_WINDOW,
                       help=f"Seconds to collect numbers into one sentence. Default: {announcer.DEFAULT_WINDOW:g}")
    group.add_argument("--announce-cooldown", type=float, default=announcer.DEFAULT_COOLDOWN,
                       help=f"Seconds before the same number is announced again. Default: {announcer.DEFAULT_COOLDOWN:g}")
    group.add_argument("--announce-max-age", type=float, default=announcer.DEFAULT_MAX_AGE,
                       help=f"Drop a confirmation not announced within this many seconds. Default: {announcer.DEFAULT_MAX_AGE:g}")
    group.add_argument("--announce-order", choices=announcer.ORDERS, default=announcer.DEFAULT_ORDER,
                       help="Front bus: bottom (head-on camera), left / right (side camera, buses "
                            "driving left / right). Default: bottom")
    group.add_argument("--announce-min-height", type=int, default=0,
                       help="Boxes lower than this (px) do not count as a bus in the queue. Default: 0")
    group.add_argument("--no-cross-check", action="store_true",
                       help="Say front / rear even when the box-area order disagrees")
    group.add_argument("--number-style", choices=announcer.NUMBER_STYLES,
                       default=announcer.DEFAULT_NUMBER_STYLE,
                       help="mixed: 1-2 digits as a value, 3-4 digit by digit. Default: mixed")
    args = parser.parse_args()
    if not 0 < args.conf < 1:
        parser.error("--conf must be between 0 and 1")
    if args.width < 64:
        parser.error("--width must be at least 64")
    if args.fps is not None and args.fps <= 0:
        parser.error("--fps must be > 0")
    if args.ocr_latency is not None and args.ocr_latency < 0:
        parser.error("--ocr-latency must be >= 0")
    return args


def make_announcer(args):
    return AnnouncerCore(
        window=args.announce_window, cooldown=args.announce_cooldown,
        max_age=args.announce_max_age, order=args.announce_order,
        min_height=args.announce_min_height, cross_check=not args.no_cross_check,
        number_style=args.number_style,
    )


# =========================
# 小工具
# =========================
def video_clock(seconds: float) -> str:
    minutes, rest = divmod(max(seconds, 0.0), 60)
    return f"{int(minutes):02d}:{rest:04.1f}"


def srt_clock(seconds: float) -> str:
    millis = int(round(max(seconds, 0.0) * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"


def processing_size(width: int, height: int, long_side: int) -> tuple[int, int]:
    scale = min(1.0, long_side / max(width, height))
    return (max(2, round(width * scale / 2) * 2), max(2, round(height * scale / 2) * 2))


def yolo_imgsz(size: tuple[int, int]) -> tuple[int, int]:
    """(高, 寬) 進位到 64 的倍數：1280x720 -> (768, 1280)，與 nohead 的 imgsz 相同。"""
    width, height = size
    return (math.ceil(height / 64) * 64, math.ceil(width / 64) * 64)


def overlay_text(record) -> str:
    """cv2.putText 不能畫中文：橫幅用英文標示前後。"""
    parts = []
    for event in record.events:
        word = ""
        if record.positioned:
            word = announcer.position_word(record.order.index(event.track_id),
                                           record.visible_count)
            word = ORDER_LABELS.get(word, word.replace("第", "#").replace("輛", ""))
        parts.append(f"{word} {event.number}".strip())
    return "ANNOUNCE: " + ", ".join(parts)


class Log:
    def __init__(self, path: Path):
        self.file = open(path, "w", encoding="utf-8")

    def __call__(self, message: str):
        print(message)
        self.file.write(message + "\n")

    def close(self):
        self.file.close()


# =========================
# OCR（在影片時間軸上模擬單一 worker 與佇列）
# =========================
class OcrCache:
    def __init__(self, path: Path | None):
        self.path = path
        self.data = {}
        if path is not None and path.exists():
            try:
                self.data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError) as error:
                print(f"Ignoring unreadable OCR cache {path}: {error}")

    def get(self, key):
        return self.data.get(key)

    def put(self, key, raw, rtt):
        self.data[key] = {"raw": raw, "rtt": rtt}

    def save(self):
        if self.path is not None:
            self.path.write_text(json.dumps(self.data, ensure_ascii=False, indent=0),
                                 encoding="utf-8")


class SimulatedOcr:
    """與 nohead 相同：一條 worker、佇列 OCR_QUEUE_SIZE 筆、每個 ID 冷卻 OCR_INTERVAL_SECONDS。

    OCR 在送出時就實際執行（或讀快取），結果延後到影片時間的 finish 才生效：
    start = max(送出時間, worker 空閒時間)，finish = start + 往返時間。
    """

    def __init__(self, args, cache: OcrCache):
        self.mode = args.ocr
        self.url = args.ollama_url
        self.fixed_latency = args.ocr_latency
        self.cache = cache
        self.session = requests.Session()
        self.session.trust_env = False
        self.stats = {"sent": 0, "cached": 0, "errors": 0, "full": 0}
        self.reset()

    def reset(self):
        self.jobs = []
        self.worker_free_at = 0.0
        self.pending_ids = set()
        self.last_ocr_time = {}

    def warmup(self):
        if self.mode != "ollama":
            return
        print(f"Ollama warming up ({self.url})...")
        dummy = np.zeros((live.OLLAMA_MIN_IMAGE_SIDE, live.OLLAMA_MIN_IMAGE_SIDE, 3), np.uint8)
        payload = {
            "model": live.MODEL_NAME,
            "messages": [{"role": "system", "content": "warmup"},
                         {"role": "user", "content": "OK", "images": [live.encode_frame(dummy)]}],
            "stream": False, "keep_alive": -1,
            "options": {"num_ctx": live.OLLAMA_NUM_CTX, "num_predict": 1, "temperature": 0},
        }
        try:
            self.session.post(self.url, json=payload, timeout=120).raise_for_status()
            print("Ollama ready")
        except requests.RequestException as error:
            print("Ollama warmup failed (OCR requests will be tried anyway):", error)

    def schedule(self, track_id, roi, now) -> bool:
        """同 nohead.schedule_ocr：處理中、冷卻中或佇列滿就不送。"""
        if track_id in self.pending_ids:
            return False
        if now - self.last_ocr_time.get(track_id, float("-inf")) < live.OCR_INTERVAL_SECONDS:
            return False
        waiting = sum(1 for job in self.jobs if job["start"] > now)
        if waiting >= live.OCR_QUEUE_SIZE:
            self.stats["full"] += 1
            return False
        job = self._run(track_id, roi)
        job["sent"] = now
        job["start"] = max(now, self.worker_free_at)
        job["finish"] = job["start"] + job["rtt"]
        self.worker_free_at = job["finish"]
        self.jobs.append(job)
        self.pending_ids.add(track_id)
        self.last_ocr_time[track_id] = now
        return True

    def release(self, now) -> list:
        done = [job for job in self.jobs if job["finish"] <= now]
        self.jobs = [job for job in self.jobs if job["finish"] > now]
        for job in done:
            self.pending_ids.discard(job["track_id"])
        return sorted(done, key=lambda job: job["finish"])

    def _run(self, track_id, roi) -> dict:
        roi_height, roi_width = roi.shape[:2]
        job = {"track_id": track_id, "roi_size": f"{roi_width}x{roi_height}",
               "sent_size": "-", "raw": None, "result": None, "error": None, "cached": False}
        if self.mode == "track-id":
            job["raw"] = str(track_id % 10000)
            job["rtt"] = FAKE_OCR_LATENCY if self.fixed_latency is None else self.fixed_latency
            job["result"] = live.normalize_ocr_result(job["raw"])
            return job

        try:
            prepared = live.enhance_roi(live.shrink_for_ocr(roi))
        except (ValueError, cv2.error) as error:
            job["error"], job["rtt"] = str(error), 0.0
            return job
        sent_height, sent_width = prepared.shape[:2]
        job["sent_size"] = f"{sent_width}x{sent_height}"
        image = live.encode_frame(prepared)
        key = hashlib.sha1(f"{live.MODEL_NAME}\n{image}".encode("ascii")).hexdigest()
        cached = self.cache.get(key)
        if cached is not None:
            job["raw"], rtt = cached["raw"], cached["rtt"]
            job["cached"] = True
            self.stats["cached"] += 1
        else:
            started = time.monotonic()
            try:
                job["raw"] = self._ask(image)
                rtt = time.monotonic() - started
                self.cache.put(key, job["raw"], rtt)
                self.stats["sent"] += 1
            except (requests.RequestException, KeyError, ValueError) as error:
                job["error"] = str(error)
                rtt = time.monotonic() - started
                self.stats["errors"] += 1
        job["rtt"] = rtt if self.fixed_latency is None else self.fixed_latency
        if job["raw"] is not None:
            job["result"] = live.normalize_ocr_result(job["raw"])
        return job

    def _ask(self, image_b64) -> str:
        # 提示詞與 nohead_onetime_main.process_ocr 相同；改那邊時這裡也要改
        payload = {
            "model": live.MODEL_NAME,
            "messages": [
                {"role": "system",
                 "content": "你是公車 LED 號碼辨識系統。只能回傳 1 至 4 位數字；無法辨識時只回傳 NONE。"},
                {"role": "user", "content": "辨識圖片中的公車路線號碼。", "images": [image_b64]},
            ],
            "stream": False, "keep_alive": -1,
            "options": {"num_ctx": live.OLLAMA_NUM_CTX, "temperature": 0},
        }
        response = self.session.post(self.url, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["message"]["content"].strip()


# =========================
# 排序分析
# =========================
class OrderStats:
    def __init__(self, out_dir: Path, args):
        self.args = args
        self.image_dir = out_dir / "order_disagree"
        self.file = open(out_dir / "order_frames.csv", "w", newline="", encoding="utf-8-sig")
        self.writer = csv.writer(self.file)
        self.writer.writerow(["video_time", "frame", "buses", "bottom_order", "area_order",
                              "height_order", "area_agrees", "height_agrees", "boxes"])
        self.frames = self.area_agree = self.height_agree = 0
        self.saved = 0
        self.last_saved = float("-inf")

    def update(self, frame, frame_id, now, boxes: dict):
        if len(boxes) < 2:
            return
        by_position = order_by_position(boxes, self.args.announce_order)
        by_area = order_by_area(boxes)
        by_height = sorted(boxes, key=lambda track_id: (-boxes[track_id].height, track_id))
        area_ok, height_ok = by_position == by_area, by_position == by_height
        self.frames += 1
        self.area_agree += area_ok
        self.height_agree += height_ok
        self.writer.writerow([
            f"{now:.2f}", frame_id, len(boxes), " ".join(map(str, by_position)),
            " ".join(map(str, by_area)), " ".join(map(str, by_height)), int(area_ok),
            int(height_ok), json.dumps({str(k): list(v) for k, v in boxes.items()}),
        ])
        if (not area_ok and self.saved < self.args.max_order_images
                and now - self.last_saved >= self.args.order_image_interval):
            self._save(frame, frame_id, now, boxes, by_position, by_area)

    def _save(self, frame, frame_id, now, boxes, by_position, by_area):
        self.image_dir.mkdir(exist_ok=True)
        image = frame.copy()
        for track_id, box in boxes.items():
            pos, area = by_position.index(track_id) + 1, by_area.index(track_id) + 1
            color = (0, 255, 0) if pos == area else (0, 0, 255)
            cv2.rectangle(image, (box.x1, box.y1), (box.x2, box.y2), color, 2)
            text = f"ID{track_id} bottom#{pos} area#{area}"
            cv2.putText(image, text, (box.x1, max(20, box.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4, cv2.LINE_AA)
            cv2.putText(image, text, (box.x1, max(20, box.y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)
        cv2.putText(image, video_clock(now), (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        path = self.image_dir / f"{frame_id:06d}_{now:07.2f}s.jpg"
        cv2.imwrite(str(path), image)
        self.saved += 1
        self.last_saved = now

    def summary(self) -> list[str]:
        if not self.frames:
            return ["Frames with 2+ buses: 0 (no front / rear comparison possible)"]
        return [
            f"Frames with 2+ buses: {self.frames}",
            f"  area order agrees with bottom-edge order:   {self.area_agree / self.frames:.1%}",
            f"  height order agrees with bottom-edge order: {self.height_agree / self.frames:.1%}",
            f"  disagreement images saved: {self.saved} ({self.image_dir.name}/)",
        ]

    def close(self):
        self.file.close()


# =========================
# 單支影片
# =========================
def draw_frame(frame, annotations, banner, now, order_rank):
    image = frame.copy()
    for (x1, y1, x2, y2), (rx1, ry1, rx2, ry2), label, color, track_id in annotations:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.rectangle(image, (rx1, ry1), (rx2, ry2), (0, 0, 255), 2)
        rank = order_rank.get(track_id)
        text = label if rank is None else f"#{rank} {label}"
        cv2.putText(image, text, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(image, video_clock(now), (20, image.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    if banner:
        cv2.rectangle(image, (0, 0), (image.shape[1], 60), (0, 0, 0), -1)
        cv2.putText(image, banner, (20, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (0, 255, 255), 2, cv2.LINE_AA)
    return image


def simulate_video(path: Path, args, model_path, ocr: SimulatedOcr, out_root: Path):
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        print(f"Cannot open {path}; skipped")
        return None
    fps = args.fps or capture.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 240:
        fps = DEFAULT_FPS
        print(f"{path.name}: FPS unknown, using {fps:g} (set --fps if wrong)")
    total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_dir = out_root / path.stem
    out_dir.mkdir(parents=True, exist_ok=True)
    log = Log(out_dir / "log.txt")
    order_stats = OrderStats(out_dir, args)
    announce_file = open(out_dir / "announcements.csv", "w", newline="", encoding="utf-8-sig")
    announce_csv = csv.writer(announce_file)
    announce_csv.writerow(["video_time", "kind", "text", "speech", "duration", "track_ids",
                           "numbers", "confirmed_at", "buses_in_frame", "front_to_rear",
                           "area_agrees", "says_position", "reason"])
    subtitles = []

    model = YOLO(model_path)          # 新的 YOLO：ByteTrack 的 ID 每支影片從頭編
    core = make_announcer(args)
    ocr.reset()
    confirmed, last_seen, last_box, angled = {}, {}, {}, {}
    counts = {"ANNOUNCE": 0, "SKIP": 0, "DROP": 0, "positioned": 0, "unpositioned_multi": 0}
    writer = None
    banner, banner_until = "", float("-inf")
    frame_id, now, size = 0, 0.0, None
    visible = {}
    wall_start = last_progress = time.monotonic()

    def handle_records(records):
        nonlocal banner, banner_until
        for record in records:
            ids = " ".join(str(e.track_id) for e in record.events)
            numbers = " ".join(e.number for e in record.events)
            confirmed_at = " ".join(f"{e.time:.2f}" for e in record.events)
            counts[record.kind] += 1
            if record.kind == "ANNOUNCE":
                core.mark_busy_until(record.time + record.duration)
                counts["positioned"] += record.positioned
                counts["unpositioned_multi"] += record.visible_count > 1 and not record.positioned
                note = "" if record.consistent else " [area disagrees: no front / rear]"
                log(f"[{video_clock(record.time)}] ANNOUNCE {record.text} "
                    f"(IDs {ids}, {record.visible_count} bus(es), front->rear "
                    f"{record.order}){note}")
                subtitles.append((record.time, record.time + record.duration, record.text))
                banner, banner_until = overlay_text(record), record.time + record.duration
            else:
                log(f"[{video_clock(record.time)}] ANNOUNCE {record.kind} {numbers} "
                    f"(ID {ids}): {record.reason}")
            announce_csv.writerow([
                f"{record.time:.2f}", record.kind, record.text, record.speech,
                f"{record.duration:.2f}", ids, numbers, confirmed_at, record.visible_count,
                " ".join(map(str, record.order)), int(record.consistent),
                int(record.positioned), record.reason,
            ])

    def handle_ocr(jobs):
        for job in jobs:
            track_id = job["track_id"]
            source = " (cached)" if job["cached"] else ""
            if job["error"]:
                log(f"[{video_clock(job['finish'])}] OCR error [ID:{track_id}]: {job['error']}")
                continue
            log(f"[{video_clock(job['finish'])}] OCR [ID:{track_id}] = {job['raw']!r} | ROI "
                f"{job['roi_size']} -> sent {job['sent_size']} | {job['rtt']:.2f} s{source}")
            if job["result"] is None or track_id in confirmed:
                continue
            confirmed[track_id] = job["result"]
            log(f"[{video_clock(job['finish'])}] CONFIRMED [ID:{track_id}] {job['result']}")
            core.notify(Event(track_id, job["result"], job["finish"]))

    log(f"=== {path} ({fps:g} FPS, {total_frames or '?'} frames)")
    try:
        while True:
            ok, source_frame = capture.read()
            if not ok or source_frame is None:
                break
            frame_id += 1
            now = (frame_id - 1) / fps
            if size is None:
                size = processing_size(source_frame.shape[1], source_frame.shape[0], args.width)
                imgsz = yolo_imgsz(size)
                log(f"Source {source_frame.shape[1]}x{source_frame.shape[0]}, processing "
                    f"{size[0]}x{size[1]}, YOLO imgsz {imgsz}")
                if args.save_video:
                    video_path = out_dir / f"{path.stem}.mp4"
                    writer = cv2.VideoWriter(str(video_path), cv2.VideoWriter_fourcc(*"mp4v"),
                                             fps, size)
                    if not writer.isOpened():
                        log(f"Cannot write {video_path}; --save-video ignored")
                        writer = None
            frame = (source_frame if source_frame.shape[1::-1] == size
                     else cv2.resize(source_frame, size, interpolation=cv2.INTER_AREA))

            handle_ocr(ocr.release(now))

            results = model.track(frame, imgsz=imgsz, device=live.device, classes=[5],
                                  conf=args.conf, persist=True, verbose=False,
                                  tracker="bytetrack.yaml",
                                  **({"half": True} if live.half else {}))
            buses = []
            for result in results:
                if result.boxes is None:
                    continue
                for box in result.boxes:
                    if box.id is None:
                        continue
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    buses.append((x1, y1, x2, y2, int(box.id[0])))

            current = {}
            annotations = []
            for x1, y1, x2, y2, track_id in buses:
                width, height = x2 - x1, y2 - y1
                if width <= 0 or height <= 0:
                    continue
                aspect_ratio = width / height
                last_seen[track_id] = frame_id
                last_box[track_id] = Box(x1, y1, x2, y2)
                angled[track_id] = aspect_ratio > live.ASPECT_RATIO_MAX
                if not angled[track_id]:
                    current[track_id] = last_box[track_id]

                _, roi_box = live.extract_sign_roi(frame, x1, y1, x2, y2)
                roi = live.crop_source_roi(source_frame, roi_box, size)
                if track_id in confirmed:
                    annotations.append(((x1, y1, x2, y2), roi_box,
                                        f"ID:{track_id} -> {confirmed[track_id]}",
                                        (0, 255, 0), track_id))
                    continue
                label = f"ID:{track_id} [OCR]"
                if not live.is_valid_roi(roi):
                    label = f"ID:{track_id} [ROI TOO SMALL]"
                elif aspect_ratio < live.ASPECT_RATIO_MIN:
                    label = f"ID:{track_id} [NARROW]"
                elif aspect_ratio > live.ASPECT_RATIO_MAX:
                    label = f"ID:{track_id} [ANGLE FILTER]"
                elif frame_id % live.OCR_FRAME_INTERVAL == 0:
                    ocr.schedule(track_id, roi, now)
                annotations.append(((x1, y1, x2, y2), roi_box, label, (0, 255, 255), track_id))

            if frame_id % live.CLEANUP_INTERVAL_FRAMES == 0:
                for track_id in [t for t, seen in last_seen.items()
                                 if frame_id - seen > live.FORGET_FRAMES
                                 and t not in ocr.pending_ids]:
                    for table in (last_seen, last_box, angled, confirmed):
                        table.pop(track_id, None)

            visible = {track_id: last_box[track_id] for track_id, seen in last_seen.items()
                       if frame_id - seen <= live.GRACE_FRAMES and not angled[track_id]}
            handle_records(core.step(now, visible))
            order_stats.update(frame, frame_id, now, current)

            if writer is not None:
                rank = {t: i + 1 for i, t in enumerate(
                    order_by_position(current, args.announce_order))} if len(current) > 1 else {}
                writer.write(draw_frame(frame, annotations,
                                        banner if now < banner_until else "", now, rank))

            wall = time.monotonic()
            if wall - last_progress >= PROGRESS_INTERVAL:
                last_progress = wall
                done = f"{frame_id}/{total_frames}" if total_frames else str(frame_id)
                print(f"  ... {path.name}: frame {done} ({video_clock(now)}), "
                      f"{frame_id / (wall - wall_start):.1f} frames/s")

        # 影片結束：還沒回來的 OCR 與還沒播的確認照樣處理（車子視為停在最後一幀的位置）
        handle_ocr(ocr.release(float("inf")))
        for _ in range(100):
            flush = core.flush_time()
            if flush is None:
                break
            now = max(now, flush)
            handle_records(core.step(now, visible))
    except KeyboardInterrupt:
        log("Interrupted; writing what has been simulated so far")
        raise
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        announce_file.close()
        order_stats.close()
        with open(out_dir / f"{path.stem}.srt", "w", encoding="utf-8") as srt:
            for index, (start, end, text) in enumerate(subtitles, 1):
                srt.write(f"{index}\n{srt_clock(start)} --> {srt_clock(end)}\n{text}\n\n")
        lines = [
            f"Video: {path}",
            f"Frames: {frame_id} ({video_clock(frame_id / fps)}), {fps:g} FPS",
            f"Tracked bus IDs: {len(set(last_seen) | set(confirmed))} still in memory, "
            f"confirmed: {len(confirmed)}",
            f"Announcements: {counts['ANNOUNCE']} (with front / rear: {counts['positioned']}, "
            f"2+ buses but area disagreed: {counts['unpositioned_multi']}), "
            f"skipped: {counts['SKIP']}, dropped: {counts['DROP']}",
            *order_stats.summary(),
        ]
        (out_dir / "summary.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        log("--- summary")
        for line in lines:
            log(line)
        log(f"Output: {out_dir.resolve()}")
        log.close()
    return counts


def expand_videos(items) -> list[Path]:
    """檔案、資料夾（裡面的影片）或萬用字元（Windows 的命令列不會自己展開 *.mp4）。"""
    videos = []
    for item in items:
        path = Path(item)
        if any(ch in item for ch in "*?["):
            anchor = Path(path.anchor or ".")
            pattern = str(path.relative_to(anchor)) if path.anchor else item
            videos += sorted(p for p in anchor.glob(pattern) if p.is_file())
        elif path.is_dir():
            videos += sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
        else:
            videos.append(path)
    return videos


def main():
    sys.stdout.reconfigure(line_buffering=True)
    args = parse_args()
    if args.device is not None:
        live.device = args.device
        live.half = args.device == "cuda"     # 與 nohead 相同
    model_path = live.resolve_model_path(args.model)
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    cache = OcrCache(None if args.no_cache or args.ocr != "ollama"
                     else out_root / "ocr_cache.json")
    ocr = SimulatedOcr(args, cache)
    ocr.warmup()

    videos = expand_videos(args.videos)
    if not videos:
        print("No video files found")
        return
    try:
        for video in videos:
            if not video.is_file():
                print(f"Not a file: {video}; skipped")
                continue
            simulate_video(video, args, model_path, ocr, out_root)
            cache.save()
    except KeyboardInterrupt:
        print("Ctrl+C, exiting")
    finally:
        cache.save()
        stats = ocr.stats
        print(f"OCR requests: {stats['sent']} sent, {stats['cached']} from cache, "
              f"{stats['errors']} failed, {stats['full']} skipped (queue full)")


if __name__ == "__main__":
    main()
