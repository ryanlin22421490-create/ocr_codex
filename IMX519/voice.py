"""公車進站中文語音播報：OCR 確認號碼後，用 piper 合成並從預設喇叭播放，處理排隊進站。

nohead_onetime_main.py 使用：add_arguments / check_args 處理參數，建立
VoiceAnnouncer(args, tracks_fn) 後在 CONFIRMED 時呼叫 announce(track_id, number)，結束時呼叫 close()。
tracks_fn() 由主程式提供，回傳 {track_id: Track(present, box, gone, first_box, age)}：
present 是否還在畫面中（GRACE_FRAMES 內出現過），box 為最後的框 (x1, y1, x2, y2)，
gone 為最後一次被看到到現在幾秒，first_box 為第一次出現時的框，age 為第一次出現到現在幾秒。

排隊進站（VOICE_ANNOUNCE_PLAN.md「排隊進站的播報邏輯」）：
A. announce()（OCR worker 呼叫，只做判斷、不等播報）：
   - 同一個 track ID 只播一次。
   - 同一個號碼要再播時，拿新車「剛出現時」的框和前一台同號碼「最後」的框比：
     新車出現之後前一台還被看到（兩框同時存在）→ 位置不近：真的是第二台，照常播報；位置相近：同一台被重複框到，丟棄。
     前一台在新車出現前就沒再被看到 → 間隔不到 --voice-switch-window 秒且位置相近：ByteTrack 換了 ID，丟棄；
     否則是新的一班車，照常播報。
     用「剛出現時」的框：確認要等 OCR（約 0.7 s），進站中的車這段時間可能已移動超過半個車身；
     用「同時存在」而不是 present：換 ID 時舊 ID 還會在寬限期內算 present 約 0.9 s。
B. 背景執行緒：收到第一筆後再等 --voice-merge 秒合併；播報前確認車還在（離開的印 skipped (left)）；
   依 --voice-front 排出前後（預設 right：畫面右邊的車排前面）；句子用「前車 / 後車」說出前後關係
   （同號碼的第二台也一樣，不加「也是 / 另一台」：使用者 2026-11-17 試聽後決定，那種說法較不清楚）；
   合併的多台車分句播：「前車12號公車即將進站」播完再播「後車5號公車即將進站」，中間不插入其他播報；
   整句播完才播下一句，不會互相打斷。OCR 只有一個 worker，兩台車的確認至少相隔一次 OCR（實測 0.7 s 左右）。

播放：常駐的 piper 子程序（--json-input，模型只載入一次，約 0.4 s），每句寫一行 JSON 給 piper，
piper 在暫存資料夾寫出 wav 並把路徑印到 stdout，再用 paplay 播放（走 PulseAudio 預設輸出），播完刪除 wav。
啟動時先合成一句不播放的暖機句子（WARMUP_TEXT），第一台車的播報就不用等第一次推論。

數字唸法（2026-11-17 實測 piper zh_CN-huayan-medium）：
1～2 位數直接交給 piper，會照整數唸（63 → 六十三）；3～4 位數先轉成國字逐字唸
（358 → 三五八、1050 → 一零五零；使用者 2026-11-17 決定，三位數 piper 本來會唸成三百五十八，
4 位數則會唸錯：1050 → 一千五十，少了「零」）。--voice-digits 時 1～2 位數也逐字唸。

piper 用官方 aarch64 單獨執行檔（~/piper/piper/），不用 pip 的 piper-tts：
pip 版會裝一般的 onnxruntime，跟 Jetson 的 onnxruntime-gpu 衝突。
任何一步失敗都只印警告並停用語音，不影響辨識主程式。

單獨執行：python3 voice.py 5 300 1050（每個號碼當成一台車、每 3 秒確認一台，試聽用）
模擬：    python3 voice.py --simulate（不接相機、不出聲，驗證排隊播報規則；加 --play 實際播出）
"""
import argparse
import collections
import json
import math
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time

# ---- sentences -----------------------------------------------------------
# {bus} 例如 "5號"；{pos} 為 POS_FRONT / POS_BACK。同號碼的第二台也用同樣的說法。
# 合併的多台車一台一句、前車先播（使用者 2026-11-17 試聽後決定，不合成一句）。
VOICE_ONE = "{bus}公車即將進站"              # 畫面中沒有其他車（或分不出前後）：5號公車即將進站
VOICE_POS_ONE = "{pos}{bus}公車即將進站"     # 有其他車：前車12號公車即將進站 / 後車5號公車即將進站
POS_FRONT = "前車"
POS_BACK = "後車"

DEFAULT_PIPER_BIN = "~/piper/piper/piper"
DEFAULT_PIPER_MODEL = "~/piper/voices/zh_CN-huayan-medium.onnx"
PLAYER = ["paplay"]            # 走 PulseAudio 預設輸出（目前是 USB 耳機）
QUEUE_SIZE = 4                 # 待播佇列上限；滿了就丟掉新的
DEFAULT_MERGE = 0.8            # 合併視窗秒數
DEFAULT_SWITCH_WINDOW = 5.0    # 前一台同號碼消失幾秒內、位置相近，視為 ByteTrack 換 ID
NEAR_RATIO = 0.5               # 框中心距離小於前一個框寬度 × 此值，視為相近
FRONT_CHOICES = ("right", "left", "bottom")
DEFAULT_FRONT = "right"        # flip 2 後的畫面：公車從左方進入，前車在右邊
SPEED_RANGE = (0.5, 2.0)       # --voice-speed 允許範圍（piper length_scale，越大越慢）
DEFAULT_SPEED = 1.25           # 比 piper 原速慢 25%（使用者 2026-11-17 試聽 1.0 / 1.25 / 1.5 後選定）
MERGE_MAX = 3.0
PIPER_TIMEOUT = 10.0           # 單句合成逾時（實測約 0.2～0.3 s）
PLAY_TIMEOUT = 15.0            # 單句播放逾時（實測一句約 2～3 s）
CHINESE_DIGITS = "零一二三四五六七八九"
DIGITWISE_MIN_LEN = 3          # 幾位數以上逐字唸（358 → 三五八）；以下照整數唸（63 → 六十三）
# 啟動時先合成一次、不播放：第一次推論較慢（實測 0.6～0.8 s，之後約 0.25 s），
# 先暖機讓第一台車的播報不用多等。句型與實際播報相同，數字涵蓋 0～9。
WARMUP_TEXT = VOICE_POS_ONE.format(pos=POS_FRONT, bus=CHINESE_DIGITS + "號")

NORMAL = "normal"
ANOTHER = "another"            # 同號碼、前一台還在畫面中而且位置不同：真的是第二台

Track = collections.namedtuple("Track", "present box gone first_box age")


def log(message: str):
    print(f"[{time.strftime('%H:%M:%S')}] [VOICE] {message}", flush=True)


# ---- text ----------------------------------------------------------------
def spoken_number(number: str, digits: bool = False) -> str:
    """把 OCR 號碼轉成要給 piper 的文字（不含「號」）。"""
    number = number.strip()
    if not re.fullmatch(r"\d+", number):
        return number
    if digits or len(number) >= DIGITWISE_MIN_LEN:
        return "".join(CHINESE_DIGITS[int(d)] for d in number)
    return str(int(number))    # 去掉開頭的 0，例如 "05" → "5"


def front_key(box, front: str) -> float:
    """越大越靠前（離站牌越近）。"""
    if front == "right":
        return (box[0] + box[2]) / 2
    if front == "left":
        return -(box[0] + box[2]) / 2
    return box[3]              # bottom：框底越低越靠前


def is_near(box, prev_box) -> bool:
    """新框中心與前一個框中心的距離小於前一個框寬度的一半。"""
    dx = (box[0] + box[2] - prev_box[0] - prev_box[2]) / 2
    dy = (box[1] + box[3] - prev_box[1] - prev_box[3]) / 2
    return math.hypot(dx, dy) < (prev_box[2] - prev_box[0]) * NEAR_RATIO


def compose(items, other_boxes, front: str, digits: bool = False) -> list[str]:
    """回傳要依序播的句子。items：[(number, box)]，已依前後排好（前車在前）；
    other_boxes：畫面中其他車的框（只在單台時用來判斷前後）。"""
    if len(items) == 1:
        number, box = items[0]
        bus = f"{spoken_number(number, digits)}號"
        if other_boxes:
            key = front_key(box, front)
            others = [front_key(b, front) for b in other_boxes]
            if all(key > k for k in others):
                return [VOICE_POS_ONE.format(pos=POS_FRONT, bus=bus)]
            if all(key < k for k in others):
                return [VOICE_POS_ONE.format(pos=POS_BACK, bus=bus)]
        return [VOICE_ONE.format(bus=bus)]  # 沒有其他車，或夾在中間分不出前後

    sentences = []
    for i, (number, _box) in enumerate(items):
        bus = f"{spoken_number(number, digits)}號"
        pos = POS_FRONT if i == 0 else POS_BACK if i == len(items) - 1 else None
        sentences.append(VOICE_POS_ONE.format(pos=pos, bus=bus) if pos else VOICE_ONE.format(bus=bus))
    return sentences


# ---- arguments -----------------------------------------------------------
def add_arguments(parser):
    group = parser.add_argument_group("voice announcement")
    group.add_argument("--no-voice", action="store_true",
                       help="Disable voice announcement")
    group.add_argument("--voice-merge", type=float, default=DEFAULT_MERGE,
                       help="Seconds to wait after a confirmation for more buses to announce "
                            f"in one sentence; 0 = one sentence per bus (default: {DEFAULT_MERGE:g})")
    group.add_argument("--voice-switch-window", type=float, default=DEFAULT_SWITCH_WINDOW,
                       help="A bus number confirmed again at a nearby position within this many "
                            "seconds after the previous bus with that number was last seen counts "
                            f"as a ByteTrack ID switch and is not announced (default: {DEFAULT_SWITCH_WINDOW:g})")
    group.add_argument("--voice-front", choices=FRONT_CHOICES, default=DEFAULT_FRONT,
                       help="Where the front bus (nearest the stop) is in the picture "
                            f"(default: {DEFAULT_FRONT})")
    group.add_argument("--voice-digits", action="store_true",
                       help="Read every bus number digit by digit "
                            "(numbers with 3 or more digits are always read digit by digit)")
    group.add_argument("--voice-speed", type=float, default=DEFAULT_SPEED,
                       help="piper length_scale; larger is slower "
                            f"({SPEED_RANGE[0]:g}-{SPEED_RANGE[1]:g}, default: {DEFAULT_SPEED:g})")
    group.add_argument("--piper-bin", default=DEFAULT_PIPER_BIN,
                       help=f"piper executable (default: {DEFAULT_PIPER_BIN})")
    group.add_argument("--piper-model", default=DEFAULT_PIPER_MODEL,
                       help=f"piper voice model (default: {DEFAULT_PIPER_MODEL})")


def check_args(args):
    """Validate ranges and expand ~ in paths; exits through SystemExit on bad values."""
    if not 0 <= args.voice_merge <= MERGE_MAX:
        raise SystemExit(f"--voice-merge must be in 0-{MERGE_MAX:g}")
    if args.voice_switch_window < 0:
        raise SystemExit("--voice-switch-window must be >= 0")
    if not SPEED_RANGE[0] <= args.voice_speed <= SPEED_RANGE[1]:
        raise SystemExit(f"--voice-speed must be in {SPEED_RANGE[0]:g}-{SPEED_RANGE[1]:g}")
    args.piper_bin = os.path.expanduser(args.piper_bin)
    args.piper_model = os.path.expanduser(args.piper_model)


# ---- speakers ------------------------------------------------------------
class PiperSpeaker:
    """常駐 piper + paplay。start() 失敗時回傳 False（已印警告）。"""

    def __init__(self, args):
        self.piper_bin = args.piper_bin
        self.piper_model = args.piper_model
        self.speed = args.voice_speed
        self.piper = None
        self.player = None
        self.tmpdir = None
        self.lock = threading.Lock()   # 保護「開新 player」與 stop() 結束子程序之間的競爭
        self.stopping = threading.Event()

    def start(self) -> bool:
        if not os.path.isfile(self.piper_bin) or not os.access(self.piper_bin, os.X_OK):
            log(f"WARNING: piper not found: {self.piper_bin}; voice disabled")
            return False
        if not os.path.isfile(self.piper_model):
            log(f"WARNING: piper model not found: {self.piper_model}; voice disabled")
            return False
        if shutil.which(PLAYER[0]) is None:
            log(f"WARNING: {PLAYER[0]} not found; voice disabled")
            return False
        self.tmpdir = tempfile.mkdtemp(prefix="voice_")
        try:
            # start_new_session：終端機 Ctrl+C 會送 SIGINT 給整個程序群組，子程序若一起收到，
            # 會在 close() 之前先結束，背景執行緒就可能接著開下一句的 paplay 而留下孤兒程序；
            # 讓子程序不在同一群組，一律由 stop() 結束。
            self.piper = subprocess.Popen(
                [self.piper_bin, "-m", self.piper_model, "--json-input",
                 "-d", self.tmpdir, "-q", "--length_scale", f"{self.speed:g}"],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                text=True, encoding="utf-8", bufsize=1, start_new_session=True)
        except OSError as error:
            log(f"WARNING: cannot start piper: {error}; voice disabled")
            return False
        try:
            started = time.monotonic()
            os.remove(self._synthesize(WARMUP_TEXT))
        except (OSError, RuntimeError, ValueError) as error:
            if not self.stopping.is_set():
                log(f"WARNING: piper warmup error: {error}; voice disabled")
            return False
        log(f"ready (piper warmup {time.monotonic() - started:.2f} s)")
        return True

    def _synthesize(self, text: str) -> str:
        """回傳 wav 路徑；失敗時丟 RuntimeError。"""
        if self.piper.poll() is not None:
            raise RuntimeError(f"piper exited with code {self.piper.returncode}")
        self.piper.stdin.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
        self.piper.stdin.flush()
        # piper 寫完 wav 才印路徑；readline 沒有逾時，用計時執行緒在卡住時結束 piper
        timed_out = threading.Event()

        def kill():
            timed_out.set()
            self.piper.kill()

        timer = threading.Timer(PIPER_TIMEOUT, kill)
        timer.start()
        try:
            path = self.piper.stdout.readline().strip()
        finally:
            timer.cancel()
        if not path or not os.path.isfile(path):
            if timed_out.is_set():
                raise RuntimeError(f"piper timeout ({PIPER_TIMEOUT:g} s)")
            try:
                code = self.piper.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                code = "still running"
            raise RuntimeError(f"piper produced no audio (exit code {code})")
        return path

    def say(self, text: str) -> bool:
        """合成並播放；piper 失敗時回傳 False（呼叫端停用語音）。"""
        try:
            started = time.monotonic()
            path = self._synthesize(text)
        except (OSError, RuntimeError, ValueError) as error:
            if not self.stopping.is_set():
                log(f"WARNING: piper error: {error}; voice disabled")
            return False
        log(f"{text}  (synth {time.monotonic() - started:.2f} s)")
        try:
            return self._play(path)
        finally:
            try:
                os.remove(path)
            except OSError:
                pass

    def _play(self, path: str) -> bool:
        with self.lock:
            if self.stopping.is_set():
                return True
            try:
                self.player = subprocess.Popen(PLAYER + [path], stdout=subprocess.DEVNULL,
                                               stderr=subprocess.PIPE, text=True,
                                               start_new_session=True)
            except OSError as error:
                log(f"WARNING: cannot run {PLAYER[0]}: {error}; voice disabled")
                return False
        try:
            _, err = self.player.communicate(timeout=PLAY_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.player.kill()
            self.player.communicate()
            log(f"WARNING: {PLAYER[0]} timeout")
            return True
        if self.player.returncode != 0 and not self.stopping.is_set():
            # 播放失敗只警告、不停用：PulseAudio 暫時不在（例如耳機重新連線）時下一則可能就好了
            log(f"WARNING: {PLAYER[0]} failed ({self.player.returncode}): {err.strip()}")
        return True

    def stop(self):
        """中斷正在播的句子並結束 piper（任何執行緒都可呼叫）。"""
        with self.lock:
            self.stopping.set()
            for proc in (self.player, self.piper):
                if proc is not None and proc.poll() is None:
                    proc.terminate()

    def cleanup(self):
        """背景執行緒結束時呼叫。"""
        if self.piper is not None and self.piper.poll() is None:
            try:
                self.piper.stdin.close()
            except OSError:
                pass
            try:
                self.piper.wait(timeout=1.0)
            except subprocess.TimeoutExpired:
                self.piper.kill()
        if self.tmpdir is not None:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


class RecordingSpeaker:
    """模擬用：不出聲，記下每句的文字與開始 / 結束時間，每句佔 duration 秒。"""

    def __init__(self, duration: float = 1.0):
        self.duration = duration
        self.spoken = []           # (text, start, end)
        self.stopping = threading.Event()

    def start(self) -> bool:
        return True

    def say(self, text: str) -> bool:
        log(text)
        start = time.monotonic()
        self.stopping.wait(self.duration)
        self.spoken.append((text, start, time.monotonic()))
        return True

    def stop(self):
        self.stopping.set()

    def cleanup(self):
        pass


# ---- announcer -----------------------------------------------------------
class VoiceAnnouncer:
    """排隊進站播報器；announce() 不會卡住呼叫的執行緒。"""

    def __init__(self, args, tracks_fn, speaker=None):
        self.enabled = not args.no_voice
        self.tracks_fn = tracks_fn
        self.merge = args.voice_merge
        self.switch_window = args.voice_switch_window
        self.front = args.voice_front
        self.digits = args.voice_digits
        self.speaker = speaker if speaker is not None else PiperSpeaker(args)
        self.queue = queue.Queue(maxsize=QUEUE_SIZE)
        self.announced_ids = set()         # 已處理過的 track ID（播報、排隊或判定換 ID）
        self.last_by_number = {}           # 號碼 -> 最後一台這個號碼的 track ID（只記會播或已播的）
        self.successor = {}                # 判定換 ID 時：舊 track ID -> 新 track ID
        self.stats = collections.Counter()  # left / id_switch / queue_full / spoken（模擬檢查用）
        self.lock = threading.Lock()
        self.stopping = threading.Event()
        self.thread = None
        if self.enabled:
            self.thread = threading.Thread(target=self._run, name="voice", daemon=True)
            self.thread.start()

    # -- A：確認號碼時的判斷 -----------------------------------------------
    def _classify(self, track_id, number, tracks):
        """回傳 NORMAL / ANOTHER；判定為換 ID 時回傳 None。"""
        prev_id = self.last_by_number.get(number)
        if prev_id is None:
            return NORMAL
        prev, me = tracks.get(prev_id), tracks.get(track_id)
        if prev is None or me is None:
            return NORMAL          # 前一台已被主程式遺忘（約 18 s 沒出現），當成新的一班車
        near = is_near(me.first_box, prev.box)
        if prev.gone <= me.age:    # 新車出現之後前一台還被看到：兩個框同時存在
            return None if near else ANOTHER
        gap = prev.gone - me.age   # 前一台最後被看到 → 新車第一次出現
        if near and gap <= self.switch_window:
            return None
        return NORMAL

    def announce(self, track_id: int, number: str) -> bool:
        """確認號碼時呼叫；排入佇列時回傳 True。"""
        if not self.enabled or self.stopping.is_set():
            return False
        number = str(number).strip()
        tracks = self.tracks_fn()
        with self.lock:
            if track_id in self.announced_ids:
                return False
            self.announced_ids.add(track_id)
            kind = self._classify(track_id, number, tracks)
            prev_id = self.last_by_number.get(number)
            self.last_by_number[number] = track_id
            if kind is None:
                # 舊 ID 若還在佇列裡、播報前就消失了，播報執行緒會改看新 ID
                self.successor[prev_id] = track_id
                self.stats["id_switch"] += 1
                log(f"id switch skip {number} [ID:{prev_id} -> ID:{track_id}]")
                return False
            try:
                self.queue.put_nowait((track_id, number, kind))
            except queue.Full:
                self._forget(number, track_id)
                self.stats["queue_full"] += 1
                log(f"queue full, dropped {number} [ID:{track_id}]")
                return False
        return True

    def _chain(self, track_id):
        """track_id 以及之後因換 ID 接手的 ID（呼叫端持有 self.lock）。"""
        chain = [track_id]
        while chain[-1] in self.successor and len(chain) < 10:
            chain.append(self.successor[chain[-1]])
        return chain

    def _forget(self, number, track_id):
        """這台車沒播出：之後同號碼的車不能再被當成「已播過的車換了 ID」（呼叫端持有 self.lock）。"""
        if self.last_by_number.get(number) in self._chain(track_id):
            del self.last_by_number[number]

    def wait_idle(self, timeout: float | None = None) -> bool:
        """等佇列播完（單獨執行 / 模擬用）；背景執行緒結束時也會返回。"""
        deadline = None if timeout is None else time.monotonic() + timeout
        while self.queue.unfinished_tasks and self.thread is not None and self.thread.is_alive():
            if deadline is not None and time.monotonic() > deadline:
                return False
            time.sleep(0.05)
        return True

    def close(self):
        """停止播報：中斷正在播的句子、結束 piper，不等佇列播完。"""
        if self.thread is None:
            return
        self.stopping.set()
        self.speaker.stop()
        try:
            self.queue.put_nowait(None)
        except queue.Full:
            pass
        self.thread.join(timeout=2.0)
        self.thread = None

    # -- B：播報執行緒 -----------------------------------------------------
    def _collect(self, first):
        """收到第一筆後再等 merge 秒；回傳 (batch, 是否收到結束訊號)。"""
        batch = [first]
        deadline = time.monotonic() + self.merge
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0 or self.stopping.is_set():
                return batch, self.stopping.is_set()
            try:
                item = self.queue.get(timeout=remaining)
            except queue.Empty:
                return batch, False
            if item is None:
                self.queue.task_done()
                return batch, True
            batch.append(item)

    def _speak_batch(self, batch) -> bool:
        tracks = self.tracks_fn()
        alive = []
        batch_ids = set()
        with self.lock:
            for track_id, number, kind in batch:
                # 排隊期間 ByteTrack 換了 ID：舊 ID 不在了就看接手的新 ID
                chain = self._chain(track_id)
                batch_ids.update(chain)
                # 優先用最新接手的 ID：舊 ID 可能只是還在寬限期內，框已經過時
                track = next((tracks[tid] for tid in reversed(chain)
                              if tid in tracks and tracks[tid].present), None)
                if track is None:
                    self._forget(number, track_id)
                    self.stats["left"] += 1
                    log(f"skipped (left) {number} [ID:{track_id}]")
                else:
                    alive.append((number, track.box))
        if not alive:
            return True
        alive.sort(key=lambda item: front_key(item[1], self.front), reverse=True)
        # 其他車：不在這批裡、而且不在這批任何一台附近（附近的框多半是同一台換了 ID、還沒確認）
        other_boxes = [t.box for tid, t in tracks.items()
                       if t.present and tid not in batch_ids
                       and not any(is_near(t.box, box) or is_near(box, t.box) for _, box in alive)]
        # 一句播完才播下一句；這批播完前不會取佇列，其他車的播報不會插進來
        for text in compose(alive, other_boxes, self.front, self.digits):
            if self.stopping.is_set():
                break
            self.stats["spoken"] += 1
            if not self.speaker.say(text):
                return False
        return True

    def _run(self):
        ok = self.speaker.start()
        try:
            while True:
                first = self.queue.get()
                if first is None or self.stopping.is_set():
                    self.queue.task_done()
                    break
                batch, stop = self._collect(first)
                try:
                    if ok and self.enabled and not self.stopping.is_set():
                        if not self._speak_batch(batch):
                            self.enabled = False
                finally:
                    for _ in batch:
                        self.queue.task_done()
                if stop:
                    break
        finally:
            self.speaker.cleanup()


# ---- standalone test / simulation -----------------------------------------
class FakeTracks:
    """模擬主程式的 tracks_fn。show：出現 / 移動到 cx；hide：離開（立刻不算在畫面中）；
    lose：追蹤中斷（不再被看到，但和主程式一樣在寬限期 LOSE_GRACE 秒內仍算在畫面中）。"""

    LOSE_GRACE = 0.9               # 主程式 GRACE_FRAMES 15 幀，17 FPS 約 0.9 s

    def __init__(self):
        self.lock = threading.Lock()
        self.boxes = {}            # track_id -> 最後的框
        self.first = {}            # track_id -> (第一次出現的框, 時間)
        self.lost = {}             # track_id -> (不再被看到的時間, 寬限秒數)

    @staticmethod
    def box_at(cx: float, width: float = 400, y2: float = 650):
        return (cx - width / 2, y2 - width * 0.6, cx + width / 2, y2)

    def show(self, track_id, cx, width=400):
        with self.lock:
            box = self.box_at(cx, width)
            self.boxes[track_id] = box
            self.first.setdefault(track_id, (box, time.monotonic()))
            self.lost.pop(track_id, None)

    def hide(self, track_id):
        with self.lock:
            self.lost[track_id] = (time.monotonic(), 0.0)

    def lose(self, track_id):
        with self.lock:
            self.lost[track_id] = (time.monotonic(), self.LOSE_GRACE)

    def __call__(self):
        now = time.monotonic()
        with self.lock:
            tracks = {}
            for tid, box in self.boxes.items():
                lost_at, grace = self.lost.get(tid, (now, None))
                gone = now - lost_at
                present = grace is None or gone <= grace
                first_box, first_time = self.first[tid]
                tracks[tid] = Track(present, box, gone, first_box, now - first_time)
            return tracks


# 每個情境：(名稱, 事件 [(秒, 動作, 參數...)], 預期句子, 預期統計)
# 畫面 1280 寬，--voice-front right：x 越大越前面。
SCENARIOS = [
    ("單台車",
     [(0, "show", 1, 640), (0, "confirm", 1, "5")],
     ["5號公車即將進站"], {}),
    ("兩台同時確認（前車後確認）",
     [(0, "show", 1, 350), (0, "show", 2, 900), (0, "confirm", 1, "5"), (0.3, "confirm", 2, "12")],
     ["前車12號公車即將進站", "後車5號公車即將進站"], {}),
    ("兩台相隔較久",
     [(0, "show", 1, 900), (0, "confirm", 1, "5"), (3, "show", 2, 350), (3, "confirm", 2, "12")],
     ["5號公車即將進站", "後車12號公車即將進站"], {}),
    ("連續兩台同號碼",
     [(0, "show", 1, 900), (0, "confirm", 1, "5"), (3, "show", 2, 350), (3, "confirm", 2, "5")],
     ["5號公車即將進站", "後車5號公車即將進站"], {}),
    ("兩台同號碼同時確認",
     [(0, "show", 1, 900), (0, "show", 2, 350), (0, "confirm", 1, "5"), (0.3, "confirm", 2, "5")],
     ["前車5號公車即將進站", "後車5號公車即將進站"], {}),
    ("ByteTrack 換 ID（舊 ID 已消失）",
     [(0, "show", 1, 640), (0, "confirm", 1, "5"), (1, "hide", 1), (1.5, "show", 3, 660),
      (3, "confirm", 3, "5")],
     ["5號公車即將進站"], {"id_switch": 1}),
    ("ByteTrack 換 ID（舊 ID 仍在寬限期內）",
     [(0, "show", 1, 640), (0, "confirm", 1, "5"), (1.9, "lose", 1), (2, "show", 3, 650),
      (2.5, "confirm", 3, "5")],
     ["5號公車即將進站"], {"id_switch": 1}),
    ("ByteTrack 換 ID，車子邊開邊確認（確認時已移動超過半個車身）",
     [(0, "show", 1, 300), (0, "confirm", 1, "5"), (1.9, "lose", 1), (2, "show", 3, 320),
      (2.3, "show", 3, 450), (2.6, "show", 3, 600), (2.7, "confirm", 3, "5")],
     ["5號公車即將進站"], {"id_switch": 1}),
    ("同一台車被重複框到（兩個框同時存在、位置相近）",
     [(0, "show", 1, 640), (0, "confirm", 1, "5"), (2, "show", 3, 660), (2.5, "confirm", 3, "5")],
     ["5號公車即將進站"], {"id_switch": 1}),
    ("單台車等播報時換 ID（新 ID 還沒確認），不能說成後車",
     [(0, "show", 1, 300), (0, "confirm", 1, "5"), (0.3, "lose", 1), (0.4, "show", 2, 380)],
     ["5號公車即將進站"], {}),
    ("同號碼的下一班車（消失超過 5 秒）",
     [(0, "show", 1, 640), (0, "confirm", 1, "5"), (1, "hide", 1), (7, "show", 4, 640),
      (7, "confirm", 4, "5")],
     ["5號公車即將進站", "5號公車即將進站"], {}),
    ("車已經開走",
     [(0, "show", 1, 640), (0, "confirm", 1, "5"), (0.3, "hide", 1)],
     [], {"left": 1}),
    ("播報中有新的確認",
     [(0, "show", 1, 900), (0, "confirm", 1, "5"), (1.2, "show", 2, 350), (1.2, "confirm", 2, "12")],
     ["5號公車即將進站", "後車12號公車即將進站"], {}),
    ("沒播到就被擋住，換 ID 重新出現",
     [(0, "show", 1, 640), (0, "confirm", 1, "5"), (0.3, "hide", 1), (1.5, "show", 3, 660),
      (2, "confirm", 3, "5")],
     ["5號公車即將進站"], {"left": 1, "id_switch": 0}),
    ("排隊等播報時換 ID",
     [(0, "show", 1, 900), (0, "confirm", 1, "12"), (1.0, "show", 2, 640), (1.0, "confirm", 2, "5"),
      (1.2, "show", 3, 650), (1.3, "confirm", 3, "5"), (1.5, "hide", 2)],
     ["12號公車即將進站", "後車5號公車即將進站"], {"id_switch": 1, "left": 0}),
    ("佇列滿了",
     [(0, "show", 1, 1100), (0, "confirm", 1, "5")]
     + [(1.2, "show", i, 1000 - i * 120, 100) for i in range(2, 7)]
     + [(1.2 + i * 0.01, "confirm", i, str(10 + i)) for i in range(2, 7)],
     ["5號公車即將進站", "前車12號公車即將進站", "13號公車即將進站", "14號公車即將進站",
      "後車15號公車即將進站"], {"queue_full": 1}),
]


def run_scenario(name, events, expected, expected_stats, args, play: bool) -> bool:
    tracks = FakeTracks()
    speaker = PiperSpeaker(args) if play else RecordingSpeaker()
    spoken = []
    if play:
        orig_say = speaker.say

        def say(text):
            start = time.monotonic()
            result = orig_say(text)
            spoken.append((text, start, time.monotonic()))
            return result
        speaker.say = say
    else:
        spoken = speaker.spoken
    announcer = VoiceAnnouncer(args, tracks, speaker)
    if play:
        time.sleep(2.0)    # 等 piper 載入與暖機（約 1.3 s），避免暖機時間吃掉合併視窗
    print(f"\n=== {name}")
    t0 = time.monotonic()
    for at, action, *params in sorted(events, key=lambda e: e[0]):
        delay = t0 + at - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        if action == "show":
            tracks.show(*params)
        elif action == "hide":
            tracks.hide(*params)
        elif action == "lose":
            tracks.lose(*params)
        elif action == "confirm":
            announcer.announce(*params)
    time.sleep(args.voice_merge + 0.2)
    announcer.wait_idle(60)
    announcer.close()

    texts = [text for text, _, _ in spoken]
    overlap = any(spoken[i][1] < spoken[i - 1][2] - 0.01 for i in range(1, len(spoken)))
    stats_ok = all(announcer.stats[k] == v for k, v in expected_stats.items())
    passed = texts == expected and stats_ok and not overlap
    print(f"  expected: {expected}  {dict(expected_stats) or ''}")
    print(f"  actual:   {texts}  {dict((k, announcer.stats[k]) for k in expected_stats) or ''}"
          + ("  OVERLAP" if overlap else ""))
    print(f"  {'PASS' if passed else 'FAIL'}")
    return passed


def simulate(args, play: bool):
    results = [run_scenario(*scenario, args, play) for scenario in SCENARIOS]
    print(f"\n{sum(results)}/{len(results)} scenarios passed")
    return all(results)


def main():
    parser = argparse.ArgumentParser(
        description="Announce bus numbers in Mandarin. "
                    "python3 voice.py 5 300 1050: each number is a separate bus, one every 3 s. "
                    "python3 voice.py --simulate: check the queued-arrival rules without sound.")
    add_arguments(parser)
    parser.add_argument("--simulate", action="store_true",
                        help="Run the built-in queued-arrival scenarios (no camera, no sound)")
    parser.add_argument("--play", action="store_true",
                        help="With --simulate: actually speak through piper")
    parser.add_argument("numbers", nargs="*", help="bus numbers to announce, in order")
    args = parser.parse_args()
    check_args(args)
    if args.simulate:
        sys.exit(0 if simulate(args, args.play) else 1)
    if not args.numbers:
        parser.error("give bus numbers, or --simulate")

    # 每個號碼是不同的車，每 3 秒確認一台；全部停在同一個位置、不離開，所以分不出前後，
    # 每句都是單台車的說法。同一個號碼重複時，前一台還在、位置相同，會被當成 ByteTrack 換 ID 而不播。
    tracks = FakeTracks()
    announcer = VoiceAnnouncer(args, tracks)
    try:
        for i, number in enumerate(args.numbers, start=1):
            if i > 1:
                time.sleep(3.0)
            tracks.show(i, 640)
            announcer.announce(i, number)
        time.sleep(args.voice_merge + 0.2)
        announcer.wait_idle()
    except KeyboardInterrupt:
        pass
    finally:
        announcer.close()


if __name__ == "__main__":
    main()
