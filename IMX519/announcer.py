"""公車號碼語音播報（espeak-ng，離線）。nohead_onetime_tts_main.py 使用。

用法：
    announcer = Announcer()
    announcer.observe(track_id, box_height, now)   # 主迴圈：每幀、每台偵測到的公車
    announcer.say(track_id, "307")                 # OCR 執行緒：號碼確認時
    announcer.close()                              # 程式結束

say() 只把號碼放進佇列就返回；合成與播放都在自己的執行緒，不會卡住相機、YOLO、OCR。

播報規則（都在「播報那一刻」用當下畫面判斷）：
  * 句型「三零七號公車進站」，號碼逐字念，每句念一遍，不打斷正在播的句子。
  * 第一個號碼確認後等 MERGE_WINDOW 秒，期間確認的號碼合併成一句，
    一句最多 MAX_PER_SENTENCE 個，框高的（近的、前車）先念，其餘留到下一句。
  * 畫面上有另一台公車的框高 >= 自己的 REAR_HEIGHT_RATIO 倍：自己是後車，
    念「後方三零七號」。框高取最近 HEIGHT_FRAMES 幀平均；前車開走後就不加「後方」。
  * 每個追蹤 ID 只播一次。同一號碼 COOLDOWN_SECONDS 內不重播（ByteTrack 跟丟後
    同一台車會換新 ID 再確認一次），除非上次播報的那台車現在仍在畫面上
    （CONCURRENT_SECONDS 內還有偵測到）：那是同路線的另一台車，照常播報。
  * 輪到播報時車已離開畫面（VISIBLE_SECONDS 沒偵測到）就不播。

聲音輸出：espeak-ng 先合成 WAV，再用 paplay（PulseAudio；藍牙耳機走這裡）播放，
沒有 paplay 時用 aplay。藍牙耳機閒置後重新出聲會吃掉開頭約零點幾秒，
所以每句前面補 LEAD_SILENCE 秒靜音，號碼的第一個字才不會被切掉。

espeak-ng 1.51 把中文字念成英文（「三」念成 "san one"）；1.50 與 1.52 之後正常。
啟動時先試念一個字，不正常就改送拼音給 cmn-latn-pinyin 語音（PINYIN 表），發音相同。
要念新的字時記得把拼音補進 PINYIN。

單獨執行可以測試喇叭 / 耳機：
    python3 announcer.py 307          # 「三零七號公車進站」
    python3 announcer.py 307 12       # 「三零七號，一二號公車進站」
"""
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from pathlib import Path
from queue import Empty, Full, Queue

VOICE = "cmn"                 # espeak-ng 的國語語音
PINYIN_VOICE = "cmn-latn-pinyin"   # 中文字念不對的版本（1.51）改用：輸入拼音 + 聲調數字
SPEED_WPM = 180               # espeak-ng -s；每個字之間有停頓，「三零七號公車進站」約 2.7 秒
LEAD_SILENCE = 0.4            # 每句前補的靜音秒數（藍牙耳機喚醒）
SYNTH_TIMEOUT = 15.0
PLAY_TIMEOUT = 20.0

MERGE_WINDOW = 0.5            # 第一個號碼確認後，等多久收其他號碼合併成一句
MAX_PER_SENTENCE = 2
COOLDOWN_SECONDS = 30.0       # 同一號碼多久內不重播
VISIBLE_SECONDS = 1.0         # 多久內有偵測到算「還在畫面上」（17 FPS 約 15 幀）
CONCURRENT_SECONDS = 0.5      # 上次播同號碼的那台車，多久內有偵測到算「和新車同時在畫面上」
REAR_HEIGHT_RATIO = 1.2       # 別台車的框高 >= 自己的幾倍，自己算後車；換架設位置要重新量
HEIGHT_FRAMES = 5             # 框高取最近幾幀平均
FORGET_SECONDS = 60.0         # 這麼久沒出現的 ID 清掉（ByteTrack 不會重用 ID）
QUEUE_SIZE = 8

DIGITS = "零一二三四五六七八九"
STARTUP_TEXT = "語音播報已啟動"
# 會念到的每一個字（拼音備援用）
PINYIN = {
    "零": "ling2", "一": "yi1", "二": "er4", "三": "san1", "四": "si4",
    "五": "wu3", "六": "liu4", "七": "qi1", "八": "ba1", "九": "jiu3",
    "號": "hao4", "後": "hou4", "方": "fang1",
    "公": "gong1", "車": "che1", "進": "jin4", "站": "zhan4",
    "語": "yu3", "音": "yin1", "播": "bo1", "報": "bao4",
    "已": "yi3", "啟": "qi3", "動": "dong4",
    "，": ",",
}


def number_text(number: str) -> str:
    """ "307" -> "三零七"（逐字念）。"""
    return "".join(DIGITS[int(ch)] if ch.isdigit() else ch for ch in number)


def sentence(items) -> str:
    """items: [(number, is_rear), ...] -> 「三零七號，後方一二號公車進站」。"""
    parts = [("後方" if rear else "") + number_text(number) + "號" for number, rear in items]
    return "，".join(parts) + "公車進站"


def to_pinyin(text: str) -> str:
    return " ".join(PINYIN.get(ch, ch) for ch in text)


def hanzi_works() -> bool:
    """這一版 espeak-ng 能不能直接念中文字（1.51 會夾英文音素，輸出帶 "(en)"）。"""
    try:
        result = subprocess.run(["espeak-ng", "-v", VOICE, "-q", "-x", "三號"],
                                capture_output=True, text=True, timeout=SYNTH_TIMEOUT)
    except (OSError, subprocess.SubprocessError):
        return True
    return result.returncode != 0 or "(en)" not in result.stdout


def find_player(player: str | None = None) -> list[str] | None:
    """播放 WAV 的指令（最後接檔名）。player 可指定，例如 "aplay -D plughw:1,0"。"""
    if player:
        return shlex.split(player)
    if shutil.which("paplay"):
        return ["paplay"]
    if shutil.which("aplay"):
        return ["aplay", "-q"]
    return None


def add_lead_silence(path: Path, seconds: float) -> None:
    if seconds <= 0:
        return
    with wave.open(str(path), "rb") as source:
        params = source.getparams()
        frames = source.readframes(source.getnframes())
    silence = bytes(int(params.framerate * seconds) * params.sampwidth * params.nchannels)
    with wave.open(str(path), "wb") as target:
        target.setparams(params)
        target.writeframes(silence + frames)


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}")


class Announcer:
    def __init__(self, player: str | None = None, speak=None, clock=time.monotonic):
        """speak / clock 可換掉（測試用）；speak(text) 預設用 espeak-ng 合成並播放。"""
        self._clock = clock
        self._lock = threading.Lock()
        self._heights = {}        # track_id -> deque(最近幾幀框高)
        self._seen = {}           # track_id -> 最後偵測到的時間
        self._announced = set()   # 處理過（播過或因冷卻跳過）的 track_id
        self._last_spoken = {}    # number -> (播報時間, track_id)
        self._queue = Queue(maxsize=QUEUE_SIZE)
        self._stop = threading.Event()
        self._process = None
        self._tmpdir = None
        self._wav = None
        self._pinyin = False

        self._speak = speak
        if speak is None:
            self._player = find_player(player)
            if shutil.which("espeak-ng") is None:
                print("[TTS] 找不到 espeak-ng，不會有聲音。安裝：sudo apt install espeak-ng",
                      file=sys.stderr)
            elif self._player is None or shutil.which(self._player[0]) is None:
                print("[TTS] 找不到播放程式（paplay / aplay），不會有聲音。"
                      "安裝：sudo apt install pulseaudio-utils alsa-utils", file=sys.stderr)
            else:
                self._tmpdir = tempfile.mkdtemp(prefix="bus_tts_")
                self._wav = Path(self._tmpdir) / "say.wav"
                self._speak = self._espeak
                self._pinyin = not hanzi_works()
                voice = f"{PINYIN_VOICE}, pinyin input" if self._pinyin else VOICE
                print(f"[TTS] espeak-ng ({voice}) -> {' '.join(self._player)}")

        self._thread = threading.Thread(target=self._run, daemon=True, name="tts-announcer")
        self._thread.start()

    # ---- called from the main loop / OCR worker ---------------------------
    def observe(self, track_id: int, height: float, now: float) -> None:
        """主迴圈每幀回報一台公車的框高（處理畫面像素）。"""
        with self._lock:
            heights = self._heights.get(track_id)
            if heights is None:
                heights = self._heights[track_id] = deque(maxlen=HEIGHT_FRAMES)
            heights.append(height)
            self._seen[track_id] = now

    def say(self, track_id: int, number: str) -> None:
        """號碼確認時呼叫；立刻返回。"""
        try:
            self._queue.put_nowait((track_id, number))
        except Full:
            log(f"TTS SKIP [ID:{track_id}] {number}: queue full")

    def say_text(self, text: str) -> None:
        """直接念一句（開機提示用），不經過去重。"""
        try:
            self._queue.put_nowait(text)
        except Full:
            pass

    def close(self) -> None:
        self._stop.set()
        try:
            self._queue.put_nowait(None)
        except Full:
            pass
        process = self._process
        if process is not None and process.poll() is None:
            process.terminate()
        self._thread.join(timeout=2.0)
        if self._tmpdir:
            shutil.rmtree(self._tmpdir, ignore_errors=True)

    # ---- announcer thread --------------------------------------------------
    def _run(self) -> None:
        pending = []
        while not self._stop.is_set():
            if pending:
                # 上一句念不完的號碼：直接接著處理，順便收播放期間新確認的
                wait_until = None
            else:
                item = self._queue.get()
                if item is None:
                    return
                if isinstance(item, str):
                    self._output(item, item)
                    continue
                pending.append(item)
                wait_until = time.monotonic() + MERGE_WINDOW
            while True:
                try:
                    if wait_until is None:
                        item = self._queue.get_nowait()
                    else:
                        item = self._queue.get(timeout=max(0.0, wait_until - time.monotonic()))
                except Empty:
                    break
                if item is None:
                    return
                if isinstance(item, str):
                    continue
                pending.append(item)

            items = self.select(pending, self._clock())
            if items:
                label = ", ".join(("後方 " if rear else "") + number for number, rear in items)
                self._output(label, sentence(items))

    def select(self, pending: list, now: float) -> list:
        """從 pending 選出這一句要念的 [(number, is_rear), ...]；pending 只留下一句再處理的。"""
        with self._lock:
            self._forget(now)
            scene = {
                track_id: sum(heights) / len(heights)
                for track_id, heights in self._heights.items()
                if now - self._seen[track_id] <= VISIBLE_SECONDS
            }
            candidates = []
            for track_id, number in pending:
                if track_id in self._announced or any(track_id == c[0] for c in candidates):
                    continue
                if track_id not in scene:
                    self._announced.add(track_id)
                    log(f"TTS SKIP [ID:{track_id}] {number}: left the frame")
                    continue
                candidates.append((track_id, number))
            # 框高的（前車）先念；sort 穩定，一樣高時維持確認順序
            candidates.sort(key=lambda c: -scene[c[0]])

            items, leftover = [], []
            for track_id, number in candidates:
                if len(items) >= MAX_PER_SENTENCE:
                    leftover.append((track_id, number))
                    continue
                self._announced.add(track_id)
                last = self._last_spoken.get(number)
                if last is not None and now - last[0] < COOLDOWN_SECONDS:
                    previous_id = last[1]
                    previous_seen = self._seen.get(previous_id)
                    same_bus = (previous_seen is None
                                or now - previous_seen > CONCURRENT_SECONDS)
                    if same_bus:
                        log(f"TTS SKIP [ID:{track_id}] {number}: announced "
                            f"{now - last[0]:.0f} s ago (ID:{previous_id})")
                        continue
                rear = any(height >= REAR_HEIGHT_RATIO * scene[track_id]
                           for other_id, height in scene.items() if other_id != track_id)
                self._last_spoken[number] = (now, track_id)
                items.append((number, rear))
            pending[:] = leftover
            return items

    def _forget(self, now: float) -> None:
        stale = [track_id for track_id, seen in self._seen.items()
                 if now - seen > FORGET_SECONDS]
        for track_id in stale:
            del self._seen[track_id]
            del self._heights[track_id]
            self._announced.discard(track_id)
        for number in [n for n, (spoken, _) in self._last_spoken.items()
                       if now - spoken > COOLDOWN_SECONDS]:
            del self._last_spoken[number]

    def _output(self, label: str, text: str) -> None:
        log(f"TTS: {label} -> 「{text}」" if label != text else f"TTS: 「{text}」")
        if self._speak is None:
            return
        try:
            self._speak(text)
        except (OSError, subprocess.SubprocessError, wave.Error, EOFError) as error:
            log(f"TTS error: {error}")

    def _espeak(self, text: str) -> None:
        voice, spoken = (PINYIN_VOICE, to_pinyin(text)) if self._pinyin else (VOICE, text)
        subprocess.run(
            ["espeak-ng", "-v", voice, "-s", str(SPEED_WPM), "-w", str(self._wav), spoken],
            check=True, timeout=SYNTH_TIMEOUT,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        add_lead_silence(self._wav, LEAD_SILENCE)
        if self._stop.is_set():
            return
        self._process = subprocess.Popen(self._player + [str(self._wav)],
                                         stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        try:
            _, errors = self._process.communicate(timeout=PLAY_TIMEOUT)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.communicate()
            raise
        if self._process.returncode and not self._stop.is_set():
            raise OSError(f"{self._player[0]} failed: {errors.decode(errors='replace').strip()}")


def main() -> None:
    numbers = sys.argv[1:] or ["307"]
    announcer = Announcer()
    for index, number in enumerate(numbers):
        announcer.say(index, number)
    # 假裝每台車都一直在畫面上，後面的框比較小（示範合併與「後方」）
    sentences = -(-len(numbers) // MAX_PER_SENTENCE)
    deadline = time.monotonic() + MERGE_WINDOW + 4.0 * sentences
    while time.monotonic() < deadline:
        for index in range(len(numbers)):
            announcer.observe(index, 400 / (1 + index), time.monotonic())
        time.sleep(0.1)
    announcer.close()


if __name__ == "__main__":
    main()
