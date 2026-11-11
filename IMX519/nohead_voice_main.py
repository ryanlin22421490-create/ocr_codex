"""nohead_onetime_main.py 加上語音播報的版本。

辨識流程完全沿用 nohead_onetime_main.py（直接匯入它，不複製程式碼，那邊的修正這裡自動跟上），
只多做一件事：號碼一確認（CONFIRMED）就馬上用語音唸出來。nohead 的所有參數與按鍵照常可用。

    python3 IMX519/nohead_voice_main.py                       # 與 nohead 相同，另加語音
    python3 IMX519/nohead_voice_main.py --voice-test 307,262  # 只測喇叭與語音，不開相機
    python3 IMX519/nohead_voice_main.py --voice-prepare 307,262,18   # 先合成常見路線的句子後結束

語音引擎（--tts，預設 auto：有 Piper 用 Piper，否則 espeak-ng）：
    piper    離線神經網路語音，自然；只用 CPU，不佔 GPU。
             pip3 install piper-tts
             python3 -m piper.download_voices zh_CN-huayan-medium --data-dir IMX519/voice
    espeak   espeak-ng，安裝最簡單但聲音機械。 sudo apt install espeak-ng
播放一律用 aplay（sudo apt install alsa-utils），--audio-device 指定喇叭（例如 plughw:1,0）。

播報規則在 announcer.py：
- 確認後立刻播（--announce-window 預設 0）；正在播上一句時排隊，播完接著播。
- 畫面上只有一輛：「三零七號公車進站」；有兩輛：依框下緣判斷「前方 / 後方」，
  框面積排序不一致時不說前後。三輛以上：「第一輛 / 第二輛…」。
- 同號碼 60 秒內不重播（同一輛車被擋後換了追蹤 ID）；兩輛同號車同時在畫面則都播。
- 確認後超過 10 秒還沒輪到，或車已離開畫面：不播。
合成過的句子存在 --voice-cache（預設 ~/.cache/bus_voice），同一句第二次起直接播放。
"""
import argparse
import hashlib
import os
import shutil
import subprocess
import sys
import threading
import time
import wave
from pathlib import Path
from queue import Empty, Queue

sys.path.insert(0, str(Path(__file__).resolve().parent))

import announcer
from announcer import AnnouncerCore, Box, Event

SCRIPT_DIR = Path(__file__).resolve().parent
VOICE_DIR = SCRIPT_DIR / "voice"            # Piper 模型（*.onnx 與 *.onnx.json）預設位置
VISIBLE_SECONDS = 1.0                       # 這段時間內偵測到的車算「在畫面上」（約 nohead 的 GRACE_FRAMES）
BOX_FORGET_SECONDS = 30.0
PLAY_TIMEOUT = 30.0
ESPEAK_VOICE = "cmn"                        # espeak-ng 的國語
ESPEAK_SPEED = 150                          # 每分鐘字數


# =========================
# 參數（語音專用；其餘原樣交給 nohead）
# =========================
def parse_voice_args(argv):
    parser = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    group = parser.add_argument_group("voice options (nohead_voice_main.py)")
    group.add_argument("--no-voice", action="store_true", help="Run without voice (same as nohead)")
    group.add_argument("--tts", choices=("auto", "piper", "espeak"), default="auto",
                       help="Speech engine. Default: auto (Piper if available, else espeak-ng)")
    group.add_argument("--piper-model", default=None,
                       help="Piper voice .onnx; default: the first *.onnx in IMX519/voice")
    group.add_argument("--audio-device", default=None,
                       help="aplay -D device, e.g. plughw:1,0; default: system default")
    group.add_argument("--voice-cache", default=str(Path.home() / ".cache" / "bus_voice"),
                       help="Folder for synthesized sentences. Default: ~/.cache/bus_voice")
    group.add_argument("--voice-test", default=None, metavar="NUMBERS",
                       help='Speak a test sentence and exit, e.g. "307" or "307,262"')
    group.add_argument("--voice-prepare", default=None, metavar="NUMBERS",
                       help="Synthesize the sentences for these route numbers into the cache "
                            'and exit, e.g. "307,262,18"')
    group.add_argument("--announce-window", type=float, default=0.0,
                       help="Seconds to wait for more numbers before speaking. Default: 0 (at once)")
    group.add_argument("--announce-cooldown", type=float, default=announcer.DEFAULT_COOLDOWN,
                       help=f"Seconds before the same number is spoken again. Default: {announcer.DEFAULT_COOLDOWN:g}")
    group.add_argument("--announce-max-age", type=float, default=announcer.DEFAULT_MAX_AGE,
                       help=f"Drop a number not spoken within this many seconds. Default: {announcer.DEFAULT_MAX_AGE:g}")
    group.add_argument("--announce-order", choices=announcer.ORDERS, default=announcer.DEFAULT_ORDER,
                       help="Front bus: bottom (head-on camera), left / right (side camera). Default: bottom")
    group.add_argument("--announce-min-height", type=int, default=0,
                       help="Boxes lower than this (px) do not count as a bus in the queue. Default: 0")
    group.add_argument("--no-cross-check", action="store_true",
                       help="Say front / rear even when the box-area order disagrees")
    group.add_argument("--number-style", choices=announcer.NUMBER_STYLES,
                       default=announcer.DEFAULT_NUMBER_STYLE,
                       help="mixed: 1-2 digits as a value, 3-4 digit by digit. Default: mixed")
    if "-h" in argv or "--help" in argv:
        parser.print_help()
        print("\nAll nohead_onetime_main.py options also apply:\n")
        return None, argv                   # nohead 接著印自己的說明並結束
    args, rest = parser.parse_known_args(argv)
    for name in ("announce_window", "announce_cooldown", "announce_max_age"):
        if getattr(args, name) < 0:
            parser.error(f"--{name.replace('_', '-')} must be >= 0")
    return args, rest


def parse_numbers(text):
    numbers = [part.strip() for part in text.split(",") if part.strip()]
    if not numbers or not all(n.isdigit() and len(n) <= 4 for n in numbers):
        raise SystemExit(f'Route numbers must be 1 to 4 digits separated by commas, got "{text}"')
    return numbers


# =========================
# 語音合成與播放
# =========================
class PiperEngine:
    """Piper：模型載入一次後留在記憶體，短句合成約零點幾秒（CPU）。"""

    def __init__(self, model_path: Path):
        from piper import PiperVoice
        self.name = f"piper-{model_path.stem}"
        self.voice = PiperVoice.load(str(model_path))

    def synthesize(self, text: str, path: Path):
        with wave.open(str(path), "wb") as wav_file:
            if hasattr(self.voice, "synthesize_wav"):       # piper-tts 1.3 以後
                self.voice.synthesize_wav(text, wav_file)
            else:                                           # 舊版 API
                self.voice.synthesize(text, wav_file)


class EspeakEngine:
    def __init__(self, command: str):
        self.name = f"espeak-{ESPEAK_VOICE}-{ESPEAK_SPEED}"
        self.command = command

    def synthesize(self, text: str, path: Path):
        subprocess.run([self.command, "-v", ESPEAK_VOICE, "-s", str(ESPEAK_SPEED),
                        "-w", str(path), text], check=True, timeout=30,
                       stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)


def find_piper_model(requested):
    if requested:
        path = Path(requested).expanduser()
        return path if path.is_file() else None
    models = sorted(VOICE_DIR.glob("*.onnx")) if VOICE_DIR.is_dir() else []
    return models[0] if models else None


def create_engine(args):
    """回傳 (引擎, None) 或 (None, 不能用的原因)。"""
    problems = []
    if args.tts in ("auto", "piper"):
        model = find_piper_model(args.piper_model)
        if model is None:
            problems.append(f"Piper: no voice model ({args.piper_model or VOICE_DIR / '*.onnx'})")
        else:
            try:
                return PiperEngine(model), None
            except Exception as error:      # 沒裝 piper-tts、模型或設定檔壞掉
                problems.append(f"Piper: {error}")
    if args.tts in ("auto", "espeak"):
        command = shutil.which("espeak-ng") or shutil.which("espeak")
        if command:
            return EspeakEngine(command), None
        problems.append("espeak-ng: not installed (sudo apt install espeak-ng)")
    return None, "; ".join(problems)


class Speaker:
    """文字 -> WAV（有快取）-> aplay。speak() 會等到播完。"""

    def __init__(self, engine, cache_dir, audio_device=None):
        self.engine = engine
        self.cache_dir = Path(cache_dir).expanduser()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.audio_device = audio_device
        self.player = shutil.which("aplay")
        self.process = None
        self.closed = False

    def wav_for(self, text: str) -> tuple[Path, bool]:
        """(WAV 路徑, 是否來自快取)。"""
        key = hashlib.sha1(f"{self.engine.name}\n{text}".encode("utf-8")).hexdigest()[:16]
        path = self.cache_dir / f"{key}.wav"
        if path.is_file() and path.stat().st_size > 44:
            return path, True
        partial = path.with_suffix(".part")
        self.engine.synthesize(text, partial)
        os.replace(partial, path)           # 合成到一半被中斷時不會留下壞檔
        return path, False

    def speak(self, text: str) -> str:
        """回傳一段給紀錄用的說明；失敗時丟出例外。"""
        started = time.monotonic()
        path, cached = self.wav_for(text)
        synth = time.monotonic() - started
        if self.player is None:
            raise RuntimeError("aplay not found (sudo apt install alsa-utils)")
        command = [self.player, "-q"]
        if self.audio_device:
            command += ["-D", self.audio_device]
        if self.closed:
            return "cancelled"
        self.process = subprocess.Popen(command + [str(path)], stdout=subprocess.DEVNULL,
                                        stderr=subprocess.PIPE)
        try:
            _, stderr = self.process.communicate(timeout=PLAY_TIMEOUT)
        except subprocess.TimeoutExpired:
            self.process.kill()
            self.process.communicate()
            raise RuntimeError(f"aplay did not finish within {PLAY_TIMEOUT:g} s")
        if self.process.returncode != 0 and not self.closed:
            raise RuntimeError("aplay failed: " + stderr.decode(errors="replace").strip())
        source = "cached" if cached else f"synthesized in {synth:.2f} s"
        return f"{source}, played {time.monotonic() - started - synth:.1f} s"

    def close(self):
        """結束程式時不等待：中斷正在播放的句子。"""
        self.closed = True
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()


# =========================
# 播報執行緒
# =========================
class VoiceAnnouncer:
    """OCR 執行緒呼叫 notify()，主迴圈呼叫 update_boxes()；播報在自己的執行緒，不拖慢辨識。"""

    def __init__(self, speaker, args, clock=time.monotonic):
        self.speaker = speaker
        self.clock = clock
        self.core = AnnouncerCore(
            window=args.announce_window, cooldown=args.announce_cooldown,
            max_age=args.announce_max_age, order=args.announce_order,
            min_height=args.announce_min_height, cross_check=not args.no_cross_check,
            number_style=args.number_style,
        )
        self.queue = Queue()
        self.boxes = {}                     # track_id -> (Box, 最後看到的時間, 是否側面)
        self.lock = threading.Lock()
        self.stopped = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="voice-announcer")
        self.thread.start()

    def notify(self, track_id: int, number: str):
        self.queue.put(Event(track_id, number, self.clock()))

    def update_boxes(self, buses, aspect_ratio_max: float):
        """buses：nohead.detect_bus 的回傳值 [(x1, y1, x2, y2, conf, track_id)]。"""
        now = self.clock()
        with self.lock:
            for x1, y1, x2, y2, _, track_id in buses:
                width, height = x2 - x1, y2 - y1
                if width > 0 and height > 0:
                    self.boxes[track_id] = (Box(x1, y1, x2, y2), now,
                                            width / height > aspect_ratio_max)
            for track_id in [t for t, (_, seen, _) in self.boxes.items()
                             if now - seen > BOX_FORGET_SECONDS]:
                del self.boxes[track_id]

    def visible(self) -> dict:
        now = self.clock()
        with self.lock:
            return {track_id: box for track_id, (box, seen, angled) in self.boxes.items()
                    if now - seen <= VISIBLE_SECONDS and not angled}

    def _run(self):
        while not self.stopped.is_set():
            try:
                # 還有排隊的確認時不久等：上一句播完就接著處理
                self.core.notify(self.queue.get(timeout=0.02 if self.core.pending else 0.2))
                while True:
                    self.core.notify(self.queue.get_nowait())
            except Empty:
                pass
            for record in self.core.step(self.clock(), self.visible()):
                stamp = time.strftime("%H:%M:%S")
                numbers = " ".join(e.number for e in record.events)
                ids = " ".join(str(e.track_id) for e in record.events)
                if record.kind != "ANNOUNCE":
                    print(f"[{stamp}] ANNOUNCE {record.kind} {numbers} (ID {ids}): {record.reason}")
                    continue
                try:
                    detail = self.speaker.speak(record.speech)
                except Exception as error:
                    detail = f"FAILED: {error}"
                # 實際播完才算空閒（record.duration 只是估計值）
                self.core.mark_busy_until(self.clock())
                print(f"[{stamp}] ANNOUNCE {record.text} (ID {ids}) | {detail}")

    def close(self):
        self.stopped.set()
        self.speaker.close()
        self.thread.join(timeout=2.0)


# =========================
# 不開相機的工具模式
# =========================
def run_voice_test(speaker, args):
    numbers = parse_numbers(args.voice_test)
    ranked = list(enumerate(numbers))
    text, speech = announcer.build_sentence(ranked, len(numbers), len(numbers) > 1,
                                            args.number_style)
    print(f"Speaking: {text}  ({speech})")
    print("Result:", speaker.speak(speech))


def run_voice_prepare(speaker, args):
    """每個號碼先合成「單獨 / 前方 / 後方」三句，之後第一次播報也不用等合成。"""
    for number in parse_numbers(args.voice_prepare):
        for rank, count in ((0, 1), (0, 2), (1, 2)):
            text, speech = announcer.build_sentence([(rank, number)], count, count > 1,
                                                    args.number_style)
            _, cached = speaker.wav_for(speech)
            print(f"{'cached     ' if cached else 'synthesized'}  {text}")
    print(f"Cache: {speaker.cache_dir}")


# =========================
# 主程式：掛上 nohead 的兩個函式後執行 nohead.main()
# =========================
def main():
    args, rest = parse_voice_args(sys.argv[1:])
    sys.argv = [sys.argv[0]] + rest         # 剩下的參數交給 nohead 解析

    speaker = None
    if args is not None and not args.no_voice:
        engine, problem = create_engine(args)
        if engine is None:
            print(f"Voice disabled: {problem}")
        else:
            speaker = Speaker(engine, args.voice_cache, args.audio_device)
            print(f"Voice: {engine.name}, cache {speaker.cache_dir}"
                  + ("" if speaker.player else " | aplay NOT FOUND: nothing will be heard"))

    if args is not None and (args.voice_test or args.voice_prepare):
        if speaker is None:
            raise SystemExit("No speech engine available")
        if args.voice_prepare:
            run_voice_prepare(speaker, args)
        if args.voice_test:
            run_voice_test(speaker, args)
        return

    import nohead_onetime_main as live      # 這裡才載入 torch / YOLO（語音測試不需要）

    voice = VoiceAnnouncer(speaker, args) if speaker is not None else None
    if voice is not None:
        original_detect, original_ocr = live.detect_bus, live.process_ocr

        def detect_bus(frame, conf):
            buses = original_detect(frame, conf)
            voice.update_boxes(buses, live.ASPECT_RATIO_MAX)
            return buses

        def process_ocr(track_id, roi_img):
            original_ocr(track_id, roi_img)
            # process_ocr 只會收到尚未確認的 ID：做完後已在 confirmed_buses，就是這次剛確認的
            with live.lock:
                number = live.confirmed_buses.get(track_id)
            if number:
                voice.notify(track_id, number)

        live.detect_bus, live.process_ocr = detect_bus, process_ocr

    try:
        live.main()
    finally:
        if voice is not None:
            voice.close()


if __name__ == "__main__":
    main()
