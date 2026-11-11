"""公車號碼語音播報的判斷邏輯（不含聲音輸出）。

announce_sim.py（錄影模擬）與之後的即時程式共用這個模組，規則只寫一份。
時間一律由呼叫端傳入（錄影模擬用影片時間，即時程式用 time.monotonic()），
所以同一套規則在模擬與實際執行時的結果一致。

流程：
    OCR 確認號碼 -> notify(Event)
    呼叫端定期 step(now, visible) -> 回傳本次的紀錄（ANNOUNCE / SKIP / DROP）
    回傳 ANNOUNCE 時，呼叫端播放後呼叫 mark_busy_until(播完的時間)

step 的規則：
    1. 播報中不處理新的一組。
    2. 第一筆確認進來後等 window 秒，期間其他確認併成同一句。
    3. 每筆確認：車已離開畫面 -> DROP；確認後超過 max_age 秒 -> DROP；
       同號碼在 cooldown 秒內播過，且上次那輛已離開或就是同一輛 -> SKIP
       （同一輛車被擋後換了追蹤 ID）；上次那輛還在畫面、這次是另一輛 -> 照播（兩輛同號）。
    4. 依框的位置排前後（迎面拍：框下緣越低越前面），再用框面積交叉確認；
       兩者不一致時不說前後，避免播錯位置。
"""
from dataclasses import dataclass, field
from functools import cmp_to_key
from typing import NamedTuple

DEFAULT_WINDOW = 2.0          # 秒：第一筆確認後等多久，收集同一句的其他號碼
DEFAULT_COOLDOWN = 60.0       # 秒：同號碼不重播的時間
DEFAULT_MAX_AGE = 10.0        # 秒：確認後多久沒播到就丟掉（車可能已開走）
DEFAULT_ORDER = "bottom"      # 前方判斷：bottom 迎面拍，left / right 側面拍（車往左 / 右開）
DEFAULT_TIE_PX = 10           # 主要座標相差不到這個值（px）時，改比框高
DEFAULT_NUMBER_STYLE = "mixed"
ORDERS = ("bottom", "left", "right")
NUMBER_STYLES = ("mixed", "digits", "value")

DIGIT_WORDS = "零一二三四五六七八九"
ORDINAL_WORDS = "一二三四五六七八九十"
SUFFIX_TEXT = "號公車進站"

# 播報長度估計（只用在模擬：決定下一句最早何時能開始、字幕顯示多久）
SECONDS_PER_CHAR = 0.27
SECONDS_PER_PAUSE = 0.3
SECONDS_LEAD = 0.3


class Box(NamedTuple):
    """處理畫面座標的公車框。"""
    x1: int
    y1: int
    x2: int
    y2: int

    @property
    def width(self):
        return self.x2 - self.x1

    @property
    def height(self):
        return self.y2 - self.y1

    @property
    def area(self):
        return self.width * self.height


@dataclass
class Event:
    """一筆號碼確認。"""
    track_id: int
    number: str
    time: float


@dataclass
class Record:
    """step() 的輸出。kind 為 ANNOUNCE / SKIP / DROP。"""
    kind: str
    time: float
    text: str = ""            # ANNOUNCE：顯示用句子（阿拉伯數字）
    speech: str = ""          # ANNOUNCE：實際要唸的中文
    duration: float = 0.0     # ANNOUNCE：估計播放秒數
    events: list = field(default_factory=list)
    reason: str = ""          # SKIP / DROP 的原因
    visible_count: int = 0    # ANNOUNCE：當下畫面上參與排序的車數
    order: list = field(default_factory=list)       # ANNOUNCE：前到後的追蹤 ID
    consistent: bool = True   # ANNOUNCE：位置排序與面積排序是否一致
    positioned: bool = False  # ANNOUNCE：句子有沒有說前後


# ---- 號碼唸法 --------------------------------------------------------------
def value_words(number: int) -> str:
    """0～9999 的數值唸法：12 -> 十二，307 -> 三百零七，1010 -> 一千零一十。"""
    if not 0 <= number <= 9999:
        raise ValueError("number must be 0 to 9999")
    if number == 0:
        return DIGIT_WORDS[0]
    words = []
    pending_zero = False
    for digit, unit in zip(f"{number:04d}", ("千", "百", "十", "")):
        value = int(digit)
        if value == 0:
            pending_zero = bool(words)
            continue
        if pending_zero:
            words.append(DIGIT_WORDS[0])
            pending_zero = False
        words.append(DIGIT_WORDS[value] + unit)
    text = "".join(words)
    return text[1:] if 10 <= number < 20 else text   # 一十二 -> 十二


def number_words(number: str, style: str = DEFAULT_NUMBER_STYLE) -> str:
    """號碼的中文唸法。

    mixed：1～2 位數唸數值（十二），3～4 位數逐字唸（三零七）。
    digits：全部逐字唸。value：全部唸數值（三百零七）。
    開頭是 0 的號碼（例如 05）一律逐字唸。
    """
    if not number.isdigit():
        raise ValueError(f"number must be digits, got {number!r}")
    if style not in NUMBER_STYLES:
        raise ValueError(f"unknown number style {style!r}")
    by_digit = "".join(DIGIT_WORDS[int(c)] for c in number)
    if number.startswith("0") and len(number) > 1:
        return by_digit
    if style == "digits" or (style == "mixed" and len(number) > 2):
        return by_digit
    return value_words(int(number))


# ---- 前後排序 --------------------------------------------------------------
def _primary(box: Box, order: str) -> int:
    """越大越前面。"""
    if order == "bottom":
        return box.y2
    if order == "left":
        return -box.x1
    if order == "right":
        return box.x2
    raise ValueError(f"unknown order {order!r}")


def order_by_position(boxes: dict, order: str = DEFAULT_ORDER,
                      tie_px: int = DEFAULT_TIE_PX) -> list:
    """追蹤 ID 由前到後。主要座標相差不到 tie_px 時比框高，再相同則比 ID（較早出現的在前）。"""
    def compare(a, b):
        pa, pb = _primary(boxes[a], order), _primary(boxes[b], order)
        if abs(pa - pb) >= tie_px:
            return -1 if pa > pb else 1
        ha, hb = boxes[a].height, boxes[b].height
        if ha != hb:
            return -1 if ha > hb else 1
        return -1 if a < b else (1 if a > b else 0)
    return sorted(boxes, key=cmp_to_key(compare))


def order_by_area(boxes: dict) -> list:
    """追蹤 ID 依框面積由大到小（面積相同比 ID）。"""
    return sorted(boxes, key=lambda track_id: (-boxes[track_id].area, track_id))


def order_buses(boxes: dict, order: str = DEFAULT_ORDER,
                tie_px: int = DEFAULT_TIE_PX) -> tuple[list, bool]:
    """(由前到後的 ID, 位置排序與面積排序是否一致)。"""
    by_position = order_by_position(boxes, order, tie_px)
    return by_position, by_position == order_by_area(boxes)


def position_word(rank: int, count: int) -> str:
    """rank 從 0 起算；count 為畫面上的車數。"""
    if count <= 1:
        return ""
    if count == 2:
        return ("前方", "後方")[rank]
    word = ORDINAL_WORDS[rank] if rank < len(ORDINAL_WORDS) else str(rank + 1)
    return f"第{word}輛"


def build_sentence(ranked: list, count: int, positioned: bool,
                   style: str = DEFAULT_NUMBER_STYLE) -> tuple[str, str]:
    """ranked：[(rank, number)]，已由前到後排好。回傳 (顯示文字, 唸的中文)。

    positioned=True：「前方 307，後方 262 號公車進站」
    positioned=False：「307、262 號公車進站」
    """
    if positioned:
        text_parts, speech_parts = [], []
        for rank, number in ranked:
            word = position_word(rank, count)
            text_parts.append(f"{word} {number}" if word else number)
            speech_parts.append(word + number_words(number, style))
        return "，".join(text_parts) + " " + SUFFIX_TEXT, "，".join(speech_parts) + SUFFIX_TEXT
    numbers = [number for _, number in ranked]
    return ("、".join(numbers) + " " + SUFFIX_TEXT,
            "、".join(number_words(n, style) for n in numbers) + SUFFIX_TEXT)


def estimate_duration(speech: str) -> float:
    pauses = speech.count("，") + speech.count("、")
    chars = len(speech) - pauses
    return SECONDS_LEAD + chars * SECONDS_PER_CHAR + pauses * SECONDS_PER_PAUSE


# ---- 播報狀態 --------------------------------------------------------------
class AnnouncerCore:
    def __init__(self, window=DEFAULT_WINDOW, cooldown=DEFAULT_COOLDOWN,
                 max_age=DEFAULT_MAX_AGE, order=DEFAULT_ORDER, tie_px=DEFAULT_TIE_PX,
                 min_height=0, cross_check=True, number_style=DEFAULT_NUMBER_STYLE):
        if order not in ORDERS:
            raise ValueError(f"unknown order {order!r}")
        if number_style not in NUMBER_STYLES:
            raise ValueError(f"unknown number style {number_style!r}")
        self.window = window
        self.cooldown = cooldown
        self.max_age = max_age
        self.order = order
        self.tie_px = tie_px
        self.min_height = min_height
        self.cross_check = cross_check
        self.number_style = number_style
        self.pending = []            # 尚未處理的 Event，依確認時間排列
        self.busy_until = float("-inf")
        self.last_announced = {}     # number -> (time, track_id)

    def notify(self, event: Event) -> None:
        self.pending.append(event)
        self.pending.sort(key=lambda e: e.time)

    def mark_busy_until(self, end_time: float) -> None:
        self.busy_until = end_time

    def step(self, now: float, visible: dict) -> list:
        """visible：{track_id: Box}，畫面上（含寬限期內）參與排序的公車。"""
        if not self.pending or now < self.busy_until:
            return []
        deadline = self.pending[0].time + self.window
        if now < deadline:
            return []
        group = [e for e in self.pending if e.time <= deadline]
        self.pending = [e for e in self.pending if e.time > deadline]

        records, keep = [], []
        for event in group:
            reason = self._reject_reason(event, now, visible)
            if reason is None:
                keep.append(event)
            else:
                kind = "SKIP" if reason.startswith("played") else "DROP"
                records.append(Record(kind, now, events=[event], reason=reason))
        if not keep:
            return records

        boxes = {track_id: box for track_id, box in visible.items()
                 if box.height >= self.min_height}
        for event in keep:                       # 要播的車一定參與排序
            boxes[event.track_id] = visible[event.track_id]
        ordered, consistent = order_buses(boxes, self.order, self.tie_px)
        count = len(ordered)
        positioned = count > 1 and (consistent or not self.cross_check)
        ranked = sorted(((ordered.index(e.track_id), e) for e in keep),
                        key=lambda item: item[0])
        text, speech = build_sentence([(rank, e.number) for rank, e in ranked],
                                      count, positioned, self.number_style)
        for event in keep:
            self.last_announced[event.number] = (now, event.track_id)
        records.append(Record(
            "ANNOUNCE", now, text=text, speech=speech,
            duration=estimate_duration(speech),
            events=[e for _, e in ranked], visible_count=count, order=ordered,
            consistent=consistent, positioned=positioned,
        ))
        return records

    def _reject_reason(self, event: Event, now: float, visible: dict):
        if event.track_id not in visible:
            return f"ID {event.track_id} left"
        age = now - event.time
        if age > self.max_age:
            return f"confirmed {age:.1f} s ago"
        previous = self.last_announced.get(event.number)
        if previous is not None:
            played_at, previous_id = previous
            since = now - played_at
            if since < self.cooldown and (previous_id == event.track_id
                                          or previous_id not in visible):
                return f"played {since:.0f} s ago (ID {previous_id})"
        return None

    def flush_time(self) -> float | None:
        """還有待播的確認時，最晚要推進到哪個時間才會全部處理完（模擬收尾用）。"""
        if not self.pending:
            return None
        return max(self.busy_until, self.pending[-1].time + self.window)
