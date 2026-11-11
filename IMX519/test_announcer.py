"""announcer.py 的單元測試（不需要相機、YOLO 或 Ollama）。

    python3 -m unittest IMX519/test_announcer.py
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from announcer import (AnnouncerCore, Box, Event, build_sentence, number_words,
                       order_buses)

FRONT = Box(300, 300, 700, 700)     # 框下緣 700、面積大：前車
REAR = Box(650, 250, 800, 450)      # 框下緣 450、被擋住只剩細條：後車


class NumberWordsTest(unittest.TestCase):
    def test_mixed(self):
        cases = {"5": "五", "10": "十", "12": "十二", "20": "二十", "25": "二十五",
                 "307": "三零七", "1501": "一五零一", "05": "零五", "0": "零"}
        for number, words in cases.items():
            self.assertEqual(number_words(number), words, number)

    def test_value(self):
        cases = {"307": "三百零七", "110": "一百一十", "1010": "一千零一十",
                 "1501": "一千五百零一", "2000": "二千", "15": "十五"}
        for number, words in cases.items():
            self.assertEqual(number_words(number, "value"), words, number)

    def test_digits(self):
        self.assertEqual(number_words("12", "digits"), "一二")


class OrderTest(unittest.TestCase):
    def test_bottom_edge_and_area_agree(self):
        self.assertEqual(order_buses({1: REAR, 2: FRONT}), ([2, 1], True))

    def test_front_bus_leaving_frame(self):
        # 前車開出畫面，框被切到只剩一小塊：下緣仍最低，但面積比後車小 -> 不一致
        leaving = Box(1200, 600, 1280, 720)
        rear = Box(500, 250, 800, 500)
        self.assertEqual(order_buses({1: leaving, 2: rear}), ([1, 2], False))

    def test_tie_uses_height(self):
        a = Box(0, 300, 100, 600)
        b = Box(200, 200, 300, 605)     # 下緣只差 5 px：比框高，b 較高
        self.assertEqual(order_buses({1: a, 2: b})[0], [2, 1])

    def test_sentence(self):
        self.assertEqual(build_sentence([(0, "307"), (1, "262")], 2, True),
                         ("前方 307，後方 262 號公車進站", "前方三零七，後方二六二號公車進站"))
        self.assertEqual(build_sentence([(0, "307")], 1, False)[0], "307 號公車進站")
        self.assertEqual(build_sentence([(2, "18")], 3, True)[0], "第三輛 18 號公車進站")
        self.assertEqual(build_sentence([(0, "307"), (1, "262")], 2, False)[0],
                         "307、262 號公車進站")


class AnnouncerCoreTest(unittest.TestCase):
    def test_two_buses_in_window_share_one_sentence(self):
        core = AnnouncerCore()
        visible = {1: FRONT, 2: REAR}
        core.notify(Event(2, "262", 0.0))
        core.notify(Event(1, "307", 1.2))
        self.assertEqual(core.step(1.9, visible), [])        # 還在收集時間內
        [record] = core.step(2.0, visible)
        self.assertEqual(record.kind, "ANNOUNCE")
        self.assertEqual(record.text, "前方 307，後方 262 號公車進站")

    def test_rear_bus_hidden(self):
        core = AnnouncerCore()
        core.notify(Event(1, "307", 0.0))
        [record] = core.step(2.0, {1: FRONT, 2: REAR})        # 2 號還沒辨識也算一輛
        self.assertEqual(record.text, "前方 307 號公車進站")

    def test_inconsistent_order_drops_position(self):
        core = AnnouncerCore()
        core.notify(Event(1, "307", 0.0))
        visible = {1: Box(1200, 600, 1280, 720), 2: Box(500, 250, 800, 500)}
        [record] = core.step(2.0, visible)
        self.assertEqual(record.text, "307 號公車進站")
        self.assertFalse(record.consistent)

    def test_busy_delays_next_group(self):
        core = AnnouncerCore()
        core.notify(Event(1, "307", 0.0))
        core.step(2.0, {1: FRONT})
        core.mark_busy_until(6.0)                              # 第一句播到 6.0 秒
        core.notify(Event(2, "262", 2.5))
        self.assertEqual(core.step(4.6, {1: FRONT, 2: REAR}), [])   # 收集時間到了，但還在播
        self.assertEqual(core.step(6.0, {1: FRONT, 2: REAR})[0].text,
                         "後方 262 號公車進站")

    def test_bus_left_and_stale(self):
        core = AnnouncerCore(max_age=10)
        core.notify(Event(1, "307", 0.0))
        [record] = core.step(2.0, {})
        self.assertEqual((record.kind, record.reason), ("DROP", "ID 1 left"))
        core.notify(Event(2, "262", 10.0))
        core.mark_busy_until(25.0)
        [record] = core.step(25.0, {2: REAR})
        self.assertEqual(record.kind, "DROP")

    def test_cooldown_new_id_same_bus(self):
        core = AnnouncerCore()
        core.notify(Event(1, "307", 0.0))
        core.step(2.0, {1: FRONT})
        core.notify(Event(5, "307", 20.0))                     # 被擋後換了 ID，舊 ID 已不在
        [record] = core.step(22.0, {5: FRONT})
        self.assertEqual(record.kind, "SKIP")
        core.notify(Event(6, "307", 90.0))                     # 冷卻過了
        self.assertEqual(core.step(92.0, {6: FRONT})[0].kind, "ANNOUNCE")

    def test_two_buses_same_number(self):
        core = AnnouncerCore()
        core.notify(Event(1, "307", 0.0))
        core.step(2.0, {1: FRONT, 2: REAR})
        core.notify(Event(2, "307", 5.0))                      # 1 號還在畫面：是另一輛 307
        [record] = core.step(7.0, {1: FRONT, 2: REAR})
        self.assertEqual(record.text, "後方 307 號公車進站")

    def test_min_height_ignores_far_bus(self):
        core = AnnouncerCore(min_height=100)
        core.notify(Event(1, "307", 0.0))
        [record] = core.step(2.0, {1: FRONT, 2: Box(600, 100, 640, 140)})
        self.assertEqual(record.text, "307 號公車進站")


if __name__ == "__main__":
    unittest.main()
