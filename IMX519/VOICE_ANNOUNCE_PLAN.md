# 公車進站語音播報：Jetson 測試計畫

這份文件給在 Jetson 上執行的 Claude Code 照著做，也讓使用者知道每一步要做什麼、哪些步驟需要使用者本人確認。

目標：`IMX519/nohead_onetime_main.py` 確認公車號碼後，從 Jetson 的喇叭用**中文**播報，例如「5號公車即將進站」；兩台車排隊進站時說出前後關係，例如「前車12號公車即將進站」、「後車5號公車即將進站」。

> 2026-11-17 更新：階段 1～4 已完成。原本計畫用英文和 pyttsx3，實測後改成 **piper 中文語音**，其他差異見「已決定的事」。程式碼對應 GitHub `25e663b`（voice_add）之後、尚未 commit 的版本。

## 進度總覽

| 階段 | 狀態 | 說明 |
| --- | --- | --- |
| 1 環境檢查 | ✅ 完成 | |
| 2 讓 Jetson 發出聲音 | ✅ 完成 | USB 耳機；拔插會自動恢復 |
| 排隊進站播報邏輯 | ✅ 完成 | 規則 1～5 完成；規則 6、7 等月台實測後決定 |
| 3 實作 `voice.py` | ✅ 完成 | `--simulate` 16/16 通過 |
| 4 接進主程式 | ✅ 完成 | |
| 5 接相機整合測試 | ⚠️ 只測過舊版 | 舊版（30 秒冷卻）通過；**排隊播報版還沒接相機測** |
| 6 月台實測 | ❌ 還沒做 | |

---

## 0. 工作規則（Claude 必讀）

- 所有回覆、報告一律用**繁體中文**。
- **只修改** `IMX519/nohead_onetime_main.py`、`IMX519/voice.py`。
  - 例外（使用者明確要求）：`IMX519/main.py` 的 `--flip` 預設值改成 2；這份 md 的更新。
  - **不要動**根目錄的 `main_video_test.py`。
  - `voice_test.py` 是 Windows 上的參考範例，保持原樣。
- 需要 `sudo` 的指令（安裝套件、改群組），只列出來請使用者自己執行，Claude 不輸入密碼。
- **Claude 聽不到聲音**。標示 🔊 的步驟都要停下來問使用者「有沒有聽到、聽起來如何」，得到回覆才能繼續。
- 每個階段結束先回報結果，**使用者同意後才進下一個階段**。
- 不自動 commit。使用者要求時才 commit，而且 commit 訊息**不加** Claude 的 `Co-Authored-By`。
- 這台 Jetson 的 `~/IMX519` 不是 git repo，對應 GitHub `mitnick2100-cmd/ocr_codex` 的 `IMX519/` 子資料夾；比對最新 commit 要抓 `main` 分支（`git ls-remote` 確認 hash）。
- 測試腳本不要用 `python3 -I`：numpy 裝在 `~/.local`，`-I` 會載到不相容的版本。

---

## 已決定的事

| 項目 | 決定 | 原本計畫 |
| --- | --- | --- |
| 喇叭 | Logitech PRO X 2 LIGHTSPEED USB 耳機（ALSA card 2，只支援 48 kHz），走 PulseAudio 預設輸出 | 3.5mm / USB / HDMI 待定 |
| 語言、引擎 | **中文、piper**（`zh_CN-huayan-medium`），常駐背景、啟動時先暖機合成一次 | 英文、pyttsx3 或 espeak-ng |
| 語速 | `--voice-speed 1.25`（piper length_scale，比原速慢 25%；試聽 1.0 / 1.25 / 1.5 後選定） | `--voice-rate` |
| 句子 | 單台：「5號公車即將進站」；有其他車：「前車5號公車即將進站」/「後車5號公車即將進站」 | "Bus 5 is arriving" |
| 合併的多台車 | **一台一句、前車先播**：「前車12號公車即將進站」播完再播「後車5號公車即將進站」 | "Bus 5 and bus 12 are arriving" 一句 |
| 同號碼的第二台 | 和一般車一樣說「後車5號公車即將進站」，不加「也是」或「另一台」（試聽後覺得較不清楚） | "Another bus 5 is arriving" |
| 數字唸法 | 1～2 位數照整數唸（63 → 六十三）；**3～4 位數逐字唸**（309 → 三零九、1050 → 一零五零） | 1～3 位數照整數唸 |
| 前後方向 | `--voice-front right`：flip 2 之後的畫面，公車從左方進入，**前車在畫面右邊** | `bottom` |
| 合併視窗 | `--voice-merge 0.8` 秒 | 同 |
| 換 ID 判斷時間 | `--voice-switch-window 5` 秒；**拿掉原本的 30 秒冷卻**（由規則 3 取代） | 同 |
| V 鍵切換語音 | 不加 | 可選 |
| `--flip` | `nohead_onetime_main.py` 和 `main.py` 預設改成 2（鏡頭倒裝），不用再手動加 | 預設 0 |

---

## 階段 1：環境檢查 ✅

執行過的指令：

```bash
uname -a; cat /etc/nv_tegra_release
python3 --version; pip3 show pyttsx3
which espeak-ng espeak; espeak-ng --voices=en | head
aplay -l
pactl info 2>&1 | head -5; groups
curl -s http://localhost:11434/api/tags | head -c 300
```

結果：

| 項目 | 結果 |
| --- | --- |
| 系統 | Orin Nano，L4T R36.5.2，kernel 5.15.185-tegra，Python 3.10.12 |
| 喇叭 | 板子沒有 3.5mm；HDA（DP/HDMI）沒接有聲音的螢幕；APE（I2S）沒接 codec。接上 **USB 耳機（card 2）** 後才有聲音 |
| 音訊系統 | PulseAudio 可用；使用者已在 `audio` 群組 |
| TTS | espeak-ng 1.50 已安裝（有 `cmn` 普通話）；pyttsx3 2.99 使用者自行安裝（直接載入 `libespeak-ng.so.1`，不需要 `libespeak1`） |
| Ollama | 執行中，模型 `qwen3-vl:2b-instruct` |

---

## 階段 2：讓 Jetson 發出聲音 ✅

| 步驟 | 結果 |
| --- | --- |
| PulseAudio 預設輸出 | 一開始沒建立耳機的輸出裝置，預設是沒接喇叭的 `platform-sound`。用下面的做法 A 修好 |
| `speaker-test` | 通過 |
| `espeak-ng` 英文 | 通過；語速試過 150、100、125 |
| pyttsx3 連講兩句 | 沒卡住，但第一次 `runAndWait()` 0.01 秒就返回，沒有等播完 → 不採用 |
| espeak-ng 中文（`cmn`） | 能播，但機器人腔明顯 |
| **piper 中文** | 採用。CPU 上 RTF 約 0.11；模型載入 0.42 秒；暖機後每句合成約 0.2～0.3 秒 |
| 拔插 USB 接收器 | ✅ PulseAudio 自動重建耳機輸出（profile 變成 `analog-stereo`）並自動切回預設輸出，約 1 秒恢復。拔掉期間預設會跳到 `platform-sound`，播報沒有聲音也不會警告 |

做法 A（PulseAudio 沒有耳機輸出裝置時，不用 sudo）：

```bash
C=alsa_card.usb-Logitech_PRO_X_2_LIGHTSPEED_0000000000000000-00
pactl set-card-profile $C off; pactl set-card-profile $C output:stereo-fallback
pactl set-default-sink alsa_output.usb-Logitech_PRO_X_2_LIGHTSPEED_0000000000000000-00.stereo-fallback
```

**重開機後**還沒測過會不會自動選到耳機；執行前先用 `pactl get-default-sink` 確認是 `alsa_output.usb-Logitech_...`。

piper 安裝方式（不用 sudo；**不要** `pip install piper-tts`，會裝一般版 onnxruntime，跟 Jetson 的 `onnxruntime-gpu` 1.23 衝突）：

```bash
mkdir -p ~/piper/voices && cd ~/piper
curl -sL -O https://github.com/rhasspy/piper/releases/download/2023.11.14-2/piper_linux_aarch64.tar.gz
tar xzf piper_linux_aarch64.tar.gz
cd voices; B=https://huggingface.co/rhasspy/piper-voices/resolve/main/zh/zh_CN/huayan/medium
curl -sL -O $B/zh_CN-huayan-medium.onnx; curl -sL -O $B/zh_CN-huayan-medium.onnx.json
```

數字唸法實測（piper 用自己內建的 espeak-ng 1.52 轉音標，和系統的 espeak-ng 1.50 結果不同；要確認唸法用 `piper --debug` 看音標，不能用系統的 espeak-ng）：

| 輸入 | piper 直接唸 | 判斷 |
| --- | --- | --- |
| 63 | 六十三 | ✅ 照整數唸 |
| 300 / 358 | 三百 / 三百五十八 | 使用者決定改逐字唸 |
| 1050 | 一千**五十**（少了「零」） | ❌ → 逐字唸 |

---

## 排隊進站的播報邏輯 ✅

### 現況：兩台車排隊時會發生什麼

1. 前車的框完整，會先被確認，然後播報。
2. 後車被前車擋住、框太窄（`[NARROW]`，長寬比小於 0.5），所以不會送 OCR。要等它開到前面、框變寬才會辨識。
3. 結果是：後車通常比較晚才播報，可能已經停到月台旁邊了。如果它一直停在前車後面上下客，可能**完全不會播報**。
4. OCR 只有一個 worker（佇列上限 2），兩台車的確認至少相隔一次 OCR（實測 0.68～0.76 秒）。

### 規則

| # | 規則 | 狀態 |
| --- | --- | --- |
| 1 | 合併短時間內確認的車：收到第一筆後再等 `--voice-merge` 秒（0.8） | ✅ |
| 2 | 依前後位置排序：`--voice-front right`，框中心 x 越大越前面 | ✅ |
| 3 | 去重複要分辨「第二台同號碼」和「換 ID」 | ✅（判斷方式見下方，比原計畫補強） |
| 4 | 播報前確認車還在，離開的印 `skipped (left)` | ✅（換過 ID 時看接手的新 ID） |
| 5 | 不中途打斷，整句播完才播下一句 | ✅ |
| 6 | 提示後面還有車 | ⏸ 可選，月台實測後決定 |
| 7 | 分兩階段播報（進站 / 停妥） | ⏸ 可選，月台實測後決定 |

### 規則 3 的判斷方式

同一個號碼要再播時，拿**新車剛出現時的框**，和**前一台同號碼最後的框**比（框中心距離小於前一個框寬度的一半算「相近」）：

| 兩台的時間關係 | 位置 | 判斷 |
| --- | --- | --- |
| 新車出現之後，前一台還有被看到（兩框同時存在） | 不相近 | 真的是第二台，照常播報 |
| 新車出現之後，前一台還有被看到（兩框同時存在） | 相近 | 同一台車被重複框到，丟棄 |
| 前一台在新車出現前就不見了，間隔 ≤ `--voice-switch-window`（5 秒） | 相近 | ByteTrack 換了 ID，丟棄（`id switch skip`） |
| 其他情況 | — | 新的一班車，照常播報 |

為什麼和原計畫不同（原計畫寫「前一台還在畫面中 → 第二台」，並用確認當下的框比位置）：

- **用剛出現時的框**：確認要等 OCR（約 0.7 秒），進站中的車這段時間可能已移動超過半個車身，用確認當下的框會誤判成新的一班車而**重複播報**。
- **用「同時存在」而不是「還在畫面中」**：換 ID 時，舊 ID 還會在寬限期（`GRACE_FRAMES` 15 幀，約 0.9 秒）內算「在畫面中」，原本的判斷會把換 ID 誤判成第二台。

另外兩個補強：

- **換 ID 接手**：判定換 ID 時記下「舊 ID → 新 ID」。舊 ID 還在佇列裡、播報前就消失時，改看新 ID 是否還在、在哪裡。
- **沒播出就清掉紀錄**：車子因為離開或佇列滿而沒播出時，清掉這個號碼的「已播過」紀錄，之後同號碼的車（例如被擋住後換 ID 重新出現）會照常播報。
- **前後關係排除同一台車**：判斷單台車是前車還是後車時，排除跟它位置相近的框（多半是同一台換了 ID、還沒確認），避免單台車被說成「後車」。

### 流程圖

```
【A】OCR worker：確認號碼（process_ocr 印出 CONFIRMED 之後呼叫 announce(track_id, N)）
        │
        ▼
  ┌─────────────────────────┐
  │ 這個 track ID 處理過了嗎？ │── 是 ──▶ 丟棄
  └─────────────────────────┘
        │ 否
        ▼
  ┌──────────────────────────────────┐
  │ 之前有沒有播過（或排隊中）號碼 N？ │── 沒有 ──▶ 放進待播佇列
  └──────────────────────────────────┘
        │ 有，拿新車剛出現時的框 vs 前一台 N 號最後的框
        ├─ 兩框同時存在、位置不近 ─────────▶ 第二台，放進待播佇列
        ├─ 兩框同時存在、位置相近 ─────────▶ 同一台重複框到，丟棄
        ├─ 前一台消失 ≤ 5 秒、位置相近 ────▶ 換 ID，丟棄（記下舊 ID → 新 ID）
        └─ 其他情況 ───────────────────────▶ 新的一班車，放進待播佇列
                                              （佇列滿 → queue full，丟棄並清掉紀錄）


【B】播報執行緒
        │
        ▼
   等待佇列裡的第一筆 ◀──────────────────────────────┐
        │                                              │
        ▼                                              │
   再收集 0.8 秒內進來的其他筆（合併視窗）             │
        │                                              │
        ▼                                              │
   逐筆檢查：這台車（或接手的新 ID）還在畫面中嗎？    │
        │  已經離開的丟掉，印 skipped (left)，清掉紀錄  │
        ▼                                              │
   全部都離開了嗎？── 是 ──────────────────────────────┤
        │ 否                                           │
        ▼                                              │
   依前後位置排序（畫面右邊的排前面）                  │
        │                                              │
        ▼                                              │
   組成句子（一台一句）                                │
     單台、畫面沒有其他車：「5號公車即將進站」         │
     單台、有其他車：      「前車5號…」或「後車5號…」  │
     合併兩台：「前車12號公車即將進站」                │
               「後車5號公車即將進站」                 │
        │                                              │
        ▼                                              │
   依序播報（每句播完才播下一句，不會被打斷）          │
   每句印 [VOICE] log ─────────────────────────────────┘
```

**可選：提示後面還有車（規則 6）**，還沒做。這部分在主迴圈處理，和 A、B 平行執行：

```
已經播報前車，而且後面還有一個框太窄（[NARROW]）、還沒辨識的框
        │
        ▼
持續出現超過約 2 秒？── 否 ──▶ 不處理
        │ 是
        ▼
同一群車只放一次「後面還有公車」到待播佇列
```

### 重點

- **A 只做判斷**：決定「要不要播」，然後把結果丟進佇列，不等播報完成，所以不會拖慢 OCR。
- **B 負責時機**：合併短時間內確認的車、確認車還在、排出前後順序，然後一句一句播完。
- **共用資料**：主程式的 `get_tracks()` 在 `lock` 內讀取每台車的最後的框、最後被看到的時間、第一次出現的框和時間，以及 `last_seen_frame`。

---

## 階段 3：實作 `IMX519/voice.py` ✅

寫法比照 `daynight.py`，把語音做成獨立模組，主程式只需要接幾行。

### 對外介面

```python
add_arguments(parser)        # --no-voice / --voice-merge / --voice-switch-window / --voice-front /
                             # --voice-digits / --voice-speed / --piper-bin / --piper-model
check_args(args)             # 檢查參數範圍
VoiceAnnouncer(args, tracks_fn)
                             # 建立後就在背景執行緒待命（先啟動 piper 並暖機）
                             # tracks_fn() 由主程式提供，回傳
                             # {track_id: Track(present, box, gone, first_box, age)}
                             #   present：GRACE_FRAMES 內出現過；box：最後的框；gone：最後被看到幾秒前
                             #   first_box：第一次出現的框；age：第一次出現幾秒前
announcer.announce(track_id, number)
                             # 確認號碼時呼叫；不會卡住呼叫的執行緒
announcer.close()            # 程式結束時呼叫：中斷正在播的句子、結束 piper，不等佇列播完
```

### 參數

| 參數 | 預設 | 說明 |
| --- | --- | --- |
| `--no-voice` | 關 | 完全停用語音 |
| `--voice-merge` | 0.8 | 合併視窗秒數（0～3）；0 是每台車各自播 |
| `--voice-switch-window` | 5 | 換 ID 判斷時間 |
| `--voice-front` | `right` | 前車在畫面哪一邊（`right` / `left` / `bottom`） |
| `--voice-digits` | 關 | 1～2 位數也逐字唸 |
| `--voice-speed` | 1.25 | piper length_scale（0.5～2.0），越大越慢 |
| `--piper-bin` | `~/piper/piper/piper` | piper 執行檔 |
| `--piper-model` | `~/piper/voices/zh_CN-huayan-medium.onnx` | 中文語音模型 |

### 規格

- **播放方式**：常駐 piper 子程序（`--json-input`，模型只載入一次），每句寫一行 JSON，piper 在暫存資料夾寫出 wav 並印出路徑，再用 `paplay` 播放，播完刪掉 wav。啟動時先合成一句不播放的暖機句子，第一句合成從 0.6～0.8 秒降到約 0.22 秒。
- **佇列**：上限 4 則，用 `put_nowait`，滿了就丟掉並印 `[VOICE] queue full, dropped ...`。
- **句子**：放在檔案上方當常數，`VOICE_ONE = "{bus}公車即將進站"`、`VOICE_POS_ONE = "{pos}{bus}公車即將進站"`、`POS_FRONT = "前車"`、`POS_BACK = "後車"`。
- **失敗不影響辨識**：找不到 piper / 模型 / paplay、模型壞掉、暖機失敗時，印警告並停用語音，主程式照常執行。paplay 回傳錯誤只警告、不停用（PulseAudio 暫時不在時下一句可能就好）。
- **Ctrl+C 不留孤兒程序**：piper 和 paplay 用 `start_new_session=True` 啟動，終端機的 SIGINT 不會直接打到它們，一律由 `close()` 結束；「開新 paplay」和 `close()` 用同一把鎖保護。
- **每句都印 log**：`[HH:MM:SS] [VOICE] 5號公車即將進站  (synth 0.21 s)`；另外有 `ready (piper warmup ...)`、`skipped (left) ...`、`id switch skip ...`、`queue full, dropped ...`。
- **可以單獨執行**：`python3 IMX519/voice.py 5 300 1050`。每個號碼當成一台車、每 3 秒確認一台，全部停在同一位置（分不出前後，每句都是單台說法）。
- **模擬模式**：`python3 IMX519/voice.py --simulate`。不接相機、不出聲，用假的 `tracks_fn` 和不出聲的播報器（每句佔 1 秒）跑內建情境，印出預期和實際句子，最後印通過幾個。加 `--play` 會用 piper 實際播出。

### 模擬情境（16 個，全部通過）

| 情境 | 預期播報 |
| --- | --- |
| 單台車 | 5號公車即將進站 |
| 兩台同時確認（前車後確認） | 前車12號公車即將進站 → 後車5號公車即將進站 |
| 兩台相隔較久 | 5號公車即將進站 → 後車12號公車即將進站 |
| 連續兩台同號碼 | 5號公車即將進站 → 後車5號公車即將進站 |
| 兩台同號碼同時確認 | 前車5號公車即將進站 → 後車5號公車即將進站 |
| ByteTrack 換 ID（舊 ID 已消失） | 只有第一句（id switch 1 次） |
| ByteTrack 換 ID（舊 ID 仍在寬限期內） | 只有第一句（id switch 1 次） |
| ByteTrack 換 ID，車子邊開邊確認 | 只有第一句（id switch 1 次） |
| 同一台車被重複框到 | 只有第一句（id switch 1 次） |
| 單台車等播報時換 ID（新 ID 還沒確認） | 5號公車即將進站（不能說成後車） |
| 同號碼的下一班車（消失超過 5 秒） | 兩次都播 |
| 車已經開走 | 不播，`skipped (left)` |
| 播報中有新的確認 | 第一句播完才播第二句（不重疊） |
| 沒播到就被擋住，換 ID 重新出現 | 第二次出現時照常播 |
| 排隊等播報時換 ID | 12號公車即將進站 → 後車5號公車即將進站 |
| 佇列滿了 | 多出來的丟掉，queue full 1 次 |

「邊開邊確認」和「單台車等播報時換 ID」兩個情境，用修正前的邏輯會失敗（前者重複播報、後者說成「後車5號」），確認測得出問題。

### 驗證（不需要相機）

```bash
python3 -m py_compile IMX519/voice.py
python3 IMX519/voice.py --simulate          # 16/16 passed，約 40 秒
python3 IMX519/voice.py --simulate --play   # 🔊 實際播出，約 1.5 分鐘
python3 IMX519/voice.py 5 300 1050          # 🔊 試聽
python3 IMX519/voice.py --piper-bin /nonexistent 5   # 只印警告，不中斷
```

---

## 階段 4：接進 `nohead_onetime_main.py` ✅

| 位置 | 修改內容 |
| --- | --- |
| 檔案開頭的 docstring | 補一段語音播報的說明 |
| `import` 區 | `import voice` |
| 共用狀態區 | 新增 `last_box`、`last_seen_time`、`first_box`、`first_seen_time`、`latest_frame_id`、`announcer` |
| 新增函式 `get_tracks()` | 在 `lock` 內回傳 `{track_id: voice.Track(...)}`；「在畫面中」與 `get_visible_results` 相同（`GRACE_FRAMES` 內）；秒數用 `time.monotonic()` 記錄，不用幀數換算 |
| `process_ocr` 印出 `CONFIRMED` 之後 | `announcer.announce(track_id, result)` |
| `parse_args` | `voice.add_arguments(parser)`；`voice.check_args(args)`；`--flip` 預設改成 2 |
| `main()` 中 `warmup_ollama()` 之後 | `announcer = voice.VoiceAnnouncer(args, get_tracks)`（在 OCR worker 啟動前） |
| 主迴圈更新 `last_seen_frame` 的地方 | 同時更新最後 / 第一次出現的框和時間、`latest_frame_id`；拿掉沒用到的 `current_ids` |
| `forget_stale_tracks` | 清除過期 ID 時一起刪掉新增的字典 |
| `finally` | 存好 MP4 之後 `announcer.close()` |

**不要**在主迴圈 `bus_text` 改變的地方播報。那裡會出現 `NONE` 和 `"5,12"` 這類字串，而且車被擋住時會反覆跳動。

### 驗證

```bash
python3 -m py_compile IMX519/nohead_onetime_main.py
python3 IMX519/nohead_onetime_main.py --help | grep -A2 voice
```

另外做過不接相機的串接測試：假造 Ollama 回傳，直接呼叫主程式的 `process_ocr` 和 `get_tracks`，前後兩台、車子離開、換 ID 重新出現三個案例都正確。

---

## 階段 5：接相機的整合測試 🔊 ⚠️

在相機前放公車照片或影片（手機或平板全螢幕播放），執行（`--flip 2` 已是預設，不用加）：

```bash
pactl get-default-sink    # 確認是 alsa_output.usb-Logitech_...
python3 IMX519/nohead_onetime_main.py --record 2>&1 | tee voice_run.log
python3 IMX519/nohead_onetime_main.py --record --no-voice 2>&1 | tee novoice_run.log
```

### 舊版（30 秒冷卻）結果，2026-11-17 使用者自測

| 項目 | 結果 |
| --- | --- |
| `CONFIRMED` → `[VOICE]` 在 1 秒內 | ✅ |
| 兩台依序播、不打斷 | ✅（309 → 75） |
| `--no-voice` 沒有 `[VOICE]` | ✅ |
| 正常結束 | ✅ |
| 換 ID 不重播 | ⚪ 沒測到 |
| 幀率比較 | ⚪ 狀態列沒有 FPS，log 看不出來 |

### 排隊播報版要檢查的項目（還沒測）

- [ ] 每一行 `CONFIRMED [ID:x] N` 之後都有對應的 `[VOICE]`；加上合併視窗後時間差約 1.0～1.1 秒。
- [ ] 同一台車沒有重複播報；被擋住後換了 ID 也一樣（log 有 `id switch skip`）。
- [ ] 同時放兩張不同號碼的公車照片：**右邊那張先播**「前車…」，再播「後車…」。
- [ ] 同時放兩張相同號碼的公車照片：兩台都有播，第二台說「後車N號」。
- [ ] 拿走照片後才輪到播報的車，不會播出來（log 有 `skipped (left)`）。
- [ ] 兩台車的確認常常相隔超過 0.8 秒、被拆成兩批的話，試 `--voice-merge 1.0`。
- [ ] 按 `Q` 或 `Ctrl+C` 可以正常結束，不會卡在播報執行緒，也沒有殘留的 piper / paplay。
- [ ] 加上 `--no-voice` 時完全沒有聲音，也沒有 `[VOICE]` log。
- [ ] 執行中拔插 USB 接收器，插回後下一句恢復從耳機播出（只測過程式沒在跑時拔插）。

---

## 階段 6：月台實測（使用者執行）

1. 到公車站執行階段 5 的指令，連續跑 15～30 分鐘。
2. 使用者在現場記錄：實際進站的車號、聽到的播報、播報時車子的位置（太早、剛好或太晚）。**排隊進站**的情況要特別記下來：後車有沒有播報、什麼時候播報、前後順序對不對。
3. Claude 拿 `voice_run.log` 和使用者的紀錄比對，整理成：

| 時間 | 實際車號 | CONFIRMED | 播報內容 | 延遲 | 備註 |
| --- | --- | --- | --- | --- | --- |

4. 根據結果提出調整建議，例如合併視窗、換 ID 判斷時間、語速、句子內容，以及是否要做規則 6、7。

---

## 執行需要的檔案

| 類別 | 項目 |
| --- | --- |
| 專案程式（`IMX519/`） | `nohead_onetime_main.py`、`voice.py`、`daynight.py`、`ContinuousAutofocus_turbo.py`、`JetsonCamera.py`、`Focuser.py` |
| YOLO 模型 | 先找 `IMX519/yolov8n.pt`，再找上一層 `yolov8n.pt`（這台用 `~/yolov8n.pt`）；ByteTrack 設定用 ultralytics 內建的 `bytetrack.yaml` |
| OCR | Ollama `qwen3-vl:2b-instruct`，`http://localhost:11434` |
| 語音 | `~/piper/piper/`（執行檔與內建 espeak-ng-data、onnxruntime）、`~/piper/voices/zh_CN-huayan-medium.onnx` 和 `.onnx.json`、系統的 `paplay` |
| 對焦 | `/dev/i2c-10`（有 smbus2 就直接寫，沒有就用 `i2cset`） |
| Python 套件 | opencv、numpy、requests、torch、ultralytics（可選 smbus2） |
| 執行時產生 | `--record` 的 `record_<日期>_<時間>.mp4`、`C` 鍵截圖、`/tmp/voice_*/` 暫存 wav（播完即刪） |

---

## 已知風險與待確認

- **合併視窗貼近 OCR 間隔**：兩台車確認相隔約 0.7 秒，合併視窗 0.8 秒。超過時會拆成兩批，但後車只要在畫面中（即使還沒確認），前車那句仍會說「前車5號公車即將進站」，接著另一批播「後車12號公車即將進站」，聽起來和合併時一樣；差別只是前車那句如果後車還沒進畫面，就只說「5號公車即將進站」。
- **播報延遲**：合併視窗讓每次播報多等約 0.8 秒。
- **寬限期、車速是模擬假設**：模擬用 0.9 秒寬限期和固定位移，實際要看月台資料。
- **拔掉耳機期間沒有聲音也沒有警告**：PulseAudio 預設會跳到 `platform-sound`。
- **重開機後的預設輸出**還沒測。

## 完成時的回報格式

- 修改了哪些檔案（附上 diff 摘要）。
- 每個階段的結果：通過、失敗或跳過，以及原因。
- 使用者確認過的聽覺結果。
- 還沒驗證的項目，以及下一步建議。
