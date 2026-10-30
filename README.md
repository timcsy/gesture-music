# 手勢和弦演奏器（純前端網頁）

用瀏覽器相機辨識手勢數字 1–7，選擇 C 大調級數並持續演奏。網頁本身是靜態 HTML、CSS、JavaScript；手勢辨識在瀏覽器端透過 MediaPipe Tasks Vision 的 WASM 執行，不需要 Python 後端或伺服器 API。模型與樂器音檔放在本資料夾，MediaPipe JavaScript/WASM 套件由 jsDelivr 載入。

## 開啟

建議在此資料夾開啟終端機執行靜態檔案伺服器：

```powershell
python -m http.server 8000
```

再用瀏覽器開啟 <http://localhost:8000>，按「開啟相機」並允許相機權限。也可以把資料夾部署到支援靜態網站的主機；正式網站需使用 HTTPS 才能存取相機。直接雙擊 `index.html`（`file://`）通常不能使用相機。

首次開啟需要網路連線載入 MediaPipe 的 JS/WASM；辨識模型、鋼琴與吉他音檔由網站本身提供。按「開啟相機」後會初始化辨識與音訊。

## 演奏方式

| 手勢 | 級數 | 和弦 |
| --- | --- | --- |
| 1 | I | C 大三和弦 |
| 2 | ii | D 小三和弦 |
| 3 | iii | E 小三和弦 |
| 4 | IV | F 大三和弦 |
| 5 | V | G 大三和弦 |
| 6 | vi | A 小三和弦 |
| 7 | vii° | B 減三和弦 |

鋼琴模式演奏分解和弦。電吉他模式演奏根音、五度、八度組成的 Power Chord，以 96 BPM、每小節八下的八分音符節奏加入破音。手勢放開約半秒會停止；也可以按畫面上的級數卡或鍵盤的 **1–7** 選和弦。**Tab** 切換音色，**P** 選鋼琴，**G** 選電吉他，**Q** 或 **Esc** 停止相機。

## 音色來源

鋼琴採 Meta Open Sound 的 Meta Piano 立式鋼琴多重取樣；電吉他採 FreePats Fender 電吉他取樣並加入瀏覽器端過載效果。原始音源均為 CC0。來源與授權見 [鋼琴音源](assets/piano/ATTRIBUTION.txt)、[電吉他原始音源](assets/electric_guitar/ATTRIBUTION.txt) 與 [網頁用轉檔說明](assets/web_guitar/ATTRIBUTION.txt)。

## Python 版本

舊版桌面程式仍保留：安裝 `requirements.txt` 後執行 `python gesture_piano.py`。
