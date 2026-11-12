# FTS-Diffusion 重現與改良版

本資料夾包含兩個版本：

| 版本 | 程式 | 結果資料夾 | 內容 |
|---|---|---|---|
| 原始論文版 | `ftsd/fts_diffusion.py` | `results/<資產>/fts/` | 依 ICLR 2024 論文 *Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns* 實作：SISC、型態條件 DDPM + scaling autoencoder、馬可夫型態演化網路 |
| 改良版（簡報版） | `ftsd/fts_plus.py`、`ftsd/garch.py` | `results/<資產>/fts_plus/` | 依 `Group_10_Slides.pdf`：長記憶因果 Transformer + MDN、DiT 去噪器（移除 autoencoder）、GJR-GARCH-X 過濾、成交量條件 |

兩版共用 SISC（`ftsd/sisc.py`、`ftsd/dtw.py`）、資料載入、評估與下游實驗程式，因此比較時只有模型本身不同。基準模型 RCGAN、TimeGAN、CSDI 在 `ftsd/baselines.py`。

## 環境

Python 3.10 以上，需要 GPU（我們在 Colab 上執行）。Colab 已內建所有套件；本機執行時：

```bash
pip install -r requirements.txt
```

## 資料

`data/` 已附上三個資產的日收盤價與成交量，不需要另外下載：

| 檔案 | 來源 |
|---|---|
| `data/GSPC.csv` | Yahoo Finance `^GSPC`（`download_data.py` 下載） |
| `data/ZC_F.csv` | Yahoo Finance `ZC=F`（`download_data.py` 下載） |
| `data/archive_GOOG.csv` | Kaggle S&P 500 stocks 資料集 `sp500_stocks.csv` 中的 GOOG |

要用其他股票時，把 Kaggle 的 `sp500_stocks.csv` 放在本資料夾上一層的 `archive/`，`--asset` 填股票代號即可。

## 執行流程

每個資產的完整流程（在 Colab 上一個資產約 1–2 小時）：

```bash
python sisc_simulation.py --patterns 4         # 附錄 B.3：SISC 在模擬資料上的驗證
python train.py --asset GOOG                   # 訓練 fts, fts_plus, rcgan, timegan, csdi 並產生合成路徑
python train.py --asset GOOG --models fts_plus_nogarch fts_plus_novol   # 改良版消融實驗
python evaluate.py --asset GOOG                # 表 1/3/4、圖 5、stylized facts
python downstream.py --asset GOOG --horizon 1  # 圖 6：TMTR 與 TATR
python downstream.py --asset GOOG --horizon 5  # 圖 10：五日預測
```

`--asset` 可用 `SP500`、`GOOG`、`ZC=F`。`train.py --quick` 只跑極小的訓練量，用來檢查流程。

`results/` 只附上評估結果（表格、圖、`meta.json`）；模型權重 `model.pt` 與合成路徑 `samples.npz` 太大沒有放進 repo，重跑 `train.py` 即可產生。

## 結果摘要

完整表格在 `results/<資產>/tables.md`（生成品質、典型特徵）與 `results/<資產>/downstream_h1.md`、`downstream_h5.md`（下游 LSTM）。

**生成品質（對測試期報酬，FTS → FTS+）**

| 資產 | W ↓ | ΔSR ↓ | KS p 值 ↑ |
|---|---|---|---|
| S&P 500 | 0.0071 → 0.0025 | 0.651 → 0.364 | 0.000 → 0.010 |
| GOOG | 0.0024 → 0.0036 | 0.689 → 0.399 | 0.157 → 0.171 |
| ZC=F | 0.0020 → 0.0089 | 0.336 → 0.594 | 0.227 → 0.014 |

- 兩個 FTS 版本的生成品質都遠勝 RCGAN 與 TimeGAN。
- FTS+ 在 S&P 500 與 GOOG 上勝過原始版，在 ZC=F 上落後。
- FTS+ 在 S&P 500 與 ZC=F 上波動偏高（std 0.019 對真實 0.012、0.031 對 0.019）：GARCH 持續性接近 1，訓練期又含極端日（1987 股災、玉米期貨換月跳空），變異數上限因此過高。拿掉 GARCH 的消融版本 std 對得上。

**下游 LSTM**

- 只用真實資料的 LSTM，MAPE 幾乎等於「預測明天價格不變」：S&P 500 0.00572 對 0.0057、GOOG 0.01005 對 0.0101、ZC=F 0.01025 對 0.0102。
- TATR 加入 25,200 筆合成資料後，FTS+ 的 MAPE 變化在 −0.7% 到 +10.6% 之間，三個資產都沒有明顯改善；我們沒有重現論文報告的 15–18% 改善。
- 論文的 naive MAPE（0.046–0.057）是我們以價格計算的 5–8 倍，兩邊的 MAPE 尺度不同。論文的 TATR 從「有限的真實資料」開始，我們使用全部訓練資料。

## 論文與程式的對應

| 論文 | 程式 |
|---|---|
| Sec. 4.1、Algorithm 1（SISC） | `ftsd/sisc.py`：`SISC._init_centroids`（距離加權初始化）、`_segment`（Eq. 1 貪婪切割）、`_update`（中心更新） |
| Eq. 2（DTW） | `ftsd/dtw.py`：`dtw_to_centroids`（GPU 批次 DTW） |
| Sec. 4.2、Eq. 3–5、附錄 C | `ftsd/fts_diffusion.py`：`ScalingAE`、`PatternDenoiser`（6 個 residual TCN block，N = 100）、`_fit_generation` |
| Sec. 4.3、Eq. 6–7、附錄 D.1 | `PatternEvolutionNet`、`_fit_evolution`（Adam 4e-4，1000 epochs） |
| Algorithm 2 | `FTSDiffusion.sample` |
| 附錄 E.1 資料設定 | `ftsd/data.py`：S&P 500 1980–2020、GOOG 2005–2020、ZC=F 2001–2020，前 80% 訓練 |
| 表 1、3、4 | `evaluate.py` + `ftsd/metrics.py` |
| 圖 5 | `evaluate.py`：`fig5_<model>.png` |
| 圖 6、10（TMTR / TATR） | `downstream.py` |
| 附錄 B.3 | `sisc_simulation.py` |

## 論文沒寫清楚、由實作決定的地方

1. **SISC 的輸入與正規化**：在 log 價格上做切割，每個片段 min-max 正規化到 [0, 1]（附錄 B.3 的 normalized space）。片段包含起點，共 l + 1 個點，相鄰片段共用端點，因此可以用 `x[s] + alpha * (u - u[0])` 無誤差還原。
2. **DTW 距離**：平方誤差，再除以 (片段長度 + 中心長度)，否則較長的片段會吃虧。
3. **中心更新**：論文只寫 "update P"，這裡用 DBA（DTW barycenter averaging，10 次）。
4. **多次初始化**：SISC 從 5 組初始化中保留 inertia 最低者（同 k-means 的 n_init）。模擬資料上，單次初始化的 4 型態 Jaccard 為 0.53，5 次為 0.71。
5. **擴散目標**：依 Eq. 3 對「表示 − 型態」的殘差做擴散；殘差目標對 encoder 做 stop-gradient，避免 encoder 把殘差縮到 0 讓去噪變得沒有意義。
6. **型態演化的抽樣**：Eq. 7 對長度與幅度用 MSE，等同固定變異數的高斯概似，因此抽樣時以網路輸出為平均、訓練殘差標準差為標準差。`--no_evo_noise` 改回只用點預測。
7. **KS / AD / ES**：表 1 的數值被截在 [0, 1] 與 [0.01, 0.25]，符合 p 值而非統計量，所以這裡回報 p 值（越高越好）。p 值會隨樣本數變化，合成路徑一律截成與參考資料相同長度。
8. **下游預測的 MAPE**：論文沒寫預測目標。這裡以前 64 天價格（除以最後一天）預測下一天（或第 5 天）的價格，MAPE 以價格水準計算。論文的 naive MAPE（S&P 500 為 0.046）比實際每日價格變動大好幾倍，無法對應任何價格水準的定義，因此數值不能直接與論文比較，只能比較模型之間的相對表現。
9. **基準模型**：依附錄 E.2，都在不重疊、長度 21 的報酬片段上訓練。RCGAN 以月份（sin/cos）為條件；TimeGAN 依原始實作的三階段訓練；CSDI 為 unconditional 版本（50 步、quad schedule）。

## 改良版（簡報版）細節

| 項目 | 原始 FTS-Diffusion | 改良版 |
|---|---|---|
| 波動 | 由型態幅度隱含 | GJR-GARCH(1,1)-X，X 為相對成交量 exp(log V − 前 20 日平均)；擴散只學 z = ε / σ |
| SISC 的輸入 | log 價格 | 標準化殘差的累積路徑 Σz |
| 型態演化 | MLP，只看上一個片段 | 因果 Transformer（2 層、d = 64），看過去 16 個片段；token = 型態 + 長度 + 幅度 + 成交量水準 + 形狀嵌入 |
| 長度與幅度 | 單一點預測 | MDN（5 個高斯成分）抽樣 |
| 去噪器 | TCN + scaling autoencoder，兩個損失 | DiT（4 個 adaLN-Zero block），單一擴散損失；固定長度形狀用內插拉伸到抽到的長度 |
| 去噪器條件 | 型態 | 型態、Transformer 歷史狀態、擴散步、抽到的長度與幅度 |
| 成交量 | 不使用 | 作為第二個生成通道；驅動 GARCH 變異數並進入歷史 |
| 報酬還原 | 片段直接拼接 | r_t = μ + σ_t · z_t，σ_t 以 GARCH 遞迴往前模擬 |

為了數值穩定，抽樣時另外加了簡報沒有的處理：

- MDN 輸出截在 ±4σ（原始版的幅度抽樣也套用相同截斷）。
- 生成的成交量截在訓練期範圍內。
- 合成 z 的標準差校準到訓練期，並截在訓練期 z 的最小值與最大值之間。
- GARCH 條件變異數最多為訓練期最大值的 2 倍（`GJRGARCHX.step` 的 `cap`）。

消融實驗：`fts_plus_nogarch`（常數波動）、`fts_plus_novol`（不用成交量）。
