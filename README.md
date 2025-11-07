# 用於柔性作業車間調度與預防性維護的階層式強化學習 (HRL-DQN) 專案

## 1. 專案簡介

本專案為一篇關於聯合優化**柔性作業車間排程 (Flexible Job Shop Scheduling, FJSSP)** 與**預防性維護 (Preventive Maintenance, PM)** 問題的研究的程式碼實現。

專案採用**階層式深度強化學習 (Hierarchical Deep Reinforcement Learning, HRL)** 架構，並以**深度Q網路 (Deep Q-Network, DQN)** 作為主要的學習演算法。其核心目標是訓練一個能同時做出高效生產排程和經濟的維護決策的智慧代理人 (Agent)。

## 2. 主要功能

*   **階層式決策架構**：模型分為兩層。上層的 **Option Agent** 負責高層次的策略選擇（現在應該專注於「生產」還是「維護」？），下層的 **Production Agent** 和 **Maintenance Agent** 則負責執行具體的低層次操作（選擇哪個派工法則？對哪個設備進行維護？）。
*   **事件驅動的模擬環境**：內建一個 `FJSSP_PM_Env` 模擬器，可以模擬工件到達、機台完工、設備老化、執行維護等一系列動態事件。
*   **動態故障率模型**：可以選擇性地載入真實的作業數據 (`ai4i2020.xls`)，根據工件加工時的扭矩、轉速、工具磨損等 `X` 向量，動態地計算和調整設備的故障率，使模擬更貼近真實世界。
*   **基準方法比較**：內建了多種傳統的派工法則（如 SPT, LPT, EDD）和維護策略（如故障才修、固定風險閾值），用於與 HRL 模型的效能進行公平比較。

## 3. 檔案結構

```
hrl_dqn/
│
├── core_shared_mtdqn.py      # 定義了 RL 環境、Agent 模型、資料結構等核心組件
├── train_shared_mtdqn.py     # 主要的訓練腳本
├── run_inference.py          # 使用已訓練好的模型進行測試 (推論)
├── eval_baselines_simple.py  # 評估傳統 Baseline 方法的腳本
│
├── data_adapter.py           # 資料適配器，用於從 ai4i2020.xls 讀取作業條件數據
├── ai4i2020.xls              # (範例) 作業條件數據集 (本質為 CSV)
│
├── checkpoints/              # 存放訓練好的模型檔案 (例如 best.pt)
├── outputs/                  # 存放訓練過程的日誌 (train_log.csv)
└── baseline_output/          # 存放 Baseline 評估結果 (CSV, PNG 圖表)
```

## 4. 安裝教學

本專案基於 Python，需要安裝以下函式庫。建議使用 `pip` 進行安裝。

```bash
pip install torch numpy pandas xlrd matplotlib
```

## 5. 使用方法

所有指令都建議在 `baseline/hrl_dqn/` 目錄下執行。

### 5.1. 訓練 HRL-DQN 模型

以下指令將使用隨機生成的問題實例進行訓練，啟用 `ai4i` 動態故障率，並預設訓練 4000 個回合。

```bash
python train_shared_mtdqn.py --train_dist --ai4i ai4i2020.xls
```
*   模型和日誌會分別儲存在 `checkpoints/` 和 `outputs/` 資料夾。

### 5.2. 測試已訓練的模型

使用 `run_inference.py` 腳本來評估已儲存的最佳模型 (`checkpoints/best.pt`)。

*   **在論文的固定案例上進行測試：**
    ```bash
    python run_inference.py --instance_type full --ai4i ai4i2020.xls
    ```

*   **在特定壓力場景下進行測試 (例如，使用第 100 行的工況)：**
    ```bash
    python run_inference.py --instance_type full --ai4i ai4i2020.xls --specific_row 100
    ```

### 5.3. 評估 Baseline

使用 `eval_baselines_simple.py` 腳本來評估傳統策略。

*   **在論文的固定案例上進行評估 (以獲得公平的比較基準)：**
    ```bash
    python eval_baselines_simple.py --instance_type full --ai4i ai4i2020.xls
    ```

*   **在隨機生成的案例上進行評估：**
    ```bash
    python eval_baselines_simple.py --instance_type random --ai4i ai4i2020.xls --num_instances 50
    ```

## 6. 模型與環境簡介

*   **狀態 (State)**：RL 環境的狀態由一系列特徵組成，包括全局特徵（如逾期工件數、平均負載）和每個機器的獨立特徵（如設備年齡、可靠度、預估的 `T*` 值等）。
*   **動作 (Action)**：
    *   **上層**: 選擇 `0` (生產) 或 `1` (維護)。
    *   **下層 (生產)**: 從 7 種派工法則中選擇一種。
    *   **下層 (維護)**: 選擇要進行預防性維護的機台。
*   **獎勵 (Reward)**：獎勵函數被設計為最小化一個總成本函數，該函數是三個部分的加權和：`TTC` (總延遲成本), `TBC` (總負載均衡成本), 和 `TMC` (總維護成本)。

## 7. 核心設計：事件驅動模擬環境 (Core Design: Event-Driven Simulation)

本專案的模擬環境 (`FJSSP_PM_Env`) 並非採用傳統的、固定時間步長的推進方式，而是基於一個更高效的**事件驅動 (Event-Driven)** 模型。

與其像動畫片一樣一幀一幀地推進時間，事件驅動模型更像一本小說，模擬器的時間線會直接「跳躍」到下一個「重要事件」發生的時間點。在您的專案中，最重要的事件是**「機台完工並被釋放」(Machine Becomes Free)**。

#### 工作流程

1.  **查看事件佇列**: 模擬器內部維護一個按時間排序的「事件行事曆」。它會找到時間最早的那個「機台完工」事件。
2.  **時間跳躍**: 模擬器的當前時間，直接「跳躍」到這個事件發生的時間點，跳過所有無事發生的中間時段。
3.  **處理事件**: 將對應的機台狀態設置為「空閒」，並更新相關工件的進度。
4.  **觸發決策**: 在這個有意義的新時間點，呼叫 HRL Agent 來觀察環境，並做出下一個生產或維護決策。
5.  **產生新事件**: Agent 的新決策會產生未來的完工事件，這些新事件及其預計的完成時間會被添加到「事件行事曆」中。

#### 優點

*   **極高效率**: 由於時間是跳躍式前進的，因此極大地節省了計算資源，讓數千回合的複雜訓練得以在短時間內完成。
*   **決策精準**: 確保了 Agent 只在系統狀態發生實質性變化、需要它做出決策的關鍵時刻才行動，更符合真實世界的決策流程。

## 8. 附錄：詳細方法論 (Methodology Appendix)

本節提供專案所用之強化學習模型的詳細定義，以供論文撰寫參考。

### A. 問題定義 (Problem Formulation)

本研究旨在解決一個集成的柔性作業車間調度與預防性維護 (Joint FJSSP & PM) 問題。該問題的目標是，在一個動態環境中，透過對生產活動和維護活動的聯合決策，最小化系統的期望總成本。

系統的總成本 $TC$ 由三部分構成：總延遲成本 $TTC$、總負載均衡成本 $TBC$ 和總維護成本 $TMC$。

$
\min \mathbb{E} \left[ \sum_{t=0}^{H} (TTC_t + TBC_t + TMC_t) \right]
$

其中 $H$ 為決策時域。

1.  **總延遲成本 (Total Tardiness Cost, TTC)**:
    $
    TTC = \sum_{i \in \mathcal{J}} P_i \cdot \max(0, C_i - D_i)
    $
    其中 $\mathcal{J}$ 為工件集合， $C_i$ 和 $D_i$ 分別為工件 $i$ 的實際完工時間和交期，$P_i$ 為其單位延遲懲罰。

2.  **總負載均衡成本 (Total Balancing Cost, TBC)**:
    $
    TBC = C_b \cdot \sqrt{\frac{1}{|\mathcal{M}|} \sum_{k \in \mathcal{M}} (Z_k - \bar{Z})^2}
    $
    其中 $\mathcal{M}$ 為機台集合，$Z_k$ 為機台 $k$ 的總工作時間，$\bar{Z}$ 為所有機台的平均工作時間，$C_b$ 為均衡成本係數。此成本項旨在懲罰機台間工作負載的標準差，以促進更均衡的設備利用率。

3.  **總維護成本 (Total Maintenance Cost, TMC)**:
    $
    TMC = \sum_{k \in \mathcal{M}} \sum_{j=1}^{N_k} (C_{p,k}^{(j)} + C_{f,k}^{(j)})
    $
    其中 $N_k$ 是機台 $k$ 在時域內的維護次數。對於第 $j$ 次維護週期，其成本由預防性維護的直接成本 $C_{p,k}$ 和該週期內累積的預期故障成本 $C_{f,k}$ 組成。預期故障成本定義為：
    $
    C_{f,k}^{(j)} = C_{f,k} \cdot \int_{0}^{T'_j} \lambda_k(t | \mathbf{X}) dt
    $
    其中 $C_{f,k}$ 是機台 $k$ 發生故障的修正成本，$T'_j$ 是第 $j$ 個維護週期的時長，$\lambda_k(t | \mathbf{X})$ 是考慮了作業條件 $\mathbf{X}$ 的動態故障率。

### B. 動態故障率模型 (Dynamic Failure Rate Model)

機台的故障率並非恆定，而是遵循一個受作業條件影響的**比例風險模型 (Proportional Hazard Model, PHM)**。我們採用韋伯分佈 (Weibull Distribution) 作為基礎故障率 $\lambda_0(t)$。

$
\lambda_k(t | \mathbf{X}) = \lambda_{0,k}(t) \cdot \psi(\mathbf{X}) = \left[ \frac{\beta_k}{\eta_k} \left( \frac{t}{\eta_k} \right)^{\beta_k-1} \right] \cdot \exp(\mathbf{w}_k^T \mathbf{X})
$

*   $\beta_k$ 和 $\eta_k$ 分別是機台 $k$ 的韋伯分佈的形狀 (shape) 和尺度 (scale) 參數。
*   $t$ 是自上次維護以來的設備年齡（累積工作時間）。
*   $\psi(\mathbf{X}) = \exp(\mathbf{w}_k^T \mathbf{X})$ 是風險乘數，它根據一個即時的作業條件向量 $\mathbf{X} = [x_1, x_2, \dots, x_F]$ 和對應的權重向量 $\mathbf{w}_k$ 來動態調整基礎故障率。在本專案中，$\mathbf{X}$ 包括**扭矩、轉速、工具磨損、溫度**等從 `ai4i2020.xls` 數據集中提取的特徵。

### C. 理論最佳維護週期模型 (Optimal PM Interval Model, T*)

`T*` 是 Agent 決策時一個重要的「經濟學參考點」，它代表在當前工況下，能使「單位時間的平均總成本」最低的理論維護間隔。

這個「單位時間的平均總成本」由**成本率 (Cost Rate)** 函數 `VC(T)` 來表示：

$
VC(T) = \frac{\text{週期內總期望成本}}{\text{週期內總期望時間}} = \frac{C_p + C_f \cdot I(T)}{T + T_p + T_f \cdot I(T)}
$

*   **分子 (總期望成本)**：由預防性維護成本 $C_p$ 和週期內的預期故障成本 $C_f \cdot I(T)$ 組成。
*   **分母 (總期望時間)**：由設備正常運行時間 $T$、預防性維護時間 $T_p$ 和預期故障停機時間 $T_f \cdot I(T)$ 組成。
*   $I(T) = \int_{0}^{T} \lambda(t | \mathbf{X}) dt$ 是在時間 $T$ 內的累積故障機率。

`T*` 的計算目標，就是找到那個能讓 `VC(T)` 最小化的 `T` 值：
$
T^* = \arg\min_{T} VC(T)
$

由於直接求解該公式非常複雜，程式碼 (`_find_T_star_numeric` 函式) 採用了**兩階段數值搜索 (Two-stage Numerical Search)** 的方法來尋找近似最優解：先在一個大範圍內進行粗略搜索，然後在找到的低成本區域內進行精細搜索。

最關鍵的是，由於 `VC(T)` 的計算依賴於動態故障率 $\lambda(t | \mathbf{X})$，因此 `T*` 的值會在每個決策點被**動態更新**，為 Agent 提供即時的、符合當前工況的經濟學建議。

### D. 階層式強化學習框架 (HRL Framework)

為處理複雜的聯合決策空間，我們設計了一個兩層的 HRL 框架。

#### C.1. 上層元控制器 (Meta-Controller / Option Agent)

*   **職責**: 在每個決策時刻，進行高層次的策略選擇：執行「生產」或「維護」。
*   **狀態空間 $\mathcal{S}^{upper}$**: 包含最全面的系統資訊，以做出宏觀決策。
    *   **全局特徵 (Global Features)**:
        *   `norm_num_tardy_jobs` (已逾期工件數正規化值): **緊急信號**。代表已確定會延遲的工件比例，反映了生產進度是否嚴重落後。
        *   `due_ratio` (平均交期比率): **履約能力趨勢**。所有已完工工件的「實際完工時間 / 預計交期」的平均值，預示著未來的延遲風險。
        *   `mean_machine_load` (平均負載): **系統壓力指標**。所有機器的平均利用率，代表整個系統的繁忙程度。
        *   `std_machine_load` (負載標準差): **系統均衡性指標**。負載的標準差，值越高代表負載越不均衡，部分機器可能過勞。
        *   `CRO` (總工序完成率): **任務總進度指標**。計算方式為「已完成的總工序數 / 全部工件的總工序數」，告訴 Agent 整個生產計劃進行到了哪個階段。
        *   `max_failure_risk` (最大故障風險): **預警信號**。代表系統中「最薄弱的一環」，取所有機器中最高的預期故障率 `lambda` 值，是觸發維護決策的關鍵。
    *   **各機台特徵 (Per-Machine Features)**:
        *   `age/scale` (設備年齡): **老化程度指標**。機台自上次維護以來的累計工作時間，並進行正規化。越老的設備，故障風險越高。
        *   `lambda` (動態故障率): **核心風險指標**。結合了「老化」和「操勞」（作業條件 `X`）的當前瞬間故障機率。
        *   `R` (可靠度): **歷史累積風險指標**。機台從上次維護至今仍然正常的機率，是對 `lambda` 的時間積分。
        *   `Tst` (理論最佳維護週期 `T*`): **經濟學參考點**。從純經濟角度計算出的、能使單位時間成本最小化的理論維護間隔。
        *   `mult` (故障率乘數): **作業壓力指標**。將作業條件 `X` 的影響單獨剝離出來的風險乘數，代表當前任務對設備的損耗速度。
        *   `load` (負載): **繁忙度指標**。該機台的歷史利用率，是判斷系統瓶頸和負載均衡狀況的依據。
*   **動作空間 $\mathcal{A}^{upper}$**: $a_t^O \in \{ \text{PRODUCTION}, \text{MAINTENANCE} \}$。
*   **獎勵函數 $\mathcal{R}^{upper}$**: 採用全局成本作為獎勵信號，旨在最大化回報（即最小化總成本）。
    $
    R_t^{upper} = -(TTC_t + TBC_t + TMC_t)
    $

#### C.2. 下層控制器 (Low-level Controllers)

##### C.2.1. 生產控制器 (Production Controller)

*   **職責**: 當上層選擇「生產」時，從一組預定義的派工法則中選擇一個來執行。
*   **狀態空間 $\mathcal{S}^{prod}$**: 包含與生產緊密相關的資訊，如逾期工件數、交期比率、工序完成率和各機台負載。
*   **動作空間 $\mathcal{A}^{prod}$**: $a_t^P \in \{ \text{SPT, LPT, MWKR, RAND} \}$。
*   **獎勵函數 $\mathcal{R}^{prod}$**: 採用差異化獎勵，僅關注其決策對生產相關成本的**邊際影響**。
    $
    R_t^{prod} = -\left( (TTC_t - TTC_{t-1}) + \lambda_{tbc} \cdot (TBC_t - TBC_{t-1}) \right)
    $
    其中 $\lambda_{tbc}$ 是 TBC 成本的權重。

##### C.2.2. 維護控制器 (Maintenance Controller)

*   **職責**: 當上層選擇「維護」時，決定具體要對哪一台機器執行預防性維護。
*   **狀態空間 $\mathcal{S}^{maint}$**: 包含與維護緊密相關的資訊，如系統最大故障風險和各機台的設備年齡、故障率、可靠度等。
*   **動作空間 $\mathcal{A}^{maint}$**: $a_t^M \in \{1, 2, \dots, |\mathcal{M}|\}$，選擇機台 $k$ 進行維護。
*   **獎勵函數 $\mathcal{R}^{maint}$**: 與上層共享全局獎勵信號 $R_t^{maint} = R_t^{upper}$。

### E. 學習演算法 (Learning Algorithm)

每個 Agent 都採用**深度Q網路 (DQN)** 進行學習。對於一個給定的 Agent，其目標是學習一個動作價值函數 (Q-function) $Q(s, a; \theta)$，其中 $\theta$ 是神經網路的參數。

Q-function 的更新遵循貝爾曼方程，其損失函數定義為：
$
L(\theta) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \left[ (y - Q(s, a; \theta))^2 \right]
$
其中 $\mathcal{D}$ 是從經驗回放池 (Experience Replay Buffer) 中採樣的數據。

目標值 $y$ (TD-Target) 的計算結合了**目標網路 (Target Network)** 以增加訓練穩定性：
$
y = r + \gamma \cdot \max_{a'} Q(s', a'; \theta^-)
$
其中 $\gamma$ 是折扣因子，$\theta^-$ 是目標網路的參數，它會定期從主網路的參數 $\theta$ 進行軟更新或硬更新。

本專案採用了**共享編碼器、多頭輸出 (Shared-Encoder, Multi-Head)** 的網路架構，三個 Agent 的網路共享一部分底層的編碼器層，但擁有各自獨立的輸出頭 (Q-value Head)，以在提取共通特徵的同時，學習各自的決策策略。
