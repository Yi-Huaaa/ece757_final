## Heuristic A：針對 小 workload (4x4 矩陣)
* 特點分析：
	* 計算量小，任務多。
	* Overhead 比重高（scheduling、synchronization）。
	* 傳輸量小，但依賴可能很多（DAG 邊多）。
* Heuristic 策略：
	* Aggressive task merging
		* 將 dependency 強（例如 parent-child）的小 task 合併成同一個 partition。
		* 比方說你做 BFS/DFS 看 DAG 邊緣關係強的地方可以 merge 起來。
	* Balance task count
		* 因為每個 task 都快做完，所以平均分配 task 數是最關鍵的（not workload weight）。
	* Minimize partition switches
		* 盡量讓相鄰 node 留在同一 partition，減少 context switch 和 cache miss。

## Heuristic B：針對 大 workload (64x64 矩陣)
* 特點分析：
	* 計算主導，傳輸 latency 成本提高。
	* 每個 task 跑很久，partition 之間的資料傳輸成本開始主導。
	* Cache locality 和 memory bandwidth 開始重要。

* Heuristic 策略：
	* Edge cut minimization
		* 目標是最小化跨 partition 的邊，因為跨核心傳輸代價高。
		* 使用 greedy 合併：從 leaf node 開始，把相依性強的 task 拉進同一 partition。
	* Critical path aware 分配
		* 找出 DAG 中的 critical path，把它排進一組 partition，避免等待。
	* Coarse-grained partition
		* 大工作量下，過多 partition 沒好處，降低 partition 數反而有助於 cache reuse。

---

## 根據你目前的分析與結果，尤其是：
* G-PASTA 在小 task（小矩陣乘法）下能有效降低 runtime。
* 但在 task workload 變大（如 32×32 矩陣）時，partitioning 反而增加 runtime。
* 而你希望設計 heuristic 讓 partitioning 在大 workload 下依然有效。
* 我會建議你考慮以下幾個 heuristic-based partitioning 改進策略，針對「大工作量 per task」的情況：

* 問題核心
* 在大 workload 下，每個 task 的計算成本遠高於 scheduling 成本，因此：
	* Aggressive partitioning → 降低 parallelism → Overhead 大於 scheduling 減少 → Runtime 增加。
* 所以你的 heuristic 應該：
	* 減少合併（避免破壞 parallelism）
	* 或 調整 partition size 動態適應 task cost
	* 或 根據 critical path 考慮哪裡可以合併、不會延遲整體結束時間
	
---

## Heuristic Ideas（針對大 workload）

### 1. Task Granularity-aware Partitioning
* 根據你輸入的 matrix size 自動調整 partition size：
```
if matrix_size <= 8:
    partition_size = 16  # Aggressive merge to reduce scheduling cost
else:
    partition_size = 1   # 保留 fine-grained 並行性
```
* 簡單但有效，可直接修改 ITAP kernel 的 partitioning threshold。
* 建議修改策略：針對 matrix size 加入動態 partition 調整
* 目前的 G-PASTA 演算法是這樣的邏輯：
1. 固定 partition_size（每個 partition 最多幾個 task）
2. 每一層 BFS 遍歷，用 atomic 計數控制是否換新的 partition
3. 完全與 task 的計算成本（工作量）無關
* 這在小任務很好，但你實驗發現：
1. matrix size 一大，合併太多會變成 破壞 parallelism
2. 而 G-PASTA 不知道 task 是輕還是重，仍然硬合併
* 修改目標：「讓 partition 大小根據 task 的工作量（矩陣大小）動態調整，或甚至跳過 partitioning。」

### 2. Critical-path-aware Heuristic
* 將 critical path 上的 task 優先保持「不合併」狀態，讓非關鍵區域去合併。
* 流程：
	* 用簡單 BFS 找出 DAG 中最長路徑（或估算 critical path）
	* 只合併非 critical path 上的 task
	* 或給予 critical path task 較小 partition size（甚至保留原本 granularity）
* 這樣可以：
	* 保留 execution time 的 upper bound
	* 同時在非關鍵區做 scheduling overhead 的優化

### 3. Workload-cost-aware Cost Model + Partitioning
* 這是用於 heuristic 評估的 cost model，來幫你決定 是否該合併兩個 task：
* 定義：
```
merge_cost(task1, task2) = scheduling_saving - compute_penalty
```
* scheduling_saving: 合併兩個 task 可省下的 scheduling overhead（~2us per task）
* compute_penalty: 合併造成 delay（例如合併後必須順序執行）
* 你可以透過一個 rule：
	* 若 merge_cost < 0 就不要合併
* 你可以用這來建立一個簡單的 greedy 合併演算法。


### 4. Dynamic Partition Size Selection
* G-PASTA 預設是「每個 partition 最大 N 個 task」，你可以根據每個 task 的 workload 調整這個數字：
```
# 目標：讓每個 partition 有 roughly 相等的 computation time，而非相等的 task 數
partition_time_budget = 200 us
partition_size = partition_time_budget / task_compute_time
```
* 若你知道每個 task（如 32×32 矩陣）的估算時間（如 150us），就可以估出 partition 大小
* 這可以實作在 GPU kernel 裡，例如用 compute_weight array 替代 pid_cnt


### 5. Hybrid Partitioning Strategy
* 結合 G-PASTA（適用小 workload）+ Baseline execution（不 partition，大 workload）：
* 實作：
	* 若 matrix size < 16 → 用 G-PASTA partition
	* 若 matrix size ≥ 16 → 不 partition 或只合併 2–3 層 pipeline
* 你甚至可以讓你自己的 heuristic 決定使用哪種 partitioner（像是 runtime selector）。

---

## 小結論
### 小結論討論1
1. 先判斷 problem size
	* 當 problem size 小，就純粹用 G-PASTA
	* 當 problem size 中，要用 hsuristic 1
	* 當 problem size 大，要用 hsuristic 2
2. Heuristic A
	* Critical-path-aware Heuristic: 將 critical path 上的 task 優先保持「不合併」狀態，讓非關鍵區域去合併。(上面的方法二)
3. Heuristic B
	* Workload-cost-aware Cost Model + Partitioning
4. Heuristic C
	* Dynamic Partition Size Selection
5. Heuristic D
	* Hybrid Partitioning Strategy
		* 這可能要先做一些 sample，去看：
			1. taskflow 在什麼狀況比較好
			2. G-PASTA 在什麼狀況比較好
### 小結論討論2
1. taskflow
2. G-PASTA
3. hybrid (tf & GP)
	
### 小結論討論3 - 要做的事情
1. (20250323 週二) 先判斷 problem size
2. 數據統計：taskflow 在什麼狀況比較好、G-PASTA 在什麼狀況比較好
3. 根據這個統計去跑一次 graph （先跑到這裡就好）

