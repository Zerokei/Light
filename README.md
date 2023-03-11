
## config 参数解释

- `outputCurvePath`: 光伏发电输出曲线
  - `data/example.xlsx`: 分钟级数据
  - `data/example_sec.xlsx`: 秒级数据
- `secData`: 是否是秒级数据
- `life_span`: 光伏电站生命周期（年）
- `battery`: 电池（容量型储能装置）
  - `socMax`: SOC上限
  - `socMin`: SOC下限
  - `efficiency`: 发电效率
  - `powerCost`: 循环寿命（次）
  - `capacityCost`: 功率成本（元/kW）
  - `oeprationCost`: 容量成本（元/kWh）
  - `cycleLimit`: 运维成本（元/kWh）
  - `salvageValue`: 回收残值
- `capacitor`: 电容器（功率性储能装置）
  - 其他所有的子参数和 `battery` 一致
- `peakElectricityPrice`: 峰值电价
- `offPeakElectricityPrice`: 谷值电价
- `gridTraiff`: 并网电价
- `frequencyResponseCompensation`: 调频补偿
- `confidenceInterval`: 置信区间