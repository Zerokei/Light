# 光伏混合储能系统

## 输入数据

- `example.csv`：分钟级数据
- `example_sec.csv`：秒级数据
  - 装机容量：50Mw
- `example_139.csv`：第139天的秒级数据
  - 装机容量：5000Mw
- `example_201.csv`：第201天的秒级数据
  - 装机容量：4000Mw

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
  - `powerCost`: 功率成本（元/kW）
  - `capacityCost`: 容量成本（元/kWh）
  - `oeprationCost`: 运维成本（元/kWh）
  - `cycleLimit`: 循环寿命（次）
  - `salvageValue`: 回收残值
- `capacitor`: 电容器（功率性储能装置）
  - 其他所有的子参数和 `battery` 一致
- `peakElectricityPrice`: 峰值电价
- `offPeakElectricityPrice`: 谷值电价
- `gridTraiff`: 并网电价
- `frequencyResponseCompensation`: 调频补偿
- `confidenceInterval`: 置信区间
- `gridPrice`: 并网价格

## main.py 关键函数解释

- `plt_poewr_curve`: 储能频率曲线
- `plt_soc_curve`: 储能SOC曲线
- `plt_output_curve`: 出力曲线
- `plt_energy_curve`: 发电量柱形图
- `plt_output_time`: 并网时间曲线
- `plt_co2_reduce`: CO2减排量柱形图
- `get_capacity_config`: 获取容量型储能配置
- `get_realtime_power_data`: 获取发电实时数据
- `get_realtime_store_data`: 获取储能实时数据
- `class CapacityAllocation`:
  - `get_daily_benefit`: 获取日效益
  - `get_lifespan_benefit`: 获取生命周期效益
  - `get_b_para`: 获取容量型储能配置
  - `get_sc_para`: 获取功率型储能配置