import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sko.PSO import PSO

# 全局变量
with open("config.json", "r") as f:
    config = json.load(f)
P_r = config['installedPowerCapacity']  # 光伏电厂额定功率 Mw


# 绘制24小时时间曲线
def draw(*args):
    # 将横坐标设置为从 0 到 86400 秒
    time = np.arange(0, 86400, 1)
    # 将横坐标的标签设置为小时数
    xtick_positions = np.arange(0, 86400, 3600)  # 将横坐标分成 24 份，每份 3600 秒
    xtick_labels = np.arange(0, 24, 1)  # 将标签设置为小时数，从 0 到 24
    plt.xticks(xtick_positions, xtick_labels)

    for arg in args:
        print(arg)
        plt.plot(time, arg[0], label=arg[1])
    plt.legend()
    plt.show()


# 数据预处理，返回秒级光伏日输出曲线
def init_data():
    global config
    # 读取指定发电输出曲线文件
    df = pd.read_excel(config['outputCurvePath'], sheet_name='Sheet1')

    # 提取第三行及以后，第二列的数据，存入向量中
    data = [row[1] for i, row in df.iloc[2:].iterrows()]

    # 将分钟级数据扩展为秒级数据
    if not config['secData']:
        data = [num for num in data for _ in range(60)]

    # 填充不满24h的数据（默认为0）
    return np.pad(data, (0, 86400 - len(data)), 'constant', constant_values=(0, 0))


# TODO: 平抑光伏曲线
# 满足一分钟内不超过10%，半小时内不超过30%的频率波动
def smooth_curve(p_pv):
    gamma_1min = 0.1
    gamma_30min = 0.3

    print(P_r)

    p_o = p_pv.copy()
    for t in range(1800):
        p_o[t] = p_pv[1800]
    for t in range(60, len(p_o)):
        # 1. 平抑曲线，使得满足一分钟不超过10%
        delta_p_o_1min = (np.max(p_o[t - 59:t + 1]) - np.min(p_o[t - 59:t + 1])) / P_r
        if delta_p_o_1min < gamma_1min:
            a_1 = 1
        elif p_pv[t] > p_o[t - 1]:
            a_1 = (gamma_1min * P_r + np.min(p_o[t - 59:t]) - p_o[t - 1]) / (p_pv[t] - p_o[t - 1])
        elif p_pv[t] < p_o[t - 1]:
            a_1 = (-gamma_1min * P_r + np.max(p_o[t - 59:t]) - p_o[t - 1]) / (p_pv[t] - p_o[t - 1])
        else:
            a_1 = 1

        # 2. 平抑曲线，使得满足三十分钟不超过30%
        # TODO: 三十分钟的平抑算法有待改进
        p_o[t] = (1 - a_1) * p_o[t - 1] + a_1 * p_pv[t]
        p_o_1min_s = [np.sum(p_o[t - 60 + 1 - s * 60:t - s * 60 + 1]) / 60 for s in range(30)]
        delta_p_o_30min = (np.max(p_o_1min_s) - np.min(p_o_1min_s)) / P_r
        if delta_p_o_30min < gamma_30min:
            a_2 = 1
        elif p_o_1min_s[0] > p_o_1min_s[1]:
            a_2 = (60 * (gamma_30min * P_r + np.min(p_o_1min_s[1:])) - np.sum(p_o[t - 59:t]) - p_o[t - 1]) / (
                    p_pv[t] - p_o[t - 1])
        elif p_o_1min_s[0] < p_o_1min_s[1]:
            a_2 = (60 * (-gamma_30min * P_r + np.max(p_o_1min_s[1:])) - np.sum(p_o[t - 59:t]) - p_o[t - 1]) / (
                    p_pv[t] - p_o[t - 1])
        else:
            a_2 = 1

        a_2 = np.clip(a_2, 0, 1)
        p_o[t] = (1 - a_2) * p_o[t - 1] + a_2 * p_pv[t]

        # 3. 限值附近的平缓处理
        # TODO: 貌似没有作用，且没有理解论文的公式
        delta_p_o_30min = (np.max(p_o_1min_s) - np.min(p_o_1min_s)) / P_r

        if delta_p_o_30min < 0.9 * gamma_30min:
            delta_p_o_t_1 = 0.1 * gamma_1min + 0.9 * gamma_1min * np.cos(np.pi / 18 * delta_p_o_30min)
        elif delta_p_o_30min < gamma_30min:
            delta_p_o_t_1 = 0.1 * gamma_1min + 0.1 * gamma_1min * np.cos(np.pi / 2 * delta_p_o_30min)
        else:
            delta_p_o_t_1 = 0
        p_o[t] = p_o[t] + delta_p_o_t_1

    return p_o


# 分频算法，将输入曲线分为高频和低频两个部分
# 目前使用的是 NFT 分频
def subsampling_algorithm(curve):
    # TODO: 可以加入对n的调参
    n = 46
    X = np.fft.fft(curve)
    x_low_freq = X.copy()
    x_high_freq = X.copy()
    x_low_freq[n:] = 0
    low_freq = np.real(np.fft.ifft(x_low_freq))
    x_high_freq[:n] = 0
    high_freq = np.real(np.fft.ifft(x_high_freq))
    return high_freq, low_freq


class CapacityAllocation:
    def __init__(self, sc_output_curve, b_output_curve):
        self.life_span = config['life_span']

        self.b_soc_max = config['battery']['socMax']  # 蓄电池SOC上界
        self.b_soc_min = config['battery']['socMin']  # 蓄电池SOC下界
        self.b_power_cost = config['battery']['powerCost'] * 1000  # 蓄电池功率单价 RMB/Mw
        self.b_capacity_cost = config['battery']['capacityCost'] * 1000  # 蓄电池容量单价 RMB/Mwh
        self.b_operation_cost = config['battery']['operationCost'] * 1000  # 蓄电池维护单价 RMB/Mwh/h
        self.b_cycle_limit = config['battery']['cycleLimit'] = 3000  # 蓄电池最大循环次数

        self.sc_soc_max = config['capacitor']['socMax']  # 超级电容SOC上界
        self.sc_soc_min = config['capacitor']['socMin']  # 超级电容SOC下界
        self.sc_power_cost = config['battery']['powerCost'] * 1000  # 超级电容功率单价 RMB/Mw
        self.sc_capacity_cost = config['battery']['capacityCost'] * 1000  # 超级电容容量单价 RMB/Mwh
        self.sc_operation_cost = config['battery']['operationCost'] * 1000  # 超级电容维护单价 RMB/Mwh/h
        self.sc_cycle_limit = config['battery']['cycleLimit'] = 3000  # 超级电容最大循环次数

        self.b_capacity_curve = np.cumsum(b_output_curve) / 3600  # 蓄电池储能曲线 Mwh
        self.b_daily_capacity = np.sum(np.abs(np.diff(self.b_capacity_curve)))  # 蓄电池一天的储能变化 Mwh/天
        self.b_daily_capacity_end = self.b_capacity_curve[-1]  # 蓄电池在一天结束时的储能
        self.sc_capacity_curve = np.cumsum(sc_output_curve) / 3600  # 超级电容储能曲线 Mwh
        self.sc_daily_capacity = np.sum(np.abs(np.diff(self.sc_capacity_curve)))  # 超级电容一天的储能变化 Mwh/天
        self.sc_daily_capacity_end = self.sc_capacity_curve[-1]  # 超级电容在一天结束时的储能

        self.b_max_power = max(b_output_curve)  # 蓄电池最大功率 Mw
        self.sc_max_power = max(sc_output_curve)  # 超级电容最大功率 Mw

        self.peak_electricity_price = config['peakElectricityPrice'] * 1000  # 峰电价格 RMB/Mwh
        self.off_peak_electricity_price = config['offPeakElectricityPrice'] * 1000  # 谷电价格 RMB/Mwh

    # 成本估算函数
    def calc_cost(self, x):
        b_power, b_capacity, sc_power, sc_capacity = x
        # 蓄电池一天的循环次数
        b_daily_soc = self.b_daily_capacity / b_capacity + (self.b_daily_capacity_end / b_capacity + 0.5) + 0.5 - self.b_soc_min * 2
        # 超级电容一天的循环次数
        sc_daily_soc = self.sc_daily_capacity / sc_capacity + (self.sc_daily_capacity_end / sc_capacity + 0.5) + 0.5 - self.sc_soc_min * 2
        # 蓄电池总计更换次数
        b_replacement_times = 365 * self.life_span * (b_daily_soc / self.b_cycle_limit)
        # 超级电容总计更换次数
        sc_replacement_times = 365 * self.life_span * (sc_daily_soc / self.sc_cycle_limit)

        # 设计建设成本
        c0 = self.b_power_cost * b_power + self.b_capacity_cost * b_capacity + \
             self.sc_power_cost * sc_power + self.sc_capacity_cost * sc_capacity
        # 维护成本
        c1 = 365 * 24 * self.life_span * (self.b_operation_cost * b_capacity + self.sc_operation_cost * sc_capacity)
        # 更换成本
        c2 = (self.b_power_cost * b_power + self.b_capacity_cost * b_capacity) * b_replacement_times + \
             (self.sc_power_cost * sc_power + self.sc_capacity_cost * sc_capacity) * sc_replacement_times
        # 峰谷差价利润
        c3 = 365 * self.life_span * (((self.b_daily_capacity_end / b_capacity + 0.5) * b_capacity + (self.sc_daily_capacity_end / sc_capacity + 0.5) * sc_capacity) * self.peak_electricity_price - ((0.5 - self.b_soc_min) * b_capacity + (0.5 - self.sc_soc_min) * sc_capacity) * self.off_peak_electricity_price)

        # TODO: 先不考虑年利率
        return (c0 + c1 + c2 - c3) / self.life_span

    # 储能容量配置
    def energy_storage_capacity_allocation(self):
        lb = [self.b_max_power, max(min(self.b_capacity_curve) / (self.b_soc_min - 0.5),
                                    max(self.b_capacity_curve) / (self.b_soc_max - 0.5)),
              self.sc_max_power, max(min(self.sc_capacity_curve) / (self.sc_soc_min - 0.5),
                                     max(self.sc_capacity_curve) / (self.sc_soc_max - 0.5))]  # 下界
        ub = [1e18] * 4  # 上界
        print(lb, ub)
        c1 = 1.6  # 个体学习因子
        c2 = 1.6  # 群体学习因子
        w = 0.55  # 惯性权重
        n_particles = 40  # 种群粒子数量
        n_iterations = 300  # 最大迭代次数

        pso = PSO(func=self.calc_cost, n_dim=4, pop=n_particles, max_iter=n_iterations, lb=lb, ub=ub, w=w, c1=c1, c2=c2)
        pso.run()
        # TODO: 展示动画
        return pso.gbest_x, pso.gbest_y


# 程序入口
if __name__ == '__main__':
    # 1. 预处理光伏日输出曲线
    rawOutputCurve = init_data()  # 平抑前的出力曲线

    # 2. 平抑光伏日输出曲线，获得平抑后的出力曲线
    smoothOutputCurve = smooth_curve(rawOutputCurve)  # 平抑后的出力曲线

    # 3. 对平抑后的出力曲线进行分频，分出功率型出力曲线和容量型出力曲线
    # powerOutputCurve => 功率型出力曲线
    # capacityOutputCurve => 容量型出力曲线
    powerOutputCurve, capacityOutputCurve = subsampling_algorithm(rawOutputCurve - smoothOutputCurve)
    draw([rawOutputCurve - smoothOutputCurve, 'target'], [powerOutputCurve, 'power'], [capacityOutputCurve, 'capacity'])

    # 4. 计算最优储能容量配置

    Model = CapacityAllocation(powerOutputCurve, capacityOutputCurve)
    [bPower, bCapacity, scPower, scCapacity], yearCost = Model.energy_storage_capacity_allocation()
    print("蓄电池{}Mw, {}Mwh; 超级电容{}Mw, {}Mwh; 总费用: {}元".format(bPower, bCapacity, scPower, scCapacity, yearCost))
    # draw([rawOutputCurve, 'raw'], [smoothOutputCurve, 'smooth'], [smoothOutputCurve-rawOutputCurve, 'target'])
