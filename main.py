import json
import time
import cProfile
from scipy import signal
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
from numba import jit
from typing import Dict
from sko.PSO import PSO

# 全局变量
config: Dict[str,any]
with open("config.json", "r") as f:
    config = json.load(f)
output_curve_path = config['outputCurvePath']
sec_data = config['secData']
P_r = config['installedPowerCapacity']  # 光伏电厂额定功率 Mw
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 关闭默认使用减号字符


# 绘制24小时时间曲线
def draw(*args, title=None, x_label=None, y_label=None, y_ticks=None, y_lim=None):
    # 将横坐标设置为从 0 到 86400 秒
    time = np.arange(0, 86400, 1)
    # 将横坐标的标签设置为小时数
    xtick_positions = np.arange(0, 86400, 3600)  # 将横坐标分成 24 份，每份 3600 秒
    xtick_labels = np.arange(0, 24, 1)  # 将标签设置为小时数，从 0 到 24
    plt.xticks(xtick_positions, xtick_labels)

    if y_ticks:
        plt.yticks(y_ticks[0], y_ticks[1])

    for arg in args:
        plt.plot(time, arg[0], label=arg[1])
    plt.legend()
    if title:
        plt.title(title)
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if y_lim:
        plt.ylim(y_lim[0], y_lim[1])
    plt.show()


def draw_bar(*args, title=None, x_label=None, y_label=None):

    # 设置偏移量，使得柱形图能够依次排列
    num_curve = len(args)
    print("offset={}".format(np.linspace(-0.1 * (num_curve - 1) / 2, 0.1 * (num_curve - 1) / 2, num_curve)))
    offset = iter(np.linspace(-0.3 * (num_curve - 1) / 2, 0.3 * (num_curve - 1) / 2, num_curve))
    for arg in args:
        print(arg[0])
        curve = arg[0]
        # 将秒级数据变成分钟级
        hourly_data = [sum(curve[i*3600:i*3600+3600] / 3600) for i in range(24)]
        # 绘制图像
        xtick_positions = np.arange(0, 24, 1)  # 将横坐标分成 24 份
        xtick_labels = np.arange(0, 24, 1)  # 将标签设置为小时数，从 0 到 24
        plt.xticks(xtick_positions, xtick_labels)
        plt.bar(xtick_positions + next(offset), hourly_data, width=0.3*(num_curve-1), label=arg[1], align="center")
    plt.legend()
    if x_label:
        plt.xlabel(x_label)
    if y_label:
        plt.ylabel(y_label)
    if title:
        plt.title(title)
    plt.show()


# 数据预处理，返回秒级光伏日输出曲线
def init_data():
    # 读取指定发电输出曲线文件
    df = pd.read_csv(output_curve_path)

    # 提取第三行及以后，第二列的数据，存入向量中
    data = df.iloc[:, 1].values

    # 将分钟级数据扩展为秒级数据
    if not sec_data:
        data = [num for num in data for _ in range(60)]

    # 填充不满24h的数据（默认为0）
    return np.pad(data, (0, 86400 - len(data)), 'constant', constant_values=(0, 0))


# 满足一分钟内不超过10%，半小时内不超过30%的频率波动
def calc_smooth_curve(p_pv):

    first_non_zero = next((i for i, x in enumerate(p_pv) if x != 0), None)
    last_non_zero = next((len(p_pv) - i - 1 for i, x in enumerate(reversed(p_pv)) if x != 0), None)

    f_c_max = 0.001
    f_c_min = 0.000001
    iteration_times = 5
    for i in range(iteration_times):
        f_c = (f_c_max + f_c_min) / 2
        # 低通滤波
        b, a = signal.butter(2, f_c, 'low')  # 二阶低通滤波器
        p_o_filtered = signal.filtfilt(b, a, p_pv)
        p_o_filtered = np.concatenate((np.zeros(first_non_zero), p_o_filtered[first_non_zero:last_non_zero], np.zeros(len(p_pv) - last_non_zero)))
        if judge_satisfy(p_o_filtered, first_non_zero, last_non_zero):
            f_c_min = f_c
        else:
            f_c_max = f_c
    return p_o_filtered


# 判断当前曲线是否全部满足并网要求
@jit(nopython=True)
def judge_satisfy(p_pv, first_non_zero, last_non_zero):
    gamma_1min = 0.1
    gamma_30min = 0.3

    # p_pv_minute 预处理前一分钟平均功率，优化推导速度
    p_pv_sum = np.cumsum(p_pv)  # 预处理功率前缀和
    # p_pv_minute = [0 if t < 60 else (p_pv_sum[t] - p_pv_sum[t - 60]) / 60 for t in range(len(p_pv))]

    for t in range(first_non_zero, last_non_zero):
        p_o_1min_s = [(p_pv_sum[t - s * 60] - p_pv_sum[t - s * 60 - 60]) / 60 for s in range(30)]
        # p_o_1min_s = [p_pv_minute[t - s * 60] for s in range(30)]
        delta_p_o_30min = (max(p_o_1min_s) - min(p_o_1min_s)) / P_r
        if not delta_p_o_30min < gamma_30min:
            return False
    for t in range(first_non_zero, last_non_zero):
        delta_p_o_1min = (max(p_pv[t - 59:t + 1]) - min(p_pv[t - 59:t + 1])) / P_r
        if not delta_p_o_1min < gamma_1min:
            return False
    return True


# 计算有哪些时间段不满足并网需求
def calc_satisfy(p_pv):
    first_non_zero = next((i for i, x in enumerate(p_pv) if x != 0), None)
    last_non_zero = next((len(p_pv) - i - 1 for i, x in enumerate(reversed(p_pv)) if x != 0), None)
    return calc_satisfy_main(p_pv, first_non_zero, last_non_zero)


@jit(nopython=True)
def calc_satisfy_main(p_pv, first_non_zero, last_non_zero):
    gamma_1min = 0.1
    gamma_30min = 0.3

    # p_pv_minute 预处理前一分钟平均功率，优化推导速度
    p_pv_sum = np.cumsum(p_pv)  # 预处理功率前缀和

    # p_o 标记哪些时间可以并网
    p_o = np.ones(len(p_pv))

    for t in range(first_non_zero, last_non_zero):
        p_o_1min_s = [(p_pv_sum[t - s * 60] - p_pv_sum[t - s * 60 - 60]) / 60 for s in range(30)]
        delta_p_o_30min = (max(p_o_1min_s) - min(p_o_1min_s)) / P_r
        delta_p_o_1min = (max(p_pv[t - 59:t + 1]) - min(p_pv[t - 59:t + 1])) / P_r
        if not (delta_p_o_30min < gamma_30min and delta_p_o_1min < gamma_1min):
            p_o[t] = 0

    return p_o


# 分频算法，将输入曲线分为高频和低频两个部分
# 目前使用的是 NFT 分频
def subsampling_algorithm(curve):
    # TODO: 可以加入对n的调参
    n = 46  # 截断频率

    # 快速傅里叶变换
    curve_inv = np.fft.fft(curve)

    # 截取低频部分
    x_low_freq = curve_inv.copy()
    x_low_freq[n:] = 0
    low_freq = np.real(np.fft.ifft(x_low_freq))

    # 截取高频部分
    x_high_freq = curve_inv.copy()
    x_high_freq[:n] = 0
    high_freq = np.real(np.fft.ifft(x_high_freq))

    # 原曲线为0的部分，保证生成的曲线也为0
    low_freq = np.where(curve == 0, 0, low_freq)
    high_freq = np.where(curve == 0, 0, high_freq)

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
        c3 = 365 * self.life_span * (((self.b_daily_capacity_end / b_capacity - self.b_soc_min) * b_capacity + (self.sc_daily_capacity_end / sc_capacity - self.sc_soc_min) * sc_capacity) * self.peak_electricity_price - ((0.5 - self.b_soc_min) * b_capacity + (0.5 - self.sc_soc_min) * sc_capacity) * self.off_peak_electricity_price)

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


# 绘制储能频率曲线
def plt_power_curve(b_curve, sc_curve):
    draw([sc_curve, '功率型'], [b_curve, '容量型'], title='储能频率曲线', x_label='时间(h)', y_label='功率(Mw)')


# 绘制储能SOC曲线
def plt_soc_curve(b_soc_curve, sc_soc_curve):
    draw([sc_soc_curve, '功率型'], [b_soc_curve, '容量型'], title='储能SOC曲线', x_label='时间', y_label='百分比(%)', y_lim=[0, 1])


# 绘制出力曲线
def plt_output_curve(raw_curve, smooth_curve):
    draw([raw_curve, '纯光伏'], [smooth_curve, '光储'], title='出力曲线', x_label='时间(h)', y_label='功率(Mw)')


# 绘制发电量柱形图
def plt_energy_curve(raw_curve, smooth_curve):
    draw_bar([raw_curve, '纯光伏'], [smooth_curve, '光储'], title='发电量', x_label='时间(h)', y_label='能量(Mwh)',)


# 绘制并网时间（即满足并网需求的时间段）
def plt_output_time(raw_curve, smooth_curve):
    print(smooth_curve)
    draw([raw_curve, '纯光伏'], [smooth_curve + 2, '光储'], title="并网时间", x_label="时间(h)", y_label="是否并网", y_ticks=[[0, 1, 2, 3], ['不并网','并网','不并网','并网']])


# 绘制CO2减排量柱形图
def plt_co2_reduce(raw_energy_curve, smooth_energy_curve):
    k = 0.99  # 标准煤发电的 CO2 排放量约为 0.99t/Mwh
    draw_bar([raw_energy_curve * k, '纯光伏'], [smooth_energy_curve * k, '光储'], title="CO2减排量", x_label="时间(h)", y_label="CO2减排量(t)")


# 获取容量型储能配置
def get_capacity_config(b_power, b_capacity):
    print("容量型储能配置")
    print("功率：{{}}".format(b_power))
    print("容量：{{}}".format(b_capacity))


# 获取功率型储能配置
def get_power_config(sc_power, sc_capacity):
    print("功率型储能配置")
    print("功率：{{}}".format(sc_power))
    print("容量：{{}}".format(sc_capacity))


# 获取日效益
def get_daily_benefit():
    # TODO: 待完善算法
    print("日效益")
    return None


# 获取生命周期效益
def get_life_span_benefit():
    # TODO: 待完善算法
    return None


# 计算实时发电数据
def calc_realtime_power_data(power_curve):
    realtime_v = np.zeros(len(power_curve))  # 实时电压
    realtime_a = np.zeros(len(power_curve))  # 实时电流
    realtime_power = power_curve  # 实时功率
    energy_curve = np.cumsum(power_curve) / 3600  # 累计发电量
    realtime_energy = [energy_curve[i] if i < 3600 else energy_curve[i] - energy_curve[i-3600] for i in range(len(energy_curve))]  # 实时发电量（当前一小时发电量）
    # 实时电压，实时电流，实时功率，实时发电量，累计发电量
    return realtime_v, realtime_a, realtime_power, realtime_energy, energy_curve


# 获取实时储能型/容量型数据
def calc_realtime_bsc_data(b_soc_curve, sc_soc_curve, b_power_curve, sc_power_curve):
    b_state_curve = ["-" if x == 0 else "放电" if x > 0 else "充电" for x in b_power_curve]
    sc_state_curve = ["-" if x == 0 else "放电" if x > 0 else "充电" for x in sc_power_curve]
    b_energy_curve = np.cumsum(b_power_curve) / 3600
    sc_energy_curve = np.cumsum(sc_power_curve) / 3600
    # 储能型实时SOC，储能型充放电状态，储能型实时功率，储能型累计充放电量，功率型实时SOC，功率型充放电状态，功率型实施功率，功率型累计充放电量
    return b_soc_curve, b_state_curve, b_power_curve, b_energy_curve, sc_soc_curve, sc_state_curve, sc_power_curve, sc_energy_curve


# 获取发电实时数据
def get_realtime_power_data(curve, t):

    print("发电实时数据")
    print("电压：N/A")
    print("电流：N/A")
    print("功率：{}".format(curve(t)))


# 获取储能实时数据
def get_realtime_store_data(curve, t):
    print("储能实时数据")
    print("电压：N/A")
    print("电流：N/A")
    print("功率：{}".format(curve(t)))


# 程序入口
if __name__ == '__main__':

    start_time = time.time()
    # 1. 预处理光伏日输出曲线
    rawOutputCurve = init_data()  # 平抑前的出力曲线
    start_time1 = time.time()
    rawValidCurve = calc_satisfy(rawOutputCurve)   # 平抑前(纯光伏)并网时间曲线
    end_time1 = time.time()
    print(end_time1 - start_time1)

    # 2. 平抑光伏日输出曲线，获得平抑后的出力曲线
    smoothOutputCurve = calc_smooth_curve(rawOutputCurve)  # 平抑后的出力曲线
    # smoothValidCurve = calc_satisfy(smoothOutputCurve)   # 平抑后(光储)并网时间曲线
    smoothValidCurve = np.ones(len(smoothOutputCurve))
    # cProfile.run('calc_smooth_curve(rawOutputCurve)')

    # 3. 对平抑后的出力曲线进行分频，分出功率型出力曲线和容量型出力曲线
    # powerOutputCurve => 功率型出力曲线
    # capacityOutputCurve => 容量型出力曲线
    powerOutputCurve, capacityOutputCurve = subsampling_algorithm(rawOutputCurve - smoothOutputCurve)
    rawEnergyCurve = np.cumsum(rawOutputCurve) / 3600  # 纯光伏并网曲线
    smoothEnergyCurve = np.cumsum(smoothOutputCurve) / 3600  # 光储并网曲线
    # draw([rawOutputCurve - smoothOutputCurve, 'target'], [powerOutputCurve, 'power'], [capacityOutputCurve, 'capacity'])

    # 4. 计算最优储能容量配置

    Model = CapacityAllocation(powerOutputCurve, capacityOutputCurve)
    [bPower, bCapacity, scPower, scCapacity], yearCost = Model.energy_storage_capacity_allocation()
    bSocCurve = np.cumsum(capacityOutputCurve) / 3600 / bCapacity + 0.5
    scSocCurve = np.cumsum(powerOutputCurve) / 3600 / scCapacity + 0.5
    print("蓄电池{}Mw, {}Mwh; 超级电容{}Mw, {}Mwh; 总费用: {}元".format(bPower, bCapacity, scPower, scCapacity, yearCost))

    # 5. 计算实时数据
    calc_realtime_bsc_data(bSocCurve, scSocCurve, capacityOutputCurve, powerOutputCurve)
    calc_realtime_power_data(smoothOutputCurve)

    # plt_co2_reduce(rawOutputCurve, smoothOutputCurve)
    # plt_output_time(rawValidCurve, smoothValidCurve)
    # plt_soc_curve(bSocCurve, scSocCurve)
    # plt_energy_curve(rawOutputCurve, smoothOutputCurve)
    # plt_output_curve(rawOutputCurve, smoothOutputCurve)
    # plt_power_curve(capacityOutputCurve, powerOutputCurve)

    end_time = time.time()
    run_time = end_time - start_time
    print("程序运行时间：", run_time, "秒")
