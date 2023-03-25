#!/usr/bin/env python
# -*- coding: utf-8 -*-
from dataclasses import dataclass, is_dataclass, asdict

import numpy as np
from typing import Literal
from typing import List
from .main_algorithm import *
from dataclasses_json import dataclass_json


@dataclass_json
@dataclass
class UiChartLine:
    name: str
    curve: np.array

    def as_draw_args(self):
        return [self.curve, self.name]


@dataclass_json
@dataclass
class UiChart:
    title: str = None
    name: str = None
    x_label: str = None
    y_label: str = None
    y_ticks: any = None
    lines: List[UiChartLine] = None
    type: Literal['line', 'bar',] = 'line'


@dataclass_json
@dataclass
class UiChartData:
    出力曲线: UiChart = None  # 出力曲线
    发电量曲线: UiChart = None  # 发电量曲线
    并网曲线: UiChart = None  # 并网曲线
    储能频率曲线: UiChart = None  # 储能频率曲线
    储能SOC曲线: UiChart = None  # 储能频率曲线
    CO2减排曲线: UiChart = None  # CO2减排曲线
    field_names = [
        '出力曲线',
        '发电量曲线',
        '并网曲线',
        '储能频率曲线',
        '储能SOC曲线',
        'CO2减排曲线'
    ]


@dataclass
class UiRealTimeData:
    实时电压: np.array = None
    实时电流: np.array = None
    实时功率: np.array = None
    实时发电量: np.array = None
    累计发电量: np.array = None
    时间: np.array = None
    实时SOC值1: np.array = None
    充放电状态1: np.array = None
    实时功率1: np.array = None
    累计冲放电1: np.array = None
    时间2: np.array = None
    实时SOC值2: np.array = None
    充放电状态2: np.array = None
    实时功率2: np.array = None
    累计冲放电2: np.array = None
    时间2: np.array = None
    field_names = ['实时电压',
                   '实时电流',
                   '实时功率',
                   '实时发电量',
                   '累计发电量',
                   '时间',
                   '实时SOC值1',
                   '充放电状态1',
                   '实时功率1',
                   '累计冲放电1',
                   '时间2',
                   '实时SOC值2',
                   '充放电状态2',
                   '实时功率2',
                   '累计冲放电2',
                   '时间2']


@dataclass
class UiSocData:
    min: str = None
    max: str = None
    label: str = '100%'


@dataclass
class UiEconomicDataPair:
    光伏: str = None
    光储: str = None


class UiEconomicLeft:
    运维: UiEconomicDataPair = None
    峰谷差价: UiEconomicDataPair = None
    调峰: UiEconomicDataPair = None
    调频: UiEconomicDataPair = None
    售电: UiEconomicDataPair = None
    偏差电量: UiEconomicDataPair = None
    偏差店家惩罚: UiEconomicDataPair = None
    field_names = [
        '运维',
        '峰谷差价',
        '调峰',
        '调频',
        '售电',
        '偏差电量',
        '偏差店家惩罚',
    ]


class UiEconomicRight:
    运行年限: UiEconomicDataPair = None
    回本时间: UiEconomicDataPair = None
    容量配置: UiEconomicDataPair = None
    储能一次性投资: UiEconomicDataPair = None
    年收益率: UiEconomicDataPair = None
    field_names = [
        '运行年限',
        '回本时间',
        '容量配置',
        '储能一次性投资',
        '年收益率'
    ]


@dataclass
class UiEconomicLeftBottom:
    功率: str
    容量: str
    field_names = [
        '功率',
        '容量'
    ]


@dataclass
class UiEconomicRightBottom:
    功率: str
    容量: str
    field_names = [
        '功率',
        '容量'
    ]


@dataclass
class UiData(object):
    chart_data: UiChartData = None
    realtime_data: UiRealTimeData = None
    economic_left: UiEconomicLeft = None
    economic_right: UiEconomicRight = None
    soc: UiSocData = None


@dataclass
class ProgressInfo:
    title: str = None
    progress: int = None
    info: str = None


class UiTask(object):
    """

    """

    def __init__(self, config, update_info_func=None):
        self.config = config
        self.update_info_func = update_info_func
        self.rawOutputCurve = None
        self.rawValidCurve = None
        self.smoothOutputCurve = None
        self.smoothValidCurve = None
        self.powerOutputCurve = None
        self.capacityOutputCurve = None
        self.Model = None
        self.bSocCurve = None
        self.scSocCurve = None

        self.k = 0.99

    def update_progress_info(self, info: ProgressInfo):
        if not self.update_info_func is None:
            self.update_info_func(info)
        else:
            print(info)

    def run_init_data(self):
        self.rawOutputCurve = init_data(self.config)
        self.update_progress_info(ProgressInfo("初始化数据完成", 10, "初始化执行"))

    def pre_data(self):
        self.rawValidCurve = calc_satisfy(self.config, self.rawOutputCurve)
        self.smoothOutputCurve = calc_smooth_curve(self.config, self.rawOutputCurve)  # 平抑后的出力曲线
        self.update_progress_info(ProgressInfo("初始化数据完成", 12, "初始化执行"))
        self.smoothValidCurve = np.ones(len(self.smoothOutputCurve))
        _ = subsampling_algorithm(self.rawOutputCurve - self.smoothOutputCurve, n=70)
        self.powerOutputCurve = _[0]
        self.capacityOutputCurve = _[1]
        self.Model = CapacityAllocation(
            self.config,
            self.capacityOutputCurve,
            self.powerOutputCurve,
            np.sum(self.smoothOutputCurve) / 3600,
            np.sum(self.rawOutputCurve) / 3600)
        self.Model.energy_storage_capacity_allocation()
        self.bSocCurve = np.cumsum(self.capacityOutputCurve) / 3600 / self.Model.b_capacity + 0.5
        self.scSocCurve = np.cumsum(self.powerOutputCurve) / 3600 / self.Model.sc_capacity + 0.5

    def get_chart_data(self) -> UiChartData:
        ui_chart_data: UiChartData = UiChartData()
        ui_chart_data.出力曲线 = UiChart(
            title="出力曲线",
            x_label="时间(h)", y_label="功率(Mw)",
            lines=[
                UiChartLine('纯光伏', self.rawOutputCurve),
                UiChartLine('光储', self.smoothOutputCurve),
            ])
        ui_chart_data.发电量曲线 = UiChart(
            title="发电量",
            x_label="时间(h)", y_label="能量(Mwh)",
            lines=[
                UiChartLine('纯光伏', self.rawOutputCurve),
                UiChartLine('光储', self.smoothOutputCurve),
            ],
            type='bar'
        )
        ui_chart_data.并网曲线 = UiChart(
            title="并网时间",
            x_label="时间(h)",
            y_label="是否并网",
            y_ticks=[[0, 1, 2, 3], ['不并网', '并网', '不并网', '并网']],
            lines=[
                UiChartLine('纯光伏', self.rawValidCurve),
                UiChartLine('光储', self.smoothValidCurve + 2),
            ],
            type='line'
        )
        ui_chart_data.储能频率曲线 = UiChart(
            title="储能频率曲线",
            x_label="时间(h)",
            y_label="功率(Mw)",
            lines=[
                UiChartLine('功率型', self.powerOutputCurve),
                UiChartLine('容量型', self.capacityOutputCurve),
            ],
            type='line'
        )
        ui_chart_data.储能SOC曲线 = UiChart(
            title="储能SOC曲线",
            x_label="时间(h)",
            y_label="百分比(%)",
            lines=[
                UiChartLine('功率型', self.bSocCurve),
                UiChartLine('容量型', self.scSocCurve),
            ],
            type='line'
        )
        ui_chart_data.CO2减排曲线 = UiChart(
            title="CO2减排量",
            x_label="时间(h)",
            y_label="CO2减排量(t)",
            lines=[
                UiChartLine('纯光伏', self.rawOutputCurve * self.k),
                UiChartLine('光储', self.smoothOutputCurve * self.k),
            ],
            type='line'
        )
        return ui_chart_data

    def get_realtime_date(self) -> UiRealTimeData:
        real_time_data: UiRealTimeData = UiRealTimeData()
        b_soc_curve, b_state_curve, b_power_curve, b_energy_curve, sc_soc_curve, sc_state_curve, sc_power_curve, sc_energy_curve = calc_realtime_bsc_data(

            self.bSocCurve,
            self.scSocCurve,
            self.capacityOutputCurve,
            self.powerOutputCurve)
        realtime_v, realtime_a, realtime_power, realtime_energy, energy_curve = calc_realtime_power_data(
            self.smoothOutputCurve)
        time = np.arange(0, 86400, 1)
        real_time_data.实时电压 = realtime_v,
        real_time_data.实时电流 = realtime_a,
        real_time_data.实时功率 = realtime_power,
        real_time_data.实时发电量 = realtime_energy,
        real_time_data.累计发电量 = energy_curve,
        real_time_data.时间 = time,
        real_time_data.实时SOC值1 = b_soc_curve,
        real_time_data.充放电状态1 = b_state_curve,
        real_time_data.实时功率1 = b_power_curve,
        real_time_data.累计冲放电1 = b_energy_curve,
        real_time_data.时间2 = time,
        real_time_data.实时SOC值2 = sc_soc_curve,
        real_time_data.充放电状态2 = sc_state_curve,
        real_time_data.实时功率2 = sc_power_curve,
        real_time_data.累计冲放电2 = sc_energy_curve,
        real_time_data.时间2 = time
        return real_time_data

    def get_economic_left(self) -> UiEconomicLeft:
        """
        todo
        :return:
        """
        values = self.Model.get_daily_benefit()
        print(type(values))
        economic_left: UiEconomicLeft = UiEconomicLeft()
        economic_left.运维 = UiEconomicDataPair(values[0], values[1])
        economic_left.峰谷差价 = UiEconomicDataPair(values[2], values[3])
        economic_left.调峰 = UiEconomicDataPair(values[4], values[5])
        economic_left.调频 = UiEconomicDataPair(values[6], values[7])
        economic_left.售电 = UiEconomicDataPair(values[8], values[9])
        economic_left.偏差电量 = UiEconomicDataPair(values[10], values[11])
        economic_left.偏差店家惩罚 = UiEconomicDataPair(values[12], values[13])
        return economic_left

    def get_economic_right(self) -> UiEconomicRight:
        """
        todo
        :return:
        """
        values = self.Model.get_lifespan_benefit()
        economic_right = UiEconomicRight()
        economic_right.运行年限 = UiEconomicDataPair(values[0], values[1])
        economic_right.回本时间 = UiEconomicDataPair(values[2], values[3])
        economic_right.容量配置 = UiEconomicDataPair(values[4], values[5])
        economic_right.储能一次性投资 = UiEconomicDataPair(values[6], values[7])
        economic_right.年收益率 = UiEconomicDataPair(values[8], values[9])
        return economic_right

    def get_soc(self) -> UiSocData:
        """
        todo
        :return:
        """
        soc = UiSocData()
        return soc

    def get_economic_left_bottom(self) -> UiEconomicLeftBottom:
        v = self.Model.get_b_para()
        return UiEconomicLeftBottom(v[0], v[1])

    def get_economic_right_bottom(self) -> UiEconomicRightBottom:
        v = self.Model.get_sc_para()
        return UiEconomicLeftBottom(v[0], v[1])

    def run(self):
        pass


import os, json

DIR_PATH = os.path.abspath(os.path.dirname(__file__))

if __name__ == '__main__':
    config = json.loads(open(os.path.join(DIR_PATH, 'config.json')).read())
    config.update({"outputCurvePath": os.path.join(DIR_PATH, 'data', 'example_sec.csv')})
    task = UiTask(config, update_info_func=lambda x: print)
    task.run_init_data()
    task.pre_data()
    chart_data: UiChartData = task.get_chart_data()
    print("chart_data->", chart_data.并网曲线)
    # print(chart_data.to_json())
    print(chart_data)
    real_time_data = task.get_realtime_date()
    print('real_time_data', real_time_data)
    print(task.get_economic_left())
    print(task.get_economic_right())
    print(task.get_economic_left_bottom())
    print(task.get_economic_right_bottom())

    # print(json.dumps(chart_data, cls=EnhancedJSONEncoder, ensure_ascii=False))
