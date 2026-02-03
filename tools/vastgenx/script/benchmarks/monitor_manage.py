# ==============================================================================
#
# Copyright (C) 2025 VastaiTech Technologies Inc.  All rights reserved.
#
# ==============================================================================
#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import threading
import time
import psutil
import numpy as np
from datetime import datetime

logger = type("logger", (), {"info": print, "warning": print, "error": print})()

try:
    from loguru import logger as loguru_logger

    logger = loguru_logger
except Exception as e:

    print("Failed to initialize 'loguru':", e)

try:
    import vaml
except ImportError:
    logger.warning("Failed to import 'vaml'. Some features may be disabled.")
    vaml = None


class DeviceStatistics:
    def __init__(self) -> None:
        self.clock = []
        self.temperature = []
        self.power = []
        self.mem = []
        self.ai_utilization = []

    def update(self, clock, temperature, power, mem, ai_util):
        self.clock.append(clock)
        self.temperature.append(temperature)
        self.power.append(power)
        self.mem.append(mem)
        self.ai_utilization.append(ai_util)

    def get_average(self, keepdims=False):
        self.clock = (
            np.array(self.clock, dtype=np.int32)
            .mean(axis=0, keepdims=keepdims)
            .tolist()
        )
        self.temperature = (
            np.array(self.temperature, dtype=np.float32)
            .mean(axis=0, keepdims=keepdims)
            .tolist()
        )
        self.power = (
            np.array(self.power, dtype=np.float32)
            .mean(axis=0, keepdims=keepdims)
            .tolist()
        )
        self.mem = (
            np.array(self.mem, dtype=np.float32)
            .mean(axis=0, keepdims=keepdims)
            .tolist()
        )
        self.ai_utilization = (
            np.array(self.ai_utilization, dtype=np.float32)
            .mean(axis=0, keepdims=keepdims)
            .tolist()
        )


class MonitorManager:
    def __init__(self, cpu_list=None) -> None:
        self.cpu_list = cpu_list
        try:
            vaml.setLogLevel(vaml.LOG_LEVEL.FATAL)
            self.vaml_status = vaml.init()
            self.cpu_use = []
            if not self.vaml_status:
                logger.error("MonitorManager vaml init error!")
                print("!!!!!######MonitorManager vaml init error!")
            else:
                print("!!!!!######MonitorManager vaml init OK!")
                self.device = vaml.VastaiDevice()
                self.cards = self.device.getCards(-1)
                self.dies = []
                for card in self.cards:
                    self.dies.extend(card.getDies(-1))
                self.card_counts = vaml.getCardCount()
                self.die_counts = vaml.getAiDeviceCount()
        except Exception as e:
            self.vaml_status = False
            logger.warning(f"MonitorManager vaml init fail: {e}")
            self.die_counts = 0

        self.device_results_dict = {}
        self.sampling_flag = False
        self.cpu_sampling_flag = False
        self.sampling_time = 1

    def shut_down(self):
        if self.vaml_status:
            vaml.shutDown()

    def monitor_cpu_usage(self, cpu_list=None, sampling_time=1):
        self.cpu_use = []
        self.cpu_sampling_flag = True
        cpu_list_to_use = cpu_list if cpu_list is not None else self.cpu_list
        while self.cpu_sampling_flag:
            cpu_percents = psutil.cpu_percent(interval=sampling_time, percpu=True)
            avg_cpu = (
                sum([cpu_percents[i] for i in cpu_list_to_use]) / len(cpu_list_to_use)
                if cpu_list_to_use
                else psutil.cpu_percent(interval=sampling_time)
            )
            # print(f"avg_cpu*****{avg_cpu}")
            self.cpu_use.append(avg_cpu)

    def get_clock(self, die_id):
        return [-1] * 12

    def get_temperature(self, die_id):
        if self.vaml_status and die_id < len(self.dies):
            temp = self.dies[die_id].getTemperature().temperature
            return [
                temp[0] / 100.0,
                temp[1] / 100.0,
                sum(temp[2:6]) / 400.0,
                temp[6] / 100.0,
                sum(temp[7:]) / 800.0,
            ]
        return [-1] * 5

    def get_power(self, die_id):
        if self.vaml_status and die_id < len(self.dies):
            return [p / 1e6 for p in self.dies[die_id].getPower().power]
        return [-1] * 4

    def get_mem_utilization_rate(self, die_id):
        if self.vaml_status and die_id < len(self.dies):
            mem_info = self.dies[die_id].getMemUtilizationRate()
            UNIT = 1024 * 1024
            return [
                mem_info.total / UNIT,
                mem_info.free / UNIT,
                mem_info.used / UNIT,
                mem_info.utilizationRate / 100.0,
            ]
        return [-1] * 4

    def get_ai_utilization_rate(self, die_id):
        if self.vaml_status and die_id < len(self.dies):
            util_info = self.dies[die_id].getUtilizationRate()
            return [
                util_info.ai / 100.0,
                util_info.vdsp / 100.0,
                util_info.vemcu / 100.0,
                util_info.vdmcu / 100.0,
            ]
        return [-1] * 4

    def start(self, die_id, sampling_time=1):
        device_records = DeviceStatistics()
        self.sampling_flag = True
        self.sampling_time = sampling_time
        while self.sampling_flag:
            time.sleep(self.sampling_time)
            clock = self.get_clock(die_id)
            temp = self.get_temperature(die_id)
            power = self.get_power(die_id)
            mem = self.get_mem_utilization_rate(die_id)
            ai_util = self.get_ai_utilization_rate(die_id)
            if sum(ai_util) < 10:
                continue
            if len(device_records.clock) >= 100:
                device_records.get_average(keepdims=True)
            device_records.update(clock, temp, power, mem, ai_util)
        self.device_results_dict[die_id] = device_records

    def end(self):
        self.sampling_flag = False
        self.cpu_sampling_flag = False
        time.sleep(1)

    def summary(self, die_id):
        output = {}
        device_records = self.device_results_dict.get(die_id, None)
        if device_records:
            device_records.get_average()
            try:
                output["oclk"] = device_records.clock[0]
                output["temperature"] = device_records.temperature[0]
                output["power"] = device_records.power[-1]
                output["mem"] = device_records.mem[2]
                output["ai"] = device_records.ai_utilization[0]
            except:
                pass
        return output


class VAML_MonitorManager:
    def __init__(self, cpu_list=None, gpu_list=None):
        self.cpu_list = (
            cpu_list if cpu_list is not None else list(range(psutil.cpu_count()))
        )
        self.monitor = MonitorManager(self.cpu_list)

        self.monitor_results = {}
        self.model_tp_num = self.monitor.die_counts
        all_gpu_ids = list(range(self.monitor.die_counts))
        self.tp_device_ids_list = [
            i for i in (gpu_list or all_gpu_ids) if i in all_gpu_ids
        ]
        if self.monitor.vaml_status:
            self.model_tp_num = self.monitor.die_counts
            all_gpu_ids = list(range(self.monitor.die_counts))
            self.tp_device_ids_list = [
                i for i in (gpu_list or all_gpu_ids) if i in all_gpu_ids
            ]
            print(f"Using CPU cores: {self.cpu_list}")
            print(f"Using TP device ids: {self.tp_device_ids_list}")
        else:
            logger.warning(
                "VAML initialization failed. GPU monitoring features will be disabled."
            )
            self.model_tp_num = 0
            self.tp_device_ids_list = []
            print(f"Using CPU cores: {self.cpu_list}")
            print("VAML not initialized, no GPU monitoring.")

    def start_profile(self):
        self.monitor_results["start_time"] = datetime.now()
        self.monitor_thread_lists = []
        cpu_monitor_thread = threading.Thread(
            target=self.monitor.monitor_cpu_usage, args=(self.cpu_list,)
        )
        cpu_monitor_thread.start()
        self.monitor_thread_lists.append(cpu_monitor_thread)

        for die_id in self.tp_device_ids_list:
            monitor_thread = threading.Thread(target=self.monitor.start, args=(die_id,))
            monitor_thread.start()
            self.monitor_thread_lists.append(monitor_thread)

    def stop_profile(self):
        self.monitor_results["stop_time"] = datetime.now()
        self.monitor.end()
        for thread in self.monitor_thread_lists:
            thread.join()
        for die_id in self.tp_device_ids_list:
            self.monitor_results[die_id] = self.monitor.summary(die_id)
            # print(f"die id: {die_id}, result {self.monitor_results[die_id]}")

    def get_profile_summary(self):
        results_summary = {
            "oclk": 0,
            "temperature": 0,
            "power": 0,
            "mem": 0,
            "ai": 0,
            "temperature_mean": 0,
            "temperature_median": 0,
            "temperature_p99": 0,
            "power_mean": 0,
            "power_median": 0,
            "power_p99": 0,
            "ai_mean": 0,
            "ai_median": 0,
            "ai_p99": 0,
        }
        results = [v for k, v in self.monitor_results.items() if isinstance(k, int)]

        if not results:
            return results_summary

        results_summary["oclk"] = results[0].get("oclk", 0)
        results_summary["mem"] = sum([item.get("mem", 0) / 1000.0 for item in results])

        temperatures = [item.get("temperature", 0) for item in results]
        powers = [item.get("power", 0) for item in results]
        ais = [item.get("ai", 0) for item in results]

        if temperatures:
            results_summary.update(
                {
                    "temperature_mean": np.mean(temperatures),
                    "temperature_median": self._calculate_median(temperatures),
                    "temperature_p99": self._calculate_percentile(temperatures, 99),
                    "temperature": np.mean(temperatures),
                }
            )
        if powers:
            results_summary.update(
                {
                    "power_mean": np.mean(powers),
                    "power_median": self._calculate_median(powers),
                    "power_p99": self._calculate_percentile(powers, 99),
                    "power": np.mean(powers),
                }
            )
        if ais:
            results_summary.update(
                {
                    "ai_mean": np.mean(ais),
                    "ai_median": self._calculate_median(ais),
                    "ai_p99": self._calculate_percentile(ais, 99),
                    "ai": np.mean(ais),
                }
            )

        return results_summary

    def _calculate_median(self, data):
        sorted_data = sorted(data)
        n = len(sorted_data)
        return (
            (sorted_data[n // 2 - 1] + sorted_data[n // 2]) / 2
            if n % 2 == 0
            else sorted_data[n // 2]
        )

    def _calculate_percentile(self, data, percentile):
        return np.percentile(data, percentile)
