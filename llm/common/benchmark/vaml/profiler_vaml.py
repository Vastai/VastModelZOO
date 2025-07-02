import os
import time
from loguru import logger
import numpy as np
from datetime import datetime
import sys 
print(sys.path)
try:
    #import vaml
    from _vaststream_pybind11 import vaml
    #import vaml
    print("success")
    # from vaststream import vaml
except ImportError:
    vaml = None
    print("import error")


UNIT = 1024 * 1024
CARD_DIE_INFO = {"VA1":2, "VA1L":2,"VA10":4, "VA10L":4,"VA16":4,"VA16L":4}

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
        self.clock = np.array(self.clock, dtype=np.int32).mean(axis=0, keepdims=keepdims).tolist()
        self.temperature = np.array(self.temperature, dtype=np.float32).mean(axis=0, keepdims=keepdims).tolist()
        self.power = np.array(self.power, dtype=np.float32).mean(axis=0, keepdims=keepdims).tolist()
        self.mem = np.array(self.mem, dtype=np.float32).mean(axis=0, keepdims=keepdims).tolist()
        self.ai_utilization = np.array(self.ai_utilization, dtype=np.float32).mean(axis=0, keepdims=keepdims).tolist()


class MonitorManager:
    def __init__(self) -> None:
        try:
            vaml.setLogLevel(vaml.LOG_LEVEL.FATAL)
            self.vaml_status = vaml.init()
            if not self.vaml_status:
                logger.error("MonitorManager vaml init error!")
            else:
                self.device = vaml.VastaiDevice()
                self.cards = self.device.getCards(-1)
                # self.cards = vaml.getCardsInfo()
                self.dies = []
                for card in self.cards:
                    self.dies.extend(card.getDies(-1))
                    # cardhandle = vaml.getCardHandleByUUID(card.uuid)
                    # for die in vaml.getDiesInfo(cardhandle):
                    #     self.dies.append(vaml.die.VastaiDie(die.dieIndex))
                
                self.card_counts = vaml.getCardCount()
                self.die_counts = vaml.getAiDeviceCount()
        except:
            self.vaml_status = False
            logger.warning("MonitorManager vaml init fail")
        
        self.device_results_dict = {}
        self.sampling_flag = False
        self.sampling_time = 1 # 1s 
    
    def shut_down(self,):
        vaml.shutDown()
    
    
    def get_clock(self, die_id):
        # list(12): [OCLK, ODSPCLK,VCLK, ECLK, UCLK, VDSPCLK, UCLK, V3DCLK, CCLK, XSPICLK, PERCLK, CEDARCLK]
        # not supported in vaml now
        return [-1]*12

        if self.vaml_status and die_id < len(self.dies):
            clk_info = self.dies[die_id].getPllClock()
            return [clk/1e6 for clk in clk_info.clockArray]
        else:
            return [-1]*12

    def get_temperature(self, die_id):   
        # list(15): [Sysc,    Vnocv,      Vnec3,Vnec2,Vnec1,Vnec0,  Smu_soc,   Oak7...Oak0]
        # return:   [系统模块  Video总线        Video编码             SMU           AI]
        if self.vaml_status and die_id < len(self.dies):
            temp = self.dies[die_id].getTemperature().temperature
            return [temp[0]/100.0,  temp[1]/100.0, sum(temp[2:6])/400.0, temp[6]/100.0, sum(temp[7:])/800.0]
        else:
            return [-1]*5

    def get_power(self, die_id):
        # list: [DDR, Vid, Dlc_Soc, Total]
        if self.vaml_status and die_id < len(self.dies):
            return [p/1e6 for p in self.dies[die_id].getPower().power]
        else:
            return [-1]*4

    def get_mem_utilization_rate(self, die_id):
        # total, free, used, utilizationRate
        if self.vaml_status and die_id < len(self.dies):
            mem_info = self.dies[die_id].getMemUtilizationRate()
            # ->[MB, MB, MB, %]
            return [mem_info.total/UNIT, mem_info.free/UNIT, mem_info.used/UNIT, mem_info.utilizationRate/100.0]
        else:
            return [-1]*4

    def get_ai_utilization_rate(self, die_id):
        # ai, vdsp, vemcu, vdmcu
        if self.vaml_status and die_id < len(self.dies):
            util_info = self.dies[die_id].getUtilizationRate()
            # -> %
            return [util_info.ai/100.0, util_info.vdsp/100.0, util_info.vemcu/100.0, util_info.vdmcu/100.0]
        else:
            return [-1]*4
    
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
            
            # 过滤无效数据
            if sum(ai_util) < 10:
                continue

            # 每100轮求平均 长时间累计过大
            if len(device_records.clock) >= 100:
                device_records.get_average(keepdims=True)

            device_records.update(clock, temp, power, mem, ai_util)
        
        self.device_results_dict[die_id] = device_records
        
        
    
    def end(self,):
        self.sampling_flag = False
        time.sleep(1)



    def summary(self,die_id):
        output = {}
        device_records = self.device_results_dict.get(die_id, None)
        if device_records:
            device_records.get_average()
            try:
                output['oclk'] = device_records.clock[0]
                output['temperature'] = device_records.temperature[0]
                output['power'] = device_records.power[-1] # total
                output['mem'] = device_records.mem[2] # used
                output['ai'] = device_records.ai_utilization[0] # ai
            except:
                pass

        return output

    # def __del__(self,):
    #     if self.vaml_status:
    #         vaml.shutDown()

class Statistics:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.finish_request = 0  # 完成请求数量
        self.token_num = 0 
        self.word_num = 0  # word数量
        self.used = 0  # 请求耗时
        self.first_token_used = 0  # 首个token耗时
        self.req_token_ave_used_total = 0  # 请求的token平均耗时总和
        self.req_word_ave_used_total = 0  # 请求的word平均耗时总和
        self.calculate_context_ave_used_total = 0 # iter0计算平均时间
        self.calculate_generation_ave_used_total = 0 # 生成token计算平均时间
        self.request_ave_len = 0 # 请求平均token长度
        self.reqs_generation_duration_us = [[0,0]] # 每个请求内iter1起止时间 us
        self.stop=False

    def set_stop(self):
        with self.lock:
            self.stop = True

    def update(self, first_token_used, used, req_token_ave_used_total, req_word_ave_used_total, token_num, word_num,
                     calculate_context_used, calculate_generation_used, request_len, req_generation_duration_us):
        with self.lock:
            self.finish_request += 1
            self.token_num += token_num
            self.word_num += word_num
            self.used += used
            self.first_token_used += first_token_used
            self.req_token_ave_used_total += req_token_ave_used_total
            self.req_word_ave_used_total += req_word_ave_used_total
            self.calculate_context_ave_used_total += calculate_context_used
            self.calculate_generation_ave_used_total += calculate_generation_used
            self.request_ave_len += request_len
            self.reqs_generation_duration_us.append(req_generation_duration_us)


class VAML_MonitorManager:
    def __init__(self,):
        self.monitor = MonitorManager()
        self.monitor_results = {}
        self.model_tp_num = self.monitor.die_counts
        self.tp_device_ids_list = list(range(self.monitor.die_counts))
        print(f"tp device ids:{self.tp_device_ids_list}")

    def start_profile(self,):
        self.monitor_results['start_time'] = datetime.now()
        self.monitor_thread_lists = []
        for die_id in self.tp_device_ids_list:
            monitor_thread = threading.Thread(target=self.monitor.start, args=(die_id,))
            monitor_thread.start()
            self.monitor_thread_lists.append(monitor_thread)

    def stop_profile(self,):
        self.monitor_results['stop_time'] = datetime.now()
        self.monitor.end()
        for monitor_thread in self.monitor_thread_lists:
            monitor_thread.join()
        for die_id in self.tp_device_ids_list:
            self.monitor_results[die_id] = self.monitor.summary(die_id)
    
    def get_profile_summary(self,):
        results_summary = {'oclk':0, 'temperature': 0, 'power': 0, 'mem': 0, 'ai': 0}
        results = [v for k,v in self.monitor_results.items() if k in self.tp_device_ids_list]
        results_summary['oclk'] = results[0].get('oclk', 0)
        results_summary['mem'] = sum([item.get('mem', 0)/1000.0 for item in results])
        results_summary['temperature'] = sum([item.get('temperature', 0) for item in results])/len(results)
        results_summary['power'] = sum([item.get('power', 0) for item in results])/len(results)
        results_summary['ai'] = sum([item.get('ai', 0) for item in results])/len(results)
        return results_summary
    
    def report(self, statistics, all_elapsed_time):
        # all_gen_token_throughput = statistic.token_num / all_elapsed_time
        # # [N, 2]
        # generation_duration_avg = np.array(statistic.reqs_generation_duration_us, dtype=np.float64)  
        # generation_duration_avg = (generation_duration_avg[:,1] - generation_duration_avg[:,0]) / 1e6
        # logger.info(f"All gen token throughput : {statistic.token_num} / {all_elapsed_time} = {all_gen_token_throughput:.2f} token/s")

        html_output_request_file = f"vaml_request_benchmark.html"
        data_request_old = None
        
        if os.path.exists(html_output_request_file):
            pd_table = pd.read_html(html_output_request_file, flavor='bs4')
            data_request_old = pd_table[0].values.tolist()

        card_num = self.model_tp_num // CARD_DIE_INFO.get(self.card_type, "VA1L")
        profiler_info = self.get_profile_summary()
        data_request = [
            [self.card_type, card_num, self.model_name, self.model_weights_type, self.model_tp_num, self.model_batch_size,
            self.test_type, int(statistic.finish_request), all_elapsed_time, 
            statistic.request_ave_len/statistic.finish_request, statistic.token_num/statistic.finish_request,
            # all_gen_token_throughput, 
            all_elapsed_time/statistic.finish_request, 
            statistic.calculate_context_ave_used_total/statistic.finish_request, statistic.calculate_generation_ave_used_total/statistic.token_num,  
            generation_duration_avg.mean(),
            statistic.token_num/generation_duration_avg.mean()/statistic.finish_request, statistic.token_num/all_elapsed_time,
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            profiler_info.get('temperature',0), profiler_info.get('mem', 0), profiler_info.get('ai', 0)
            ],
        ]
        try:
            data_request.extend(data_request_old)
        except:
            pass

        df_request = pd.DataFrame(data_request, columns=["推理卡型号","推理卡数量","基准测试模型","模型权重类型","模型并行数TP", "服务并发数量",
                            "测试数据类型", "完成请求数量", "测试总体时间(s)",
                            "请求平均长度(token)", "请求平均生成长度(token)",
                            # "请求总吞吐(token/s)", 
                            "单个请求平均耗时(s)", 
                            "prefill平均耗时(s)", "decoding平均耗时(s)", 
                            "单个请求内decoding总平均耗时(s)",
                            "单个请求平均decoding吞吐(token/s)", "总吞吐(token/s)",
                            "测试时间戳",
                            "卡平均温度(℃)", "显存总占用(GB)", "ai利用率(%)",
                            ])


        # df_request.to_csv(f"{self.save_prefix}_request.csv")

        # 将数据转换为HTML表格
        html_request_table = df_request.to_html(index=False)

        with open(html_output_request_file, 'w') as f:
            f.write(html_request_table)


        print(f'CSV转换为HTML成功，已保存到 {html_output_request_file}')

        
if __name__ == "__main__":
    vaml_check = VAML_MonitorManager()
    exit(0)
    vaml_check.start_profile()
    ## 运行程序
    ### 产生一个statistic
    vaml_check.stop_profile()
    all_elapsed_time = (vaml_check.monitor_results['stop_time'] - vaml_check.monitor_results['start_time']).total_seconds()
    vaml_check.report(statistic, all_elapsed_time)

