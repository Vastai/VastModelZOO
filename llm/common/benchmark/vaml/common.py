# Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# coding: utf-8

__all__ = [
    "OP_STATUS", "LOG_LEVEL", "PciCardId", "PciSubSystemId", "NodeMajorMinor",
    "DieIndex", "NodeBaseInfo", "DieBaseInfo", "AiDevInfo", "CodecDevInfo",
    "PciInfo", "Capability", "CardInfo", "FanSpeedInfo", "PllClock",
    "Temperature", "TempThreshold", "PinVolt", "Power", "PowerCur",
    "PowerVolt", "MemUtilizationRate", "UtilizationRate", "McuUtilizationRate",
    "ProcessInfo", "vamlER_SUCCESS", "MAX_PLL_CLOCK_SUBMODULE",
    "MAX_TEMPERATURE_SUBMODULE", "MAX_TEMP_THRESHOLD_SUBMODULE",
    "MAX_POWER_SUBMODULE", "MAX_POWER_CURRENT_SUBMODULE",
    "MAX_POWER_VOLTAGE_SUBMODULE", "MAX_PIN_VOLT_SUBMODULE",
    "MAX_PVT_VOLT_SUBMODULE", "MAX_DIE_CMCU_NUM", "MAX_DIE_VDSP_NUM",
    "MAX_DIE_VDMCU_NUM", "MAX_DIE_VEMCU_NUM", "init", "shutDown",
    "setLogLevel", "logMsg", "errorString"
]

from ._vaststream_pybind11 import vaml as _vaml
from .utils import *
from typing import List

# =========================== ENUM =============================


class OP_STATUS():
    """An enum that defines op status.
    
    Contains SUCCESS, UNINITIALIZED, INITIALIZED, CARD,_EMPTY, INITI_FAIL, CARD_NOTFOUND,
    INPUT_PARAMETER_NULL, INPUT_PARAMETER_ERROR, PCIE_MAJOR_ERROR, USER_BUF_INSUFFICIENT,
    CALLBACK_INITIALIZED, CALLBACK_UNINITIALIZED, PROFILER_ALREADY_RUNNING, PROFILER_ALREADY_STOP,
    PROFILER_EXPORT_FILE_TYPE_INVALID, PROFILER_EXPORT_FOLDER_PATH_INVALID, AI_SCRIPT_PATH_INVALID,
    AI_SCRIPT_ROOT_PERMISSION, ERROR_MAX.
    """
    SUCCESS: int = _vaml.opStatus.VAML_SUCCESS
    UNINITIALIZED: int = _vaml.opStatus.VAML_ERROR_UNINITIALIZED
    INITIALIZED: int = _vaml.opStatus.VAML_ERROR_INITIALIZED
    CARD_EMPTY: int = _vaml.opStatus.VAML_ERROR_CARD_EMPTY
    INITI_FAIL: int = _vaml.opStatus.VAML_ERROR_INITI_FAIL
    CARD_NOTFOUND: int = _vaml.opStatus.VAML_ERROR_CARD_NOTFOUND
    INPUT_PARAMETER_NULL: int = _vaml.opStatus.VAML_ERROR_INPUT_PARAMETER_NULL
    INPUT_PARAMETER_ERROR: int = _vaml.opStatus.VAML_ERROR_INPUT_PARAMETER_ERROR
    PCIE_MAJOR_ERROR: int = _vaml.opStatus.VAML_ERROR_PCIE_MAJOR_ERROR
    USER_BUF_INSUFFICIENT: int = _vaml.opStatus.VAML_USER_BUF_INSUFFICIENT
    CALLBACK_INITIALIZED: int = _vaml.opStatus.VAML_ERROR_CALLBACK_INITIALIZED
    CALLBACK_UNINITIALIZED: int = _vaml.opStatus.VAML_ERROR_CALLBACK_UNINITIALIZED
    PROFILER_ALREADY_RUNNING: int = _vaml.opStatus.VAML_ERROR_PROFILER_ALREADY_RUNNING
    PROFILER_ALREADY_STOP: int = _vaml.opStatus.VAML_ERROR_PROFILER_ALREADY_STOP
    PROFILER_EXPORT_FILE_TYPE_INVALID: int = _vaml.opStatus.VAML_ERROR_PROFILER_ALREADY_STOP
    PROFILER_EXPORT_FOLDER_PATH_INVALID: int = _vaml.opStatus.VAML_ERROR_PROFILER_EXPORT_FOLDER_PATH_INVALID
    AI_SCRIPT_PATH_INVALID: int = _vaml.opStatus.VAML_ERROR_AI_SCRIPT_PATH_INVALID
    AI_SCRIPT_ROOT_PERMISSION: int = _vaml.opStatus.VAML_ERROR_AI_SCRIPT_ROOT_PERMISSION
    ERROR_MAX: int = _vaml.opStatus.VAML_ERROR_MAX


class LOG_LEVEL():
    """An enum that defines log level.

    Contains TRACE, DEBUG, INFO, WARN, ERROR, FATAL, NONE.
    """
    TRACE: int = _vaml.logLevel.VAML_LOG_TRACE
    DEBUG: int = _vaml.logLevel.VAML_LOG_DEBUG
    INFO: int = _vaml.logLevel.VAML_LOG_INFO
    WARN: int = _vaml.logLevel.VAML_LOG_WARN
    ERROR: int = _vaml.logLevel.VAML_LOG_ERROR
    FATAL: int = _vaml.logLevel.VAML_LOG_FATAL
    NONE: int = _vaml.logLevel.VAML_LOG_NONE


# =========================== UNION =============================


class PciCardId(_vaml.pciCardId):
    """A union that defines pci card id.

    Attributes:
        renderId(int): The render id.
        cardId(int): The card id.
    """
    renderId: int
    cardId: int


class PciSubSystemId(_vaml.pciSubSystemId):
    """A union that defines pci sub system id.

    Attributes:
        subVenderId(int): The sub render id.
        subcardId(int): The sub card id.
    """
    subVenderId: int
    subcardId: int


class NodeMajorMinor(_vaml.nodeMajorMinor):
    """A union that defines attribute of node major and minor.

    Attributes:
        minor(int): The minor data.
        major(int): The major data.
    """
    minor: int
    major: int


class DieIndex(_vaml.dieIndex):
    """A union that defines the physical index of die.

    Attributes:
        dieId(int): The die id.
        cardId(int): The card id.
        seqNum(int): The die's unique global sequence number.
    """
    dieId: int
    cardId: int
    seqNum: int


#=========================== STRUCT =============================


class NodeBaseInfo(_vaml.nodeBaseInfo):
    """A struct that defines base information of node.

    Attributes:
        name(str): The name of node.
        majorMinor(NodeMajorMinor): The base attribtue of node.
    """
    name: str
    majorMinor: NodeMajorMinor


class DieBaseInfo(_vaml.dieBaseInfo):
    """A struct that defines die base information.

    Attributes:
        dieIndex(DieIndex): The physical index of die.
        vaccBaseInfo(NodeBaseInfo): The base information of vacc node.
        renderBaseInfo(NodeBaseInfo): The base information of render node.
        videoBaseInfo(NodeBaseInfo): The base information of video node.
    """
    dieIndex: DieIndex
    vaccBaseInfo: NodeBaseInfo
    renderBaseInfo: NodeBaseInfo
    videoBaseInfo: NodeBaseInfo


class AiDevInfo(_vaml.aiDevInfo):
    """A struct that defines ai node information.

    Attributes:
        dieIndex(DieIndex): The physical index of die.
        aiBaseInfo(NodeBaseInfo): The ai base information.
    """
    dieIndex: DieIndex
    aiBaseInfo: NodeBaseInfo


class CodecDevInfo(_vaml.codecDevInfo):
    """A struct that defines video encode/decode information.

    Attributes:
        dieIndex(DieIndex): The physical index of die.
        videoBaseInfo(NodeBaseInfo): The video base information.
    """
    dieIndex: DieIndex
    videoBaseInfo: NodeBaseInfo


class PciInfo(_vaml.pciInfo):
    """A struct that defines pci detail information.

    Attributes:
        busId(str): The card bus id.
        domain(int): The card domain information.
        bus(int): The number of bus where the pci device resides.
        card(int): The card number.
        pciId(PciCardId): The pci id information.
        pciSubId(PciSubSystemId): The pci sub id information.
        pcieCardBaseInfo(NodeBaseInfo): The pcie card base information.
        pcieCardVersionBaseInfo(NodeBaseInfo): The pcie card version base information.
        pcieCardCtlBaseInfo(NodeBaseInfo): The pcie card control base information.
    """
    busId: str
    domain: int
    bus: int
    card: int
    pciId: PciCardId
    pciSubId: PciSubSystemId
    pcieCardBaseInfo: NodeBaseInfo
    pcieCardVersionBaseInfo: NodeBaseInfo
    pcieCardCtlBaseInfo: NodeBaseInfo


class Capability(_vaml.capability):
    """A struct that defines capability of card.

    Attributes:
        aiCapability(int): The ai capability.
        videoCapability(int): The video capability.
    """
    aiCapability: int
    videoCapability: int


class CardInfo(_vaml.cardInfo):
    """A struct that defines card information.

    Attributes:
        cardId(int): The card id.
        uuid(str): The uuid.
        cardTypeName(str): The name of card type.
        pciInfo(PciInfo): The pci information.
        cardCapability(Capability): The card capability.
        manNodeBaseInfo(NodeBaseInfo): The managed node base information.
        dieNum(int): The number of die.
        dieInfo(List[DieBaseInfo]): The die information.
    """
    cardId: int
    uuid: str
    cardTypeName: str
    pciInfo: PciInfo
    cardCapability: Capability
    manNodeBaseInfo: NodeBaseInfo
    dieNum: int
    dieInfo: List[DieBaseInfo]


class FanSpeedInfo(_vaml.fanSpeedInfo):
    """A struct that defines fan speed information.

    Attributes:
        fanSpeedLevel(int): The fan speed level.
    """
    fanSpeedLevel: int


class PllClock(_vaml.pllClock):
    """A struct that defines die's clock frequency information, including 12 types of clocks.

    Attributes:
        unit(int): The uint is a bit that specifies which sub-module's clock frequency to get.
        clockArray(List[int]): The clockArray is 12 groups of clock values corresponding to the unit(its unit of measurement is ``HZ``).
    """
    unit: int
    clockArray: List[int]


class Temperature(_vaml.temperature):
    """A struct that defines die's temperature information, including 15 temperature sensor nodes can be collected.

    Attributes:
        unit(int): The uint is a bit that specifies which node's temperature to get.
        temperature(List[int]): The temperature is 15 groups of temperature values corresponding to the unit(its unit of measurement is ``°C``).
    """
    unit: int
    temperature: List[int]


class TempThreshold(_vaml.tempThreshold):
    """A struct that defines die's temperature alarm threshold, including 3 threshold modes.

    Attributes:
        unit(int): The uint is a bit that specifies which temperature alarm threshold to get.
        tempThreshold(List[int]): The tempThreshold is 3 threshold modes' values corresponding to the unit.
    """
    unit: int
    tempThreshold: List[int]


class PinVolt(_vaml.pinVolt):
    """A struct that defines die's voltage threshold of the PIN module, including 3 types of PIN voltage thresholds.

    Attributes:
        unit(int): The uint is a bit that specifies which PIN module's voltage threshold to get.
        pinVolt(List[int]): The pinVolt is 3 PIN voltage threshold modes' values corresponding to the unit(its unit of measurement is ``mV``).
    """
    unit: int
    pinVolt: List[int]


class Power(_vaml.power):
    """A struct that defines die's power, including 4 sub-modules.

    Attributes:
        unit(int): The uint is a bit that specifies which sub-module's power to get.
        power(List[int]): The power is 4 sub-modules' power values corresponding to the unit(its unit of measurement is ``uW``).
    """
    unit: int
    power: List[int]


class PowerCur(_vaml.powerCur):
    """A struct that defines die's power current, including 3 sub-modules.

    Attributes:
        unit(int): The uint is a bit that specifies which sub-module's power current to get.
        powerCur(List[int]): The powerCur is 3 sub-modules' power current values corresponding to the unit(its unit of measurement is ``mA``).
    """
    unit: int
    powerCur: List[int]


class PowerVolt(_vaml.powerVolt):
    """A struct that defines die's power voltage, including 3 sub-modules.

    Attributes:
        unit(int): The uint is a bit that specifies which sub-module's power voltage to get.
        powerVolt(List[int]): The powerVolt is 3 sub-modules' power voltage values corresponding to the unit(its unit of measurement is ``mV``).
    """
    unit: int
    powerVolt: List[int]


class MemUtilizationRate(_vaml.memUtilizationRate):
    """A struct that defines die's memory utilization rate.

    Attributes:
        total(int): The total memory (its unit of measurement is bytes).
        free(int): The free memory (its unit of measurement is bytes).
        used(int): The used memory (its unit of measurement is bytes).
        utilizationRate(int): The memory utilization rate(its unit of measurement is ``‱``).
    """
    total: int
    free: int
    used: int
    utilizationRate: int


class UtilizationRate(_vaml.utilizationRate):
    """A struct that defines die's utilization rate.

    Attributes:
        ai(int): The ai utilization rate (its unit of measurement is ``‱``).
        vdsp(int): The vdsp utilization rate (its unit of measurement is ``‱``).
        vemcu(int): The vencu utilization rate (its unit of measurement is ``‱``).
        vdmcu(int): The vdmcu utilization rate (its unit of measurement is ``‱``).
    """
    ai: int
    vdsp: int
    vemcu: int
    vdmcu: int


class McuUtilizationRate(_vaml.mcuUtilizationRate):
    """A struct that defines die's mcu utilization rate.

    Attributes:
        ai(int): The ai utilization rate (its unit of measurement is ``‱``).
        vdsp(List[int]): The vdsp utilization rate (its unit of measurement is ``‱``).
        vemcu(List[int]): The vencu utilization rate (its unit of measurement is ``‱``).
        vdmcu(List[int]): The vdmcu utilization rate (its unit of measurement is ``‱``).
    """
    ai: int
    vdsp: List[int]
    vemcu: List[int]
    vdmcu: List[int]


class ProcessInfo(_vaml.processInfo):
    """A struct that defines die's process information.

    Attributes:
        pid(int): The process id.
        memused(int): The memory usage.
        name(str): The process name.
    """
    pid: int
    memused: int
    name: str


# =========================== DEFINE =============================

vamlER_SUCCESS = _vaml.vamlER_SUCCESS
MAX_PLL_CLOCK_SUBMODULE = _vaml.MAX_PLL_CLOCK_SUBMODULE
MAX_TEMPERATURE_SUBMODULE = _vaml.MAX_TEMPERATURE_SUBMODULE
MAX_TEMP_THRESHOLD_SUBMODULE = _vaml.MAX_TEMP_THRESHOLD_SUBMODULE
MAX_POWER_SUBMODULE = _vaml.MAX_POWER_SUBMODULE
MAX_POWER_CURRENT_SUBMODULE = _vaml.MAX_POWER_CURRENT_SUBMODULE
MAX_POWER_VOLTAGE_SUBMODULE = _vaml.MAX_POWER_VOLTAGE_SUBMODULE
MAX_PIN_VOLT_SUBMODULE = _vaml.MAX_PIN_VOLT_SUBMODULE
MAX_PVT_VOLT_SUBMODULE = _vaml.MAX_PVT_VOLT_SUBMODULE
MAX_DIE_CMCU_NUM = _vaml.MAX_DIE_CMCU_NUM
MAX_DIE_VDSP_NUM = _vaml.MAX_DIE_VDSP_NUM
MAX_DIE_VDMCU_NUM = _vaml.MAX_DIE_VDMCU_NUM
MAX_DIE_VEMCU_NUM = _vaml.MAX_DIE_VEMCU_NUM

# =========================== API =============================


@err_check
def init() -> int:
    """Initialize the environment for VAML API.
    
    Hint:
        Please initialize before using vaml.

    Returns:
        int: The return code. 0 for success, False otherwise.
    """
    return _vaml.init()


@err_check
def shutDown() -> int:
    """Release the environment for VAML API.
    
    Hint:
        Please shutDown after using vaml.

    Returns:
        int: The return code. 0 for success, False otherwise.
    """
    return _vaml.shutDown()


# def getVersion() -> str:
#     """Get the VAML API version information.

#     Returns:
#         str: vaml version string.
#     """
#     return _vaml.getVersion()


@err_check
def setLogLevel(logLevel: LOG_LEVEL) -> int:
    """Set logger system for message logging.
    
    Args:
        logLevel(LOG_LEVEL): The log Level type.

    Returns:
        int: The return code. 0 for success, False otherwise.
    """
    return _vaml.setLogLevel(logLevel)


@err_check
def logMsg(logLevel: LOG_LEVEL, fmt: str, *args) -> int:
    """Write a message to the console or file.
    
    Args:
        logLevel(LOG_LEVEL): The log Level type.
        fmt(str): The string format.
        *args: The log information associated with fmt.
     
    Returns:
        int: The return code. 0 for success, False otherwise.
    """
    return _vaml.logMsg(logLevel, fmt, args[0], args[1])


def errorString(opStatus: OP_STATUS) -> str:
    """Convert the description of an error code from error code.
    
    Args:
        opStatus(OP_STATUS): The error code.
    
    Returns:
        str: The error code as constant string.
    """
    return _vaml.errorString(opStatus)
