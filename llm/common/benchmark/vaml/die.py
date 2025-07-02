# Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# coding: utf-8

__all__ = [
    "getDieCount", "getDiesInfo", "getDieHandleByIndex",
    "getDieHandleByDieIndex", "getPllClocks", "getTemperature", "getPower",
    "getPinVolt", "getPowerCur", "getPowerVolt", "getTempThreshold",
    "getMemUtilizationRate", "getMcuUtilizationRate", "getUtilizationRate",
    "getRunningProcesses", "getNetWorksRunningInfo", "VastaiDie"
]

from ._vaststream_pybind11 import vaml as _vaml
from .common import *
from typing import List


# =========================== API =============================
def getDieCount(cardHandle: int) -> int:
    """Get the count of die of the specified handle of card.
    
    Args:
        cardHandle(int): The Handle of the specified card.
    
    Returns:
        int: The die number of a card.
    """
    return _vaml.getDieCount(cardHandle)


def getDiesInfo(cardHandle: int) -> List[DieBaseInfo]:
    """Get the information of die of the specified handle of card.
    
    Args:
        cardHandle(int): The Handle of the specified card.

    Returns:
        List[DieBaseInfo]: The die information list of a card.
    """
    dieInfo = []
    return _vaml.getDieInfo(cardHandle, dieInfo)


def getDieHandleByIndex(cardHandle: int, index: int) -> int:
    """Get the handle of die by card handle and index.
    
    Args:
        cardHandle(int): The Handle of the specified card.
        index(int): The index of the die.
    
    Returns:
        int: The die handle value.
    """
    return _vaml.getDieHandleByIndex(cardHandle, index)


def getDieHandleByDieIndex(dieIndex: DieIndex) -> int:
    """Get the handle of die by its index.
   
    Hint:
        The propertie of the parameter ``dieIndex`` is readable only, it must be obtained from ``dieInfo`` or 
        ``getDieIndexByDevIndex`` or ``aiDevInfo`` or ``codecDevInfo``, users cannot create it by themselves.
    
    Args:
        dieIndex(int): The dieIndex (it is a struct, including dieId、cardId and seqNum).

    Returns:
        int: The die handle value.

    Examples:
        >>> device = vaml.VastaiDevice()
        >>> dieIndex = device.getDieIndexByDevIndex(index = 0)
        >>> aiDevicesInfo = device.getAiDevicesInfo()
        >>> for aiDeviceInfo in aiDevicesInfo:
        >>>     dieIndex = aiDeviceInfo.dieIndex
        >>> # or you can get get index like this
        >>> diesInfo = getDiesInfo(cardHandle)
        >>> for dieInfo in diesInfo:
        >>>     dieIndex = dieInfo.dieIndex
    """
    return _vaml.getDieHandleByDieIndex(dieIndex)


def getPllClocks(dieHandle: int) -> PllClock:
    """Get the pll clocks information by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        PllClock: The die's clock frequency information.
    
    Examples:
        >>> pllclocks = vaml.getPllClocks(dieHandle)
        >>> MAX_PLL_CLOCK_SUBMODULE = vaml.MAX_PLL_CLOCK_SUBMODULE
        >>> assert pllclocks.unit == (2**MAX_PLL_CLOCK_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_PLL_CLOCK_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (pllclocks.unit & flag) != 0:
        >>>         clock_frequency = int(pllclocks.clockArray[nIndex] / 1000000)
    """
    pllClock = _vaml.pllClock()
    ret = _vaml.getPllClocks(dieHandle, pllClock)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get PllClocks error, ret: {ret}.")
    return pllClock


def getTemperature(dieHandle: int) -> Temperature:
    """Get the temperature information by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        Temperature: The die's temperature.
    
    Examples:
        >>> temperatures = vaml.getTemperature(dieHandle)
        >>> MAX_TEMPERATURE_SUBMODULE = vaml.MAX_TEMPERATURE_SUBMODULE
        >>> assert temperatures.unit == (2**MAX_TEMPERATURE_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_TEMPERATURE_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (temperatures.unit & flag) != 0:
        >>>         temperature = temperatures.temperature[nIndex] / 100.0
    """
    temperature = _vaml.temperature()
    ret = _vaml.getTemperature(dieHandle, temperature)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get temperature error, ret: {ret}.")
    return temperature


def getPower(dieHandle: int) -> Power:
    """Get the power information by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        Power: The die's power.
    
    Examples:
        >>> powers = vaml.getPower(dieHandle)
        >>> MAX_POWER_SUBMODULE = vaml.MAX_POWER_SUBMODULE
        >>> assert powers.unit == (2**MAX_POWER_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_POWER_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (powers.unit & flag) != 0:
        >>>         power = powers.power[nIndex] / 1000000.0
    """
    power = _vaml.power()
    ret = _vaml.getPower(dieHandle, power)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get power error, ret: {ret}.")
    return power


def getPinVolt(dieHandle: int) -> PinVolt:
    """Get the voltage threshold of the PIN module by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        PinVolt: The die's pin voltage information.
    
    Examples:
        >>> pinVolts = vaml.getPinVolt(dieHandle)
        >>> MAX_PIN_VOLT_SUBMODULE = vaml.MAX_PIN_VOLT_SUBMODULE
        >>> assert pinVolts.unit == (2**MAX_PIN_VOLT_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_PIN_VOLT_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (pinVolts.unit & flag) != 0:
        >>>         power = pinVolts.pinVolt[nIndex] / 1.0
    """
    pinVolt = _vaml.pinVolt()
    ret = _vaml.getPinVolt(dieHandle, pinVolt)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get pin voltage error, ret: {ret}.")
    return pinVolt


def getPowerCur(dieHandle: int) -> PowerCur:
    """Get the power current by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        PowerCur: The die's power current.
    
    Examples:
        >>> powerCurs = vaml.getPowerCur(dieHandle)
        >>> MAX_POWER_CURRENT_SUBMODULE = vaml.MAX_POWER_CURRENT_SUBMODULE
        >>> assert powerCurs.unit == (2**MAX_POWER_CURRENT_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_POWER_CURRENT_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (powerCurs.unit & flag) != 0:
        >>>         powerCur = powerCurs.powerCur[nIndex] / 1000.0
    """
    powerCur = _vaml.powerCur()
    ret = _vaml.getPowerCur(dieHandle, powerCur)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get power current error, ret: {ret}.")
    return powerCur


def getPowerVolt(dieHandle: int) -> PowerVolt:
    """Get the power voltage by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        PowerVolt: The die's power voltage.
    
    Examples:
        >>> powerVolts = vaml.getPowerVolt(dieHandle)
        >>> MAX_POWER_VOLTAGE_SUBMODULE = vaml.MAX_POWER_VOLTAGE_SUBMODULE
        >>> assert powerVolts.unit == (2**MAX_POWER_VOLTAGE_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_POWER_VOLTAGE_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (powerVolts.unit & flag) != 0:
        >>>         powerVolt = powerVolts.powerVolt[nIndex] / 1000.0
    """
    powerVolt = _vaml.powerVolt()
    ret = _vaml.getPowerVolt(dieHandle, powerVolt)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get power voltage error, ret: {ret}.")
    return powerVolt


def getTempThreshold(dieHandle: int) -> TempThreshold:
    """Get the temperature threshold by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        TempThreshold: The die's temperature threshold.
    
    Examples:
        >>> tempThresholds = vaml.getTempThreshold(dieHandle)
        >>> MAX_TEMP_THRESHOLD_SUBMODULE = vaml.MAX_TEMP_THRESHOLD_SUBMODULE
        >>> assert tempThresholds.unit == (2**MAX_TEMP_THRESHOLD_SUBMODULE) - 1
        >>> flag = 0
        >>> for nIndex in range(MAX_TEMP_THRESHOLD_SUBMODULE):
        >>>     flag = 1 << nIndex
        >>>     if (tempThresholds.unit & flag) != 0:
        >>>         tempThreshold = tempThresholds.tempThreshold[nIndex]
    """
    tempThreshold = _vaml.tempThreshold()
    ret = _vaml.getTempThreshold(dieHandle, tempThreshold)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get temperature threshold error, ret: {ret}.")
    return tempThreshold


def getMemUtilizationRate(dieHandle: int) -> MemUtilizationRate:
    """Get the memory utilization rate by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        MemUtilizationRate: The die's memory utilization rate.
    
    Examples:
        >>> memUtilizationRate = vaml.getMemUtilizationRate(dieHandle)
        >>> total = memUtilizationRate.total
        >>> free = memUtilizationRate.free
        >>> used = memUtilizationRate.used
        >>> utilizationRate = memUtilizationRate.utilizationRate / 100.0
    """
    return _vaml.getMemUtilizationRate(dieHandle)


def getMcuUtilizationRate(dieHandle: int) -> McuUtilizationRate:
    """Get the mcu utilization rate by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        McuUtilizationRate: The die's mcu utilization rate.
    
    Examples:
        >>> mcuUtilizationRate = vaml.getMcuUtilizationRate(dieHandle)
        >>> ai = mcuUtilizationRate.ai / 100.0
        >>> for nIndex in range(vaml.MAX_DIE_VDSP_NUM):
        >>>     vdsp = mcuUtilizationRate.vdsp[nIndex] / 100.0
        >>> for nIndex in range(vaml.MAX_DIE_VEMCU_NUM):
        >>>     vemcu = mcuUtilizationRate.vemcu[nIndex] / 100.0
        >>> for nIndex in range(vaml.MAX_DIE_VDMCU_NUM):
        >>>     vdmcu = mcuUtilizationRate.vdmcu[nIndex] / 100.0
    """
    mcuUtilizationRate = _vaml.mcuUtilizationRate()
    ret = _vaml.getMcuUtilizationRate(dieHandle, mcuUtilizationRate)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get mcu utilization rate error, ret: {ret}.")
    return mcuUtilizationRate


def getUtilizationRate(dieHandle: int) -> UtilizationRate:
    """Get the utilization rate by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        UtilizationRate: The die's utilization rate.
    
    Examples:
        >>> dieUtilizationRate = vaml.getUtilizationRate(dieHandle)
        >>> ai = dieUtilizationRate.ai / 100.0
        >>> vdsp = dieUtilizationRate.vdsp / 100.0
        >>> vemcu = dieUtilizationRate.vemcu / 100.0
        >>> vdmcu = dieUtilizationRate.vdmcu / 100.0
    """
    return _vaml.getUtilizationRate(dieHandle)


def getRunningProcesses(dieHandle: int) -> List[ProcessInfo]:
    """Get the process information by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        List[ProcessInfo]: The die's process information.
    """
    processInfo = []
    return _vaml.getRunningProcesses(dieHandle, processInfo)


def getNetWorksRunningInfo(dieHandle: int) -> List[ProcessInfo]:
    """Get the netWork process information by the specified handle of die.
    
    Args:
        dieHandle(int): The die handle.
    
    Returns:
        List[ProcessInfo]: The die's netWork process information.
    """
    modelProcessInfo = []
    return _vaml.getNetWorkRunningInfo(dieHandle, modelProcessInfo)


class VastaiDie():
    """Die tool class.

    The die tool class can get information about a die, including die's information(dieHandle、dieId、cardId), 
    some statis(pllClocks、temperature、tempThreshold、power、pinVolt、powerCur、powerVolt、memUtilizationRate、
    mcuUtilizationRate、utilizationRate、processInfo).
    
    Args:
        dieIndex(DieIndex): The dieIndex of the die.
        cardHandle(int): The card handle.
        index(int): The die index.
   
    Hint:
        The propertie of the parameter ``dieIndex``  is readable only, it must be obtained from ``dieInfo`` or 
        ``getDieIndexByDevIndex`` or ``aiDevInfo`` or ``codecDevInfo``, users cannot create it by themselves.
    
    Examples:
        >>> die = vaml.VastaiDie(...)
        >>> dieId = die.dieId
        >>> cardId = die.cardId
        >>> temperature = die.getTemperature()
    """

    def __init__(self,
                 dieIndex: DieIndex = None,
                 cardHandle: int = None,
                 index: int = None) -> None:
        if dieIndex == None and cardHandle == None and index == None:
            raise RuntimeError(
                "Get die error, need to set parameters(dieIndex or cardHandle and index)."
            )
        else:
            if dieIndex != None:
                self.dieHandle = getDieHandleByDieIndex(dieIndex)
                self._dieId = dieIndex.dieId
                self._cardId = dieIndex.cardId

            elif cardHandle != None:
                diesInfo = getDiesInfo(cardHandle)
                indexs = []
                for idx in range(len(diesInfo)):
                    indexs.append(idx)
                if index != None:
                    if index in indexs:
                        self.dieHandle = getDieHandleByIndex(cardHandle, index)
                        i = indexs.index(index)
                        self._dieId = diesInfo[i].dieIndex.dieId
                        self._cardId = diesInfo[i].dieIndex.cardId
                    else:
                        raise RuntimeError(
                            f"The index out of the die count range in a card, get die error, need to reset index, VastaiDie support 0 <= index < {len(diesInfo)}."
                        )
                else:
                    raise RuntimeError(f"Get die error, need to set index, VastaiDie support 0 <= index < {len(diesInfo)}.")
            else:
                raise RuntimeError("get die error, need to reset cardHandle.")

    @property
    def dieId(self):
        """The die id."""
        return self._dieId

    @property
    def cardId(self):
        """The card id."""
        return self._cardId

    def getPllClock(self) -> PllClock:
        """Get the pll clocks information of the die.

        Returns:
            PllClock: The die's clock frequency information.

        Examples:
            >>> die = VastaiDie(...)
            >>> pllclocks = die.getPllClocks()
            >>> MAX_PLL_CLOCK_SUBMODULE = vaml.MAX_PLL_CLOCK_SUBMODULE
            >>> assert pllclocks.unit == (2**MAX_PLL_CLOCK_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_PLL_CLOCK_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (pllclocks.unit & flag) != 0:
            >>>         clock_frequency = int(pllclocks.clockArray[nIndex] / 1000000)
        """
        pllClock = getPllClocks(self.dieHandle)
        return pllClock

    def getTemperature(self) -> Temperature:
        """Get the temperature information of the die.

        Returns:
            Temperature: The die's temperature.

        Examples:
            >>> die = VastaiDie(...)
            >>> temperatures = die.getTemperature()
            >>> MAX_TEMPERATURE_SUBMODULE = vaml.MAX_TEMPERATURE_SUBMODULE
            >>> assert temperatures.unit == (2**MAX_TEMPERATURE_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_TEMPERATURE_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (temperatures.unit & flag) != 0:
            >>>         temperature = temperatures.temperature[nIndex] / 100.0
        """
        temperature = getTemperature(self.dieHandle)
        return temperature

    def getPower(self) -> Power:
        """Get the power information of the die.

        Returns:
            Power: The die's power.

        Examples:
            >>> die = VastaiDie(...)
            >>> powers = die.getPower()
            >>> MAX_POWER_SUBMODULE = vaml.MAX_POWER_SUBMODULE
            >>> assert powers.unit == (2**MAX_POWER_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_POWER_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (powers.unit & flag) != 0:
            >>>         power = powers.power[nIndex] / 1000000.0
        """
        power = getPower(self.dieHandle)
        return power

    def getPinVolt(self) -> PinVolt:
        """Get the voltage threshold of the PIN module of the die.

        Returns:
            PinVolt: The die's pin voltage information.

        Examples:
            >>> die = VastaiDie(...)
            >>> pinVolts = die.getPinVolt()
            >>> MAX_PIN_VOLT_SUBMODULE = vaml.MAX_PIN_VOLT_SUBMODULE
            >>> assert pinVolts.unit == (2**MAX_PIN_VOLT_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_PIN_VOLT_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (pinVolts.unit & flag) != 0:
            >>>         power = pinVolts.pinVolt[nIndex] / 1.0
        """
        pinVolt = getPinVolt(self.dieHandle)
        return pinVolt

    def getPowerCur(self) -> PowerCur:
        """Get the power current information of the die.

        Returns:
            PowerCur: The die's power current.

        Examples:
            >>> die = VastaiDie(...)
            >>> powerCurs = die.getPowerCur()
            >>> MAX_POWER_CURRENT_SUBMODULE = vaml.MAX_POWER_CURRENT_SUBMODULE
            >>> assert powerCurs.unit == (2**MAX_POWER_CURRENT_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_POWER_CURRENT_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (powerCurs.unit & flag) != 0:
            >>>         powerCur = powerCurs.powerCur[nIndex] / 1000.0
        """
        powerCur = getPowerCur(self.dieHandle)
        return powerCur

    def getPowerVolt(self) -> PowerVolt:
        """Get the power voltage information of the die.

        Returns:
            PowerVolt: The die's power voltage.

        Examples:
            >>> die = VastaiDie(...)
            >>> pinVolts = die.getPinVolt()
            >>> MAX_PIN_VOLT_SUBMODULE = vaml.MAX_PIN_VOLT_SUBMODULE
            >>> assert pinVolts.unit == (2**MAX_PIN_VOLT_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_PIN_VOLT_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (pinVolts.unit & flag) != 0:
            >>>         power = pinVolts.pinVolt[nIndex] / 1.0
        """
        powerVolt = getPowerVolt(self.dieHandle)
        return powerVolt

    def getTempThreshold(self) -> TempThreshold:
        """Get the temperature threshold of the die.

        Returns:
            TempThreshold: The die's temperature threshold.

        Examples:
            >>> die = VastaiDie(...)
            >>> tempThresholds = die.getTempThreshold()
            >>> MAX_TEMP_THRESHOLD_SUBMODULE = vaml.MAX_TEMP_THRESHOLD_SUBMODULE
            >>> assert tempThresholds.unit == (2**MAX_TEMP_THRESHOLD_SUBMODULE) - 1
            >>> flag = 0
            >>> for nIndex in range(MAX_TEMP_THRESHOLD_SUBMODULE):
            >>>     flag = 1 << nIndex
            >>>     if (tempThresholds.unit & flag) != 0:
            >>>         tempThreshold = tempThresholds.tempThreshold[nIndex]
        """
        tempThreshold = getTempThreshold(self.dieHandle)
        return tempThreshold

    def getMemUtilizationRate(self) -> MemUtilizationRate:
        """Get the memory utilization rate of die.

        Returns:
            MemUtilizationRate: The die's memory utilization rate.

        Examples:
            >>> die = VastaiDie(...)
            >>> memUtilizationRate = die.getMemUtilizationRate()
            >>> total = memUtilizationRate.total
            >>> free = memUtilizationRate.free
            >>> used = memUtilizationRate.used
            >>> utilizationRate = memUtilizationRate.utilizationRate / 100.0
        """
        memUtilizationRate = getMemUtilizationRate(self.dieHandle)
        return memUtilizationRate

    def getMcuUtilizationRate(self) -> McuUtilizationRate:
        """Get the mcu utilization rate of die.

        Returns:
            McuUtilizationRate: The die's mcu utilization rate.

        Examples:
            >>> die = VastaiDie(...)
            >>> mcuUtilizationRate = die.getMcuUtilizationRate()
            >>> ai = mcuUtilizationRate.ai / 100.0
            >>> for nIndex in range(vaml.MAX_DIE_VDSP_NUM):
            >>>     vdsp = mcuUtilizationRate.vdsp[nIndex] / 100.0
            >>> for nIndex in range(vaml.MAX_DIE_VEMCU_NUM):
            >>>     vemcu = mcuUtilizationRate.vemcu[nIndex] / 100.0
            >>> for nIndex in range(vaml.MAX_DIE_VDMCU_NUM):
            >>>     vdmcu = mcuUtilizationRate.vdmcu[nIndex] / 100.0
        """
        mcuUtilizationRate = getMcuUtilizationRate(self.dieHandle)
        return mcuUtilizationRate

    def getUtilizationRate(self) -> UtilizationRate:
        """Get the utilization rate of die.

        Returns:
            UtilizationRate: The die's utilization rate.

        Examples:
            >>> die = VastaiDie(...)
            >>> dieUtilizationRate = die.getUtilizationRate()
            >>> ai = dieUtilizationRate.ai / 100.0
            >>> vdsp = dieUtilizationRate.vdsp / 100.0
            >>> vemcu = dieUtilizationRate.vemcu / 100.0
            >>> vdmcu = dieUtilizationRate.vdmcu / 100.0
        """
        utilizationRate = getUtilizationRate(self.dieHandle)
        return utilizationRate

    def getRunningProcesses(self) -> List[ProcessInfo]:
        """Get the process information of die.
        
        Returns:
            List[ProcessInfo]: The die's process information.
        """
        processInfo = getRunningProcesses(self.dieHandle)
        return processInfo

    def getNetWorksRunningInfo(self) -> List[ProcessInfo]:
        """Get the netWork process information of die.
        
        Returns:
            List[ProcessInfo]: The die's netWork process information.
        """
        modelProcessInfo = getNetWorksRunningInfo(self.dieHandle)
        return modelProcessInfo
