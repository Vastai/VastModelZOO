# Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# coding: utf-8

__all__ = [
    "getCardCount", "getCardsInfo", "getCardHandleByUUID",
    "getCardHandleByPciBusId", "getCardHandleByIndex", "getCardHandleByCardId",
    "getUUID", "getPciInfo", "getCapability", "getFanSpeedInfo",
    "getManageNodeAttribute", "getDriverSWVersion", "VastaiCard"
]

from ._vaststream_pybind11 import vaml as _vaml
from .common import *
from .die import getDieCount, getDiesInfo, VastaiDie
from typing import List


# =========================== API =============================
def getCardCount() -> int:
    """Get the count of card.

    Returns:
        int: The card count.
    """
    return _vaml.getCardCount()


def getCardsInfo() -> List[CardInfo]:
    """Get the information of card.

    Returns:
        List[CardInfo]: The card information list.
    """
    cardInfo = []
    return _vaml.getCardInfo(cardInfo)


def getCardHandleByUUID(uuid: str) -> int:
    """Get the card handle by uuid.
    
    Args:
        uuid(str): The uuid of a card.

    Returns:
        int: The card handle value.
    """
    return _vaml.getCardHandleByUUID(uuid)


def getCardHandleByPciBusId(pciBusId: str) -> int:
    """Get the card handle by pci bus id.
   
    Args:
        pciBusId(str): The pci bus id of a card.

    Returns:
        int: The card handle value.
    """
    return _vaml.getCardHandleByPciBusId(pciBusId)


def getCardHandleByIndex(index: int) -> int:
    """Get the card handle by index of card.
    
    Args:
        index(int): The card index.

    Returns:
        int: The card handle value.
    """
    return _vaml.getCardHandleByIndex(index)


def getCardHandleByCardId(cardId: int) -> int:
    """Get the card handle by card id.
    
    Args:
        cardId(int): The card id.

    Returns:
        int: The card handle value.
    """
    return _vaml.getCardHandleByCardId(cardId)


def getUUID(cardHandle: int) -> str:
    """Get the uuid by the specified handle of card.
    
    Args:
        cardHandle(int): The card handle.

    Returns:
        str: The uuid of card.
    """
    return _vaml.getUUID(cardHandle)


def getPciInfo(cardHandle: int) -> PciInfo:
    """Get the pci information by the specified handle of card.
    
    Args:
        cardHandle(int): The card handle.

    Returns:
        PciInfo: The pci information of card.
    """
    return _vaml.getPciInfo(cardHandle)


def getCapability(cardHandle: int) -> Capability:
    """Get the capability by the specified handle of card.
    
    Args:
        cardHandle(int): The card handle.

    Returns:
        Capability: The capability of card(including aiCapability and videoCapability).
    """
    capability = _vaml.capability()
    ret = _vaml.getCapability(cardHandle, capability)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(
            f"Get the capability by the specified handle of card error, ret: {ret}.")
    return capability


def getFanSpeedInfo(cardHandle: int) -> FanSpeedInfo:
    """Get the fan speed information of card.
    
    Args:
        cardHandle(int): The card handle.

    Returns:
        FanSpeedInfo: The fan speed level information of card.
    """
    speedInfo = _vaml.fanSpeedInfo()
    ret = _vaml.getFanSpeedInfo(cardHandle, speedInfo)
    if ret != _vaml.vamlER_SUCCESS:
        raise RuntimeError(f"Get the fan speed error, ret: {ret}.")
    return speedInfo


def getManageNodeAttribute() -> NodeBaseInfo:
    """Get the attributes of managed node.
    
    Returns:
        NodeBaseInfo: The base information of node.
    """
    return _vaml.getManageNodeAttribute()


def getDriverSWVersion() -> str:
    """Get the version information of driver.

    Returns:
        str: The version information of driver.
    """
    return _vaml.getDriverSWVersion()


class VastaiCard():
    """Card tool class.

    This class can get information about a card, including card's information
    (cardHandle、uuid、cardId、pciBusIds、cardTypeName、pciInfo、capability、fanSpeedInfo), 
    dies's infomation.You can get this class instance from VastaiDevice, or create 
    an instance by card's uuid or pciBusId or cardId or index.

    Args:
        uuid(str): The uuid of a card.
        pciBusId(str): The pci bud id of a card.
        cardId(int): The card id.
        index(int): The card index.
        
    Examples:
        >>> card = vaml.VastaiCard(...)
        >>> uuid = card.uuid
        >>> diecount = card.dieCount
        >>> # get Die instance by index
        >>> dies = card.getDies(-1)
        >>> for die in dies:
        >>>     pllclocks = die.getPllClock()
    """

    def __init__(self,
                 uuid: str = None,
                 pciBusId: str = None,
                 cardId: int = None,
                 index: int = None) -> None:
        cardsInfo = getCardsInfo()
        uuids = []
        pciBusIds = []
        cardIds = []
        indexs = []
        for idx in range(len(cardsInfo)):
            uuids.append(cardsInfo[idx].uuid)
            pciBusIds.append(cardsInfo[idx].pciInfo.busId)
            cardIds.append(cardsInfo[idx].cardId)
            indexs.append(idx)

        if uuid != None:
            if uuid in uuids:
                self.cardHandle = getCardHandleByUUID(uuid)
                i = uuids.index(uuid)
                self._uuid = cardsInfo[i].uuid
                self._cardId = cardsInfo[i].cardId
                self._pciBusId = cardsInfo[i].pciInfo.busId
                self._cardTypeName = cardsInfo[i].cardTypeName
                self._dieNum = cardsInfo[i].dieNum
            else:
                raise RuntimeError(
                    "The uuid is not in carsInfo, get card error, need to reset uuid."
                )
        elif pciBusId != None:
            if pciBusId in pciBusIds:
                self.cardHandle = getCardHandleByPciBusId(pciBusId)
                i = pciBusIds.index(pciBusId)
                self._uuid = cardsInfo[i].uuid
                self._cardId = cardsInfo[i].cardId
                self._pciBusId = cardsInfo[i].pciInfo.busId
                self._cardTypeName = cardsInfo[i].cardTypeName
                self._dieNum = cardsInfo[i].dieNum
            else:
                raise RuntimeError(
                    "The pciBusId is not in carsInfo, get card error, need to reset pciBusId."
                )
        elif cardId != None:
            if cardId in cardIds:
                self.cardHandle = getCardHandleByCardId(cardId)
                i = cardIds.index(cardId)
                self._uuid = cardsInfo[i].uuid
                self._cardId = cardsInfo[i].cardId
                self._pciBusId = cardsInfo[i].pciInfo.busId
                self._cardTypeName = cardsInfo[i].cardTypeName
                self._dieNum = cardsInfo[i].dieNum
            else:
                raise RuntimeError(
                    "The cardId is not in carsInfo, get card error, need to reset cardId."
                )
        elif index != None:
            if index in indexs:
                self.cardHandle = getCardHandleByIndex(index)
                i = indexs.index(index)
                self._uuid = cardsInfo[i].uuid
                self._cardId = cardsInfo[i].cardId
                self._pciBusId = cardsInfo[i].pciInfo.busId
                self._cardTypeName = cardsInfo[i].cardTypeName
                self._dieNum = cardsInfo[i].dieNum
            else:
                raise RuntimeError(
                    "The index out of the card count range, get card error, need to reset index."
                )
        else:
            raise RuntimeError(
                "Get card error, need to set a parameter(uuid or pciBusId or cardId or index)."
            )

    @property
    def pciBusId(self):
        """The pci bus id of the card.
        """
        return self._pciBusId

    @property
    def cardId(self):
        """The id of the card.
        """
        return self._cardId

    @property
    def uuid(self):
        """The uuid of the card.
        """
        return self._uuid

    @property
    def cardTypeName(self):
        """The type name of the card.
        """
        return self._cardTypeName

    @property
    def dieCount(self):
        """The die number of the card.
        """
        return self._dieNum

    def getUUID(self) -> str:
        """Get the uuid of the card.
        
        Returns:
            str: The uuid of the card.
        """
        uuid = getUUID(self.cardHandle)
        return uuid

    def getPciInfo(self) -> PciInfo:
        """Get the pci information of the card.
        
        Returns:
            PciInfo: The pci information of the card.
        """
        pciInfo = getPciInfo(self.cardHandle)
        return pciInfo

    def getCapability(self) -> Capability:
        """Get the capability of the card.
        
        Returns:
            Capability: The capability(including aiCapability and videoCapability) of the card.
        """
        capability = getCapability(self.cardHandle)
        return capability

    def getFanSpeed(self) -> FanSpeedInfo:
        """Get the fan speed level of the card.
        
        Returns:
            FanSpeedInfo: The fan speed level information of the card.
        """
        fanSpeedInfo = getFanSpeedInfo(self.cardHandle)
        return fanSpeedInfo

    def getDieCount(self) -> int:
        """Get the count of the die on the card.
        
        Returns:
            int: The die number of a card.
        """
        dieCount = getDieCount(self.cardHandle)
        return dieCount

    def getDiesInfo(self, index: int = -1) -> List[DieBaseInfo]:
        """Get the information of the die on the card.
        
        Args:
            index(int): The index of die(default -1), Vaule -1 means get all dies information.

        Returns:
            List[DieBaseInfo]: The die information list of a card.
        """
        if -1 <= index < self.dieCount:
            diesInfo = getDiesInfo(self.cardHandle)
        else:
            raise RuntimeError(
                f"The index out of the die number range in a card, get die infomation error, getDiesInfo support -1 <= index < {self.dieCount}"
            )
        return diesInfo if index == -1 else [diesInfo[index]]

    def getDies(self, index: int = -1) -> List[VastaiDie]:
        """Get an instance of die on the card.
        
        Args:
            index(int): The index of die(default -1), Vaule -1 means get all dies instance.

        Returns:
            List[VastaiDie]: The list of dies on the card.
        """
        if -1 <= index < self.dieCount:
            diesInfo = self.getDiesInfo(index)
        else:
            raise RuntimeError(
                f"The index out of the die number range in a card, get die instance error, getDies support -1 <= index < {self.dieCount}"
            )
        dies = []
        for dieInfo in diesInfo:
            die = VastaiDie(dieInfo.dieIndex)
            dies.append(die)
        return dies