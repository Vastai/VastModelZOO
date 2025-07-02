# Copyright (C) 2022-2023 VASTAI Technologies Co., Ltd. All Rights Reserved.
# coding: utf-8

__all__ = [
    "getAiDeviceCount", "getCodecDeviceCount", "getAiDevsInfo",
    "getCodecDevsInfo", "getDieIndexByDevIndex", "VastaiDevice"
]

from ._vaststream_pybind11 import vaml as _vaml
from .common import *
from .card import getCardCount, getCardsInfo, VastaiCard
from typing import List


# =========================== API =============================
def getAiDeviceCount() -> int:
    """Get the count of ai device.

    Returns:
        int: The ai device count.
    """
    return _vaml.getAiDeviceCount()


def getCodecDeviceCount() -> int:
    """Get the count of codec device.
    
    Returns:
        int: The codec device count.
    """
    return _vaml.getCodecDeviceCount()


def getAiDevsInfo(count: int) -> List[AiDevInfo]:
    """Get the information of ai device.
    
    Args:
        count(in): The count of ai device.

    Returns:
        List[AiDevInfo]: The ai device information list.
    """
    return _vaml.getAiDevInfo(count)


def getCodecDevsInfo(count: int) -> List[CodecDevInfo]:
    """Get the information of codec device.
    
    Args:
        count(in): The count of codec device.

    Returns:
        List[CodecDevInfo]: The codec device information list.
    """
    return _vaml.getCodecDevInfo(count)


def getDieIndexByDevIndex(devIndex: int) -> DieIndex:
    """Get the physical index of die by logical index.

    Args:
        devIndex(in): The logical index.

    Returns:
        DieIndex: The physical index of die.
    """
    return _vaml.getDieIndexByDevIndex(devIndex)


class VastaiDevice():
    """The device tool class.

    This class can get information about a device, including cards's information, 
    aiDevice's information, codecDevice's information, the physical index of die.
    You can get Card instance from this class by index.

    Examples:
        >>> device = vaml.VastaiDevice()
        >>> aiDeviceCount = device.aiDeviceCount
        >>> # get Card instance by index
        >>> cards = device.getCards(-1)
        >>> for card in cards:
        >>>     uuid = card.uuid
    """

    def __init__(self) -> None:
        pass

    @property
    def cardCount(self):
        """The card count of the device.
        """
        return self.getCardCount()

    @property
    def aiDeviceCount(self):
        """The ai device count.
        """
        return self.getAiDeviceCount()

    @property
    def codecDeviceCount(self):
        """The codec device count.
        """
        return self.getCodecDeviceCount()

    def getCardCount(self) -> int:
        """Get the count of card on the device.
        
        Returns:
            int: The card count.
        """
        return getCardCount()

    def getAiDeviceCount(self) -> int:
        """Get the count of ai device on the device.

        Returns:
            int: The ai device count.
        """
        return getAiDeviceCount()

    def getCodecDeviceCount(self) -> int:
        """Get the count of codec device on the device.
        
        Returns:
            int: The codec device count.
        """
        return getCodecDeviceCount()

    def getCardsInfo(self, index: int = -1) -> List[CardInfo]:
        """Get the information of card on the device.

        Args:
            index(int): The index of card(default -1), Vaule -1 means get all cards information.

        Returns:
            List[CardInfo]: The card information list.
        """
        if -1 <= index < self.cardCount:
            cardsInfo = getCardsInfo()
        else:
            raise RuntimeError(
                f"The index out of the card number range, get card infomation error, cardsInfo support -1 <= index < {self.cardCount}"
            )
        return cardsInfo if index == -1 else [cardsInfo[index]]

    def getAiDevicesInfo(self, index: int = -1) -> List[AiDevInfo]:
        """Get the information of ai device.

        Args:
            index(int): The index of ai device(default -1), Vaule -1 means get all ai devices information.

        Returns:
            List[AiDevInfo]: The ai device information list.
        """
        if -1 <= index < self.aiDeviceCount:
            aiDevicesInfo = getAiDevsInfo(self.aiDeviceCount)
        else:
            raise RuntimeError(
                f"The index out of the ai device number range, get ai device infomation error, aiDevicesInfo support -1 <= index < {self.aiDeviceCount}"
            )
        return aiDevicesInfo if index == -1 else [aiDevicesInfo[index]]

    def getCodecDevicesInfo(self, index: int = -1) -> List[CodecDevInfo]:
        """Get the information of codec device.
        
        Args:
            index(int): The index of codec device(default -1), Vaule -1 means get all codec devices information.

        Returns:
            List[CodecDevInfo]: The codec device information list.
        """
        if -1 <= index < self.codecDeviceCount:
            codecDevicesInfo = getCodecDevsInfo(self.codecDeviceCount)
        else:
            raise RuntimeError(
                f"The index out of the codec device number range, get codec device infomation error, codecDevicesInfo support -1 <= index < {self.codecDeviceCount}"
            )
        return codecDevicesInfo if index == -1 else [codecDevicesInfo[index]]

    def getDieIndexByDevIndex(self, index: int) -> DieIndex:
        """Get the physical index of die by logical index.
        
        Args:
            index(int): The logical index of die.

        Returns:
            DieIndex: The physical index of die(it is a struct, including dieId、cardId and seqNum).
        """
        if 0 <= index < self.aiDeviceCount:
            dieIndex = getDieIndexByDevIndex(index)
        else:
            raise RuntimeError(
                f"The index out of the die  number range in a device, get dieIndex infomation error, dieIndex support 0 <= index < {self.aiDeviceCount}"
            )
        return dieIndex

    def getCards(self, index: int = -1) -> List[VastaiCard]:
        """Get an instance of card in the device.

        Args:
            index(int): The index of card(default -1), Vaule -1 means get all cards intance.

        Returns:
            List[VastaiCard]: The list of cards in a device.
        """
        if -1 <= index < self.cardCount:
            cardsInfo = self.getCardsInfo(index)
        else:
            raise RuntimeError(
                f"The index out of the card number range in a device, get cards instance error, getCards support -1 <= index < {self.cardCount}"
            )
        cards = []
        for cardInfo in cardsInfo:
            card = VastaiCard(cardInfo.uuid)
            cards.append(card)
        return cards
