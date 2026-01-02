from abc import abstractmethod
from typing import Any, List


class AbstractDBHandler():

    @abstractmethod
    def connect(self) -> None:
        pass
