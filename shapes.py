from enum import Enum


class Geometry(Enum):
    EMPTY = -1
    CIRCLE = 0
    QUADRAT = 1
    TRIANGLE = 2
    HEXAGON = 3


class Color(Enum):
    EMPTY = -1
    RED = 0
    GREEN = 1
    BLUE = 2
    YELLOW = 3
