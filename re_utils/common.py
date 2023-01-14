import json
from dataclasses import dataclass
from typing import List, TypeVar, Callable, Any


@dataclass
class NerAnnotation:
    id: str
    tag: str
    start_ch_pos: int
    end_ch_pos: int
    phrase: str

    start_word_pos: int = -1
    end_word_pos: int = -1

    start_token_pos: int = -1
    end_token_pos: int = -1


@dataclass
class ReAnnotation:
    id: str
    arg1: str
    arg2: str
    tag: str


T = TypeVar("T")


def lower_bound(arr: List[T], element: float, key: Callable[[T], float] = lambda x: x) -> int:
    left = -1
    right = len(arr)

    while left < right - 1:
        mid = (left + right) // 2

        if element <= key(arr[mid]):
            right = mid
        else:
            left = mid
    return right


def upper_bound(arr: List[T], element: float, key: Callable[[T], float] = lambda x: x) -> int:
    left = -1
    right = len(arr)

    while left < right - 1:
        mid = (left + right) // 2

        if element >= key(arr[mid]):
            left = mid
        else:
            right = mid
    return right


def binary_search(arr: List[T], element: float, key: Callable[[T], float] = lambda x: x) -> int:
    left = -1
    right = len(arr)

    while left < right - 1:
        mid = (left + right) // 2

        if element > key(arr[mid]):
            left = mid
        elif element < key(arr[mid]):
            right = mid
        else:
            return mid
    return -1


def load_jsonl(filename: str) -> List[Any]:
    with open(filename, encoding="utf-8") as f:
        result = []
        for line in f:
            result.append(json.loads(line))
        return result


def save_jsonl(obj: List[Any], file: str):
    with open(file, "w") as json_file:
        for item in obj:
            json.dump(item, json_file)
            json_file.write("\n")


def save_json(obj: Any, file: str):
    with open(file, "w") as json_file:
        json.dump(obj, json_file)


def load_json(path: str):
    with open(path, "r") as file:
        data = json.load(file)
    return data
