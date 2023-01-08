from dataclasses import dataclass


@dataclass
class Annotation:
    id: str
    tag: str
    start_pos: int
    end_pos: int
    phrase: str
