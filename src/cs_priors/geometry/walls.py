from dataclasses import dataclass
from typing import Self
import numpy as np


@dataclass
class Wall:
    p1: np.ndarray
    p2: np.ndarray

    def reflect_point(self, p):
        # reflect point p across the wall segment
        ap = p - self.p1
        ab = self.p2 - self.p1
        abn = ab / np.linalg.norm(ab)
        proj = self.p1 + abn * np.dot(ap, abn)  # proj is the point on the wall
        return proj + (proj - p)

    def intersection(self, q1, q2):
        # return the intersection point of line p->q with the wall segment
        # Solving p_1 + t * (p_2 - p_1) = q_1 + u * (q_2 - q_1)
        r = self.p2 - self.p1
        s = q2 - q1
        sxr = np.cross(s, r)
        if abs(sxr) < 1e-12:  # sxr = 0 i.e. wall and line are parallel
            return None
        t = np.cross(self.p1 - q1, s) / sxr  # crossing with s and solving for t
        u = np.cross(self.p1 - q1, r) / sxr  # likewise with r, solving for u
        print(f"t: {t}, u: {u}")
        if 0 <= t <= 1 and 0 <= u <= 1:  # within the segments
            return self.p1 + t * r
        return None


@dataclass
class Reflection:
    pos: np.ndarray
    source: np.ndarray
    wall_seq: list[Wall]
    previous: Self | None


@dataclass
class Path:
    mic: np.ndarray
    source: np.ndarray
    intersection_seq: np.ndarray


def compute_reflections(sources, walls):
    """Compute first and second order reflections of a source across walls."""
    reflections = []
    for source in sources:
        for i in range(len(walls)):
            reflected_source = walls[i].reflect_point(source.get_position())
            reflections.append(
                Reflection(
                    pos=reflected_source,
                    source=source.get_position(),
                    wall_seq=[walls[i]],
                    previous=None,
                )
            )
            for j in range(i + 1, len(walls)):
                reflected_twice = walls[j].reflect_point(reflected_source)
                reflections.append(
                    Reflection(
                        pos=reflected_twice,
                        source=source.get_position(),
                        wall_seq=[walls[i], walls[j]],
                        previous=reflected_source,
                    )
                )
    return reflections


# def compute_path(reflections: list[Reflection], mic: np.ndarray):
#     paths = []
#     for ref in reflections:
#         intersection_seq = []
#         for wall in reversed(ref.wall_seq):
#             intersection = wall.intersection(mic, ref.pos)
#             if intersection is None:
#                 print(f"No intersection found for mic {mic} and reflection {ref}")
#                 break  # I hope this only ends the current wall loop
#             intersection_seq.append(intersection)
#         paths.append(
#             Path(mic=mic, source=ref.source, intersection_seq=intersection_seq)
#         )
#     return paths
