from dataclasses import dataclass
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
        if abs(sxr) < 1e-8:  # sxr = 0 i.e. wall and line are parallel
            return None
        t = np.cross(self.p1 - q1, s) / sxr  # crossing with s and solving for t
        u = np.cross(self.p1 - q1, r) / sxr  # likewise with r, solving for u
        if 0 <= t <= 1 and 0 <= u <= 1:  # within the segments
            return self.p1 + t * r
        return None
