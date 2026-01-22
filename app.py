import json
import numpy as np


class Homography:
    """
    Mapeia pontos de um painel arbitrário (pixels do PDF)
    para um sistema retangular normalizado (u,v).
    """
    def __init__(self, src_pts, dst_pts):
        self.H = self._compute_h(src_pts, dst_pts)
        self.Hinv = np.linalg.inv(self.H)

    def _compute_h(self, src, dst):
        A = []
        for (x, y), (u, v) in zip(src, dst):
            A.append([x, y, 1, 0, 0, 0, -u*x, -u*y, -u])
            A.append([0, 0, 0, x, y, 1, -v*x, -v*y, -v])
        A = np.array(A)
        _, _, Vt = np.linalg.svd(A)
        H = Vt[-1].reshape(3, 3)
        return H / H[2, 2]

    def to_uv(self, x, y):
        p = np.array([x, y, 1.0])
        q = self.H @ p
        return q[0]/q[2], q[1]/q[2]

    def to_xy(self, u, v):
        p = np.array([u, v, 1.0])
        q = self.Hinv @ p
        return q[0]/q[2], q[1]/q[2]


class LinearAxis:
    """
    Mapeamento linear pixel <-> valor físico
    """
    def __init__(self, px, values):
        a, b = np.polyfit(px, values, 1)
        self.a = a
        self.b = b

    def value(self, px):
        return self.a * px + self.b

    def pixel(self, v):
        return (v - self.b) / self.a


def load_geometry(path="capture.json"):
    with open(path) as f:
        data = json.load(f)

    panels = {}
    for name, corners in data["panel_corners"].items():
        src = corners
        dst = [(0,0), (1,0), (1,1), (0,1)]
        panels[name] = Homography(src, dst)

    axes = {}
    for axis, ticks in data["axis_ticks"].items():
        px = [t["x"] for t in ticks] if axis != "ground_roll_ft" else [t["y"] for t in ticks]
        val = [t["value"] for t in ticks]
        axes[axis] = LinearAxis(px, val)

    return panels, axes



