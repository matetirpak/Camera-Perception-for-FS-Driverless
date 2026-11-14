"""Cone position from its 7 keypoints (Dhall et al. 2019, Sec. IV-C), of which only the translation is used.

The paper put PnP inside RANSAC; here all points are used at once (SQPnP, then Levenberg-Marquardt) and the worst
keypoint is dropped instead. The points lie on two lines plus the top, so most subsets RANSAC draws are
degenerate. The model points have their origin on the ground at the cone's axis, so the translation is the cone's
base in the optical frame.
"""
from __future__ import annotations

import cv2
import numpy as np

from ..common.cone_model import model_points


def solve_cone_position(keypoints_px: np.ndarray, K: np.ndarray, model: np.ndarray | None = None,
                        reprojection_px: float = 3.0, max_dropped: int = 2) -> np.ndarray | None:
    """(3,) position of the cone's base in the optical frame (x right, y down, z forward), or None if PnP fails or
    puts the cone behind the camera. Keypoints and ``K`` have to use the same pixel convention."""
    obj = np.asarray(model_points() if model is None else model, np.float64)
    img = np.asarray(keypoints_px, np.float64).reshape(-1, 2)
    idx = np.arange(len(obj))
    t = None
    for _ in range(max_dropped + 1):
        try:
            ok, rvec, tvec = cv2.solvePnP(obj[idx], img[idx].reshape(-1, 1, 2), K, None, flags=cv2.SOLVEPNP_SQPNP)
        except cv2.error:
            # SQPnP asserts instead of returning when the keypoints carry no spread, which a collapsed
            # regression on a tiny or hidden cone can produce
            return None
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(obj[idx], img[idx].reshape(-1, 1, 2), K, None, rvec, tvec)
        t = tvec.ravel()
        projected, _ = cv2.projectPoints(obj[idx], rvec, tvec, K, None)
        error = np.linalg.norm(projected.reshape(-1, 2) - img[idx], axis=1)
        if error.max() <= reprojection_px or len(idx) <= 5:
            break
        idx = np.delete(idx, np.argmax(error))
    return t if t[2] > 0 else None


# Upright cones
#
# A cone stands upright and, being round, looks the same from every side, so of PnP's six unknowns only the position
# of its base is left once "up" is known. The rotation follows from the position: the cone's vertical axis is "up",
# and the edge keypoints are where the lines of sight touch the body, the same geometry the labels are made with
# (``cone_model.silhouette_keypoints``). With the camera's height above the ground the base lies on a known plane
# and only two unknowns remain. All cones of an image are solved together, Levenberg-Marquardt on the reprojection
# error with a numerical Jacobian, which costs about as much for twenty cones as for one.

UPRIGHT_MODES = ("upright", "ground")


def upright_keypoints(base: np.ndarray, up: np.ndarray) -> np.ndarray:
    """(N, 7, 3) keypoints in the optical frame of cones with their base at (N, 3) ``base``, for a camera at the
    origin and ``up`` the upward direction in the optical frame."""
    from ..common.cone_model import EDGE_HEIGHTS, TOP, body_radius

    base = np.atleast_2d(base)
    to_camera = -base - (-base @ up)[:, None] * up
    d = np.linalg.norm(to_camera, axis=1, keepdims=True)
    w = to_camera / d
    # left as seen from the camera, as in silhouette_keypoints
    left = np.cross(w, up)
    # the six edge points: left side top to bottom, then the right side, as (6,) heights, radii and sides
    z = np.array([*EDGE_HEIGHTS, *EDGE_HEIGHTS])
    r = body_radius(z)
    side = np.repeat([1.0, -1.0], 3)
    along = np.sqrt(np.clip(1 - (r / d) ** 2, 0.0, None))
    edges = (base[:, None] + z[:, None] * up + (r * r / d)[..., None] * w[:, None]
             + (side * r * along)[..., None] * left[:, None])
    return np.concatenate([(base + TOP * up)[:, None], edges], axis=1)


def _project(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    z = points[..., 2:3]
    return np.concatenate([K[0, 0] * points[..., :1] / z + K[0, 2], K[1, 1] * points[..., 1:2] / z + K[1, 2]], -1)


def _initial_base(keypoints: np.ndarray, K: np.ndarray, up: np.ndarray) -> np.ndarray:
    """A rough base position from the height of the keypoints in the image: the cone's size gives the distance,
    the middle of the two bottom points the direction."""
    from ..common.cone_model import BODY_BOTTOM, TOP

    bottom = (keypoints[:, 3] + keypoints[:, 6]) / 2
    height_px = np.maximum(bottom[:, 1] - keypoints[:, 0, 1], 1.0)
    depth = K[1, 1] * (TOP - BODY_BOTTOM) / height_px
    ray = np.stack([(bottom[:, 0] - K[0, 2]) / K[0, 0], (bottom[:, 1] - K[1, 2]) / K[1, 1],
                    np.ones(len(bottom))], axis=1)
    return ray * depth[:, None] - BODY_BOTTOM * up


def solve_upright_positions(keypoints_px: np.ndarray, K: np.ndarray, up, camera_height: float | None = None,
                            reprojection_px: float = 3.0, max_dropped: int = 2,
                            iterations: int = 20) -> list[np.ndarray | None]:
    """Base positions in the optical frame of upright cones from (N, 7, 2) keypoints, None where it fails.

    ``up`` is the upward direction in the optical frame, (0, -1, 0) for a level camera. With ``camera_height`` (metres
    above the ground under the cones) the base is held on the ground plane and only two unknowns are solved, without
    it three. Like ``solve_cone_position`` the worst keypoint is dropped while one is more than ``reprojection_px``
    off, at most ``max_dropped`` times."""
    kp = np.asarray(keypoints_px, np.float64).reshape(-1, 7, 2)
    n = len(kp)
    if n == 0:
        return []
    up = np.asarray(up, np.float64)
    up = up / np.linalg.norm(up)
    base0 = _initial_base(kp, K, up)
    if camera_height is None:
        origin, axes = np.zeros(3), np.eye(3)
        params = base0.copy()
    else:
        # base = origin + params @ axes, two directions across the ground plane
        origin = -camera_height * up
        e1 = np.cross(up, [0.0, 0.0, 1.0])
        e1 /= np.linalg.norm(e1)
        axes = np.stack([e1, np.cross(e1, up)])
        params = (base0 - origin) @ axes.T

    def residuals(p, weights):
        """Weighted reprojection error of the cones' keypoints for one or several stacked sets of parameters."""
        sets = len(p) // n
        projected = _project(upright_keypoints(origin + p @ axes, up), K).reshape(sets, n, 7, 2)
        return ((projected - kp) * weights[..., None]).reshape(len(p), -1)

    k = params.shape[1]
    weights = np.ones((n, 7))
    eps = 1e-5
    eye = np.eye(k)
    for _ in range(max_dropped + 1):
        damping = np.full(n, 1e-3)
        r = residuals(params, weights)
        cost = np.einsum("ij,ij->i", r, r)
        for _ in range(iterations):
            # all columns of the numerical Jacobian in one evaluation
            shifted = (params[None] + eps * eye[:, None]).reshape(k * n, k)
            J = ((residuals(shifted, weights).reshape(k, n, -1) - r[None]) / eps).transpose(1, 2, 0)
            JtJ = np.einsum("nik,nil->nkl", J, J)
            g = np.einsum("nik,ni->nk", J, r)
            A = JtJ + damping[:, None, None] * (eye * np.diagonal(JtJ, axis1=1, axis2=2)[:, None] + 1e-9 * eye)
            step = -np.linalg.solve(A, g[..., None])[..., 0]
            candidate = params + step
            r_new = residuals(candidate, weights)
            cost_new = np.einsum("ij,ij->i", r_new, r_new)
            better = np.isfinite(cost_new) & (cost_new < cost)
            params = np.where(better[:, None], candidate, params)
            r = np.where(better[:, None], r_new, r)
            cost = np.where(better, cost_new, cost)
            damping = np.where(better, damping * 0.3, damping * 10.0)
            # a tenth of a millimetre is far below what the keypoints can tell apart
            if np.all(np.abs(step).max(axis=1) < 1e-4):
                break
        error = np.linalg.norm(r.reshape(n, 7, 2), axis=2)
        worst = np.argmax(error, axis=1)
        drop = (error.max(axis=1) > reprojection_px) & (weights.sum(axis=1) > 5)
        if not drop.any():
            break
        weights[np.flatnonzero(drop), worst[drop]] = 0.0
    base = origin + params @ axes
    ok = np.isfinite(base).all(axis=1) & (base[:, 2] > 0)
    return [b if good else None for b, good in zip(base, ok)]
