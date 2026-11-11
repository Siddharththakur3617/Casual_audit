"""Acrobot (double pendulum, torque on the elbow) plus an independent distractor block.

State s = (theta1, omega1, theta2, omega2, p, pdot):
  theta1  angle of link 1 from hanging straight down (positive = counter-clockwise)
  theta2  angle of link 2 relative to link 1
  p       position (m) of the distractor block on its own track below the arm
Angles are stored unwrapped; the image is periodic in them.

Dynamics: Spong / Sutton form with uniform rods (centre of mass at l/2, I = m l^2 / 12), no joint
friction, torque u on joint 2. The distractor is a damped spring, p'' = -K p - C p', not driven by the
action and not coupled to the arm. Explicit Euler with h = 0.01 s, torque held for FRAME_SKIP = 4
sub-steps, so one stored transition is 0.04 s (as for cart-pole).

Rendering: square scene 7.6 m across, arm pivot in the centre (reach at most l1 + l2 = 3 m), the
distractor's track along the bottom, outside the arm's reach. Each pixel averages a 4 x 4 grid of
sub-samples (anti-aliased, as in sim/cartpole.py).
"""
from __future__ import annotations

import numpy as np

H = 0.01
FRAME_SKIP = 4
PHYS = dict(m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8)
K_SPRING = float(np.pi ** 2)       # (2 pi / T)^2 with period T = 2 s
C_DAMP = 0.3                       # small viscous damping of the distractor
MAX_W1, MAX_W2 = 4 * np.pi, 9 * np.pi   # an episode stops before a step that exceeds these (as Gym's clip limits)

SCENE = 7.6                        # metres across (square)
TRACK_Y = -3.55                    # distractor track height (pivot at 0, 0)
BLOCK_W, BLOCK_H = 0.4, 0.25
LINK_W = 0.14
JOINT_R = 0.09

WHITE = np.array([255, 255, 255], np.float32)
BLACK = np.array([0, 0, 0], np.float32)
LINK1_RGB = np.array([202, 152, 101], np.float32)
LINK2_RGB = np.array([101, 152, 202], np.float32)
JOINT_RGB = np.array([129, 132, 203], np.float32)
BLOCK_RGB = np.array([40, 160, 90], np.float32)


def accelerations(th1, th2, w1, w2, u, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8):
    """Joint accelerations (Spong / Sutton 'book' form) for uniform rods."""
    lc1, lc2 = l1 / 2.0, l2 / 2.0
    I1, I2 = m1 * l1 ** 2 / 12.0, m2 * l2 ** 2 / 12.0
    c2, s2 = np.cos(th2), np.sin(th2)
    d1 = m1 * lc1 ** 2 + m2 * (l1 ** 2 + lc2 ** 2 + 2 * l1 * lc2 * c2) + I1 + I2
    d2 = m2 * (lc2 ** 2 + l1 * lc2 * c2) + I2
    phi2 = m2 * lc2 * g * np.cos(th1 + th2 - np.pi / 2)
    phi1 = (-m2 * l1 * lc2 * w2 ** 2 * s2 - 2 * m2 * l1 * lc2 * w2 * w1 * s2
            + (m1 * lc1 + m2 * l1) * g * np.cos(th1 - np.pi / 2) + phi2)
    a2 = (u + d2 / d1 * phi1 - m2 * l1 * lc2 * w1 ** 2 * s2 - phi2) / (m2 * lc2 ** 2 + I2 - d2 ** 2 / d1)
    a1 = -(d2 * a2 + phi1) / d1
    return a1, a2


def euler_substep(s, u, h=H, **phys):
    s = np.asarray(s, dtype=np.float64)
    th1, w1, th2, w2, p, v = (s[..., k] for k in range(6))
    a1, a2 = accelerations(th1, th2, w1, w2, u, **phys)
    pa = -K_SPRING * p - C_DAMP * v
    return np.stack([th1 + h * w1, w1 + h * a1, th2 + h * w2, w2 + h * a2, p + h * v, v + h * pa], axis=-1)


def macro_step(s, u, n=FRAME_SKIP, h=H, **phys):
    """One stored transition: the torque is held for n sub-steps."""
    for _ in range(n):
        s = euler_substep(s, u, h=h, **phys)
    return s


def energy(s, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.8):
    """Mechanical energy of the arm (the distractor is separate)."""
    th1, w1, th2, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    lc1, lc2 = l1 / 2.0, l2 / 2.0
    I1, I2 = m1 * l1 ** 2 / 12.0, m2 * l2 ** 2 / 12.0
    v1sq = (lc1 * w1) ** 2
    v2sq = (l1 * w1) ** 2 + (lc2 * (w1 + w2)) ** 2 + 2 * l1 * lc2 * w1 * (w1 + w2) * np.cos(th2)
    T = 0.5 * m1 * v1sq + 0.5 * I1 * w1 ** 2 + 0.5 * m2 * v2sq + 0.5 * I2 * (w1 + w2) ** 2
    V = -m1 * g * lc1 * np.cos(th1) - m2 * g * (l1 * np.cos(th1) + lc2 * np.cos(th1 + th2))
    return T + V


class Renderer:
    def __init__(self, image_size=128, supersample=4):
        self.size, self.ss = image_size, supersample
        n = image_size * supersample
        g = (np.arange(n) + 0.5) / n * SCENE - SCENE / 2
        self.gx, self.gy = np.meshgrid(g, -g)          # row 0 is the top of the image

    def _segment(self, ax, ay, bx, by, r):
        dx, dy = bx - ax, by - ay
        L2 = max(dx * dx + dy * dy, 1e-12)
        t = np.clip(((self.gx - ax) * dx + (self.gy - ay) * dy) / L2, 0, 1)
        return (self.gx - ax - t * dx) ** 2 + (self.gy - ay - t * dy) ** 2 <= r * r

    def __call__(self, th1, th2, p, l1=1.0, l2=1.0):
        gx, gy = self.gx, self.gy
        img = np.empty(gx.shape + (3,), np.float32)
        img[:] = WHITE
        img[np.abs(gy - TRACK_Y) <= 0.02] = BLACK                                       # track
        img[(np.abs(gx - p) <= BLOCK_W / 2) & (np.abs(gy - TRACK_Y) <= BLOCK_H / 2)] = BLOCK_RGB
        ex, ey = l1 * np.sin(th1), -l1 * np.cos(th1)                                     # elbow
        tx, ty = ex + l2 * np.sin(th1 + th2), ey - l2 * np.cos(th1 + th2)                # tip
        img[self._segment(0.0, 0.0, ex, ey, LINK_W / 2)] = LINK1_RGB
        img[self._segment(ex, ey, tx, ty, LINK_W / 2)] = LINK2_RGB
        img[gx ** 2 + gy ** 2 <= JOINT_R ** 2] = BLACK
        img[(gx - ex) ** 2 + (gy - ey) ** 2 <= JOINT_R ** 2] = JOINT_RGB
        k = self.ss
        img = img.reshape(self.size, k, self.size, k, 3).mean((1, 3))
        return np.round(img).astype(np.uint8)


class Acrobot:
    """reset_to_state / step interface, as sim.cartpole.CartPole. step returns done=True when a joint
    speed exceeds its limit (the caller then drops that step)."""

    def __init__(self, image_size=128, frame_skip=FRAME_SKIP, action_range=(-3.0, 3.0), render=True,
                 seed=None, phys=None):
        self.frame_skip = frame_skip
        self.low, self.high = action_range
        self.render = Renderer(image_size) if render else None
        self.rng = np.random.RandomState(seed)
        self.phys = dict(PHYS if phys is None else phys)
        self.state = None

    def obs(self):
        if self.render is None:
            return np.zeros((1, 1, 3), np.uint8)
        s = self.state
        return self.render(s[0], s[2], s[4], l1=self.phys['l1'], l2=self.phys['l2'])

    def reset_to_state(self, state, phys=None):
        if phys is not None:
            self.phys = dict(phys)
        self.state = np.asarray(state, np.float64).copy()
        return self.obs(), self.state.astype(np.float32)

    def step(self, u):
        u = float(np.clip(u, self.low, self.high))
        self.state = macro_step(self.state, u, n=self.frame_skip, h=H, **self.phys)
        done = bool(abs(self.state[1]) > MAX_W1 or abs(self.state[3]) > MAX_W2)
        return self.obs(), self.state.astype(np.float32), done
