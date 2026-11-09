"""Cart-pole simulator and renderer (self-contained, no gym / pygame).

Physics: explicit Euler with h = 0.02 s, force held for FRAME_SKIP sub-steps
(cc_common.macro_step). Default parameters M = 1.0, m = 0.1, l = 0.5 (half-length),
g = 9.8, no friction.

Rendering uses the classic cart-pole layout (a 600 x 400 scene, 4.8 m wide, cart
50 x 30, pole 10 wide and 2l long, axle circle), squeezed to image_size x image_size
like a 600x400 -> square resize. Each pixel is the average of a 4 x 4 grid of
sub-samples, so edges are anti-aliased and sub-pixel motion stays visible.
"""
from __future__ import annotations

import numpy as np

from cc_common import FRAME_SKIP, H, PHYS, macro_step

SCENE_W, SCENE_H = 600.0, 400.0      # scene units ("screen pixels" of the classic layout)
WORLD_W = 4.8                        # metres across the scene
SCALE = SCENE_W / WORLD_W            # scene units per metre
CART_Y = 100.0                       # cart centre height
CART_W, CART_H = 50.0, 30.0
POLE_W = 10.0
AXLE_OFFSET = CART_H / 4.0
X_THRESHOLD = 2.4                    # episode ends beyond this

WHITE = np.array([255, 255, 255], np.float32)
BLACK = np.array([0, 0, 0], np.float32)
POLE_RGB = np.array([202, 152, 101], np.float32)
AXLE_RGB = np.array([129, 132, 203], np.float32)


class Renderer:
    def __init__(self, image_size=128, supersample=4):
        self.size, self.ss = image_size, supersample
        n = image_size * supersample
        # scene coordinates of every sub-sample; row 0 is the top of the image
        gx = (np.arange(n) + 0.5) / n * SCENE_W
        gy = SCENE_H - (np.arange(n) + 0.5) / n * SCENE_H
        self.gx, self.gy = np.meshgrid(gx, gy)

    def __call__(self, x, theta, pole_half_length=PHYS['l']):
        gx, gy = self.gx, self.gy
        img = np.empty(gx.shape + (3,), np.float32)
        img[:] = WHITE
        img[np.abs(gy - CART_Y) <= 0.5] = BLACK                                  # track
        cx = x * SCALE + SCENE_W / 2.0
        img[(np.abs(gx - cx) <= CART_W / 2) & (np.abs(gy - CART_Y) <= CART_H / 2)] = BLACK
        px, py = cx, CART_Y + AXLE_OFFSET                                         # pivot
        dx, dy = gx - px, gy - py
        c, s = np.cos(theta), np.sin(theta)
        u = dx * c - dy * s               # across the pole
        v = dx * s + dy * c               # along the pole (theta > 0 tilts right)
        pole_len = 2.0 * pole_half_length * SCALE
        img[(np.abs(u) <= POLE_W / 2) & (v >= -POLE_W / 2) & (v <= pole_len - POLE_W / 2)] = POLE_RGB
        img[dx ** 2 + dy ** 2 <= (POLE_W / 2) ** 2] = AXLE_RGB
        k = self.ss
        img = img.reshape(self.size, k, self.size, k, 3).mean((1, 3))
        return np.round(img).astype(np.uint8)


class CartPole:
    """reset / step interface. State s = (x, x_dot, theta, theta_dot).

    phys = dict(M, m, l, g) sets the physics (default cc_common.PHYS). It can be changed per
    episode with reset_to_state(state, phys); the pole is drawn with its true length 2l."""

    def __init__(self, image_size=128, frame_skip=FRAME_SKIP, action_range=(-10.0, 10.0),
                 theta_threshold=1.2, render=True, seed=None, phys=None):
        self.frame_skip = frame_skip
        self.low, self.high = action_range
        self.theta_threshold = theta_threshold
        self.render = Renderer(image_size) if render else None
        self.rng = np.random.RandomState(seed)
        self.phys = dict(PHYS if phys is None else phys)
        self.state = None

    def obs(self):
        if self.render is None:
            return np.zeros((1, 1, 3), np.uint8)
        return self.render(self.state[0], self.state[2], pole_half_length=self.phys['l'])

    def reset_to_state(self, state, phys=None):
        if phys is not None:
            self.phys = dict(phys)
        self.state = np.asarray(state, np.float64).copy()
        return self.obs(), self.state.astype(np.float32)

    def reset(self, init_range=0.05):
        return self.reset_to_state(self.rng.uniform(-init_range, init_range, size=4))

    def step(self, u):
        u = float(np.clip(u, self.low, self.high))
        self.state = macro_step(self.state, u, n=self.frame_skip, h=H, **self.phys)
        x, th = self.state[0], self.state[2]
        done = bool(abs(x) > X_THRESHOLD or abs(th) > self.theta_threshold)
        return self.obs(), self.state.astype(np.float32), done
