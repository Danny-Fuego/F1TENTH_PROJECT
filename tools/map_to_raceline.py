#!/usr/bin/env python3
"""
SLAM map  ->  track (centerline + widths, cleaned map)  ->  min-curvature racing line + speed profile.

Usage:
    python3 map_to_raceline.py ~/maps/second_floor.yaml
    python3 map_to_raceline.py ~/maps/second_floor.yaml --out ~/F1TENTH_PROJECT/pure_pursuit/racelines

Outputs (in --out, default = folder of the map), with NAME = map file name:
    NAME_centerline.csv     x,y,w_tr_right,w_tr_left  (header line starts with #)
    NAME_track.pgm/.yaml    cleaned map: only the loop is free, everything else is wall (for the sim)
    NAME_raceline.csv       x,y,v  no header  (Pure Pursuit)
    NAME_raceline_slow.csv  same line, v * SLOW_FACTOR  (first real-car runs)
    NAME_preview.png        racing line coloured by speed over the original map

Needs:  pip install numpy scipy opencv-python pillow pyyaml trajectory-planning-helpers quadprog
If 'import quadprog' fails with "undefined symbol":  pip install --no-binary quadprog --force-reinstall quadprog

Works for a map with ONE loop around ONE central block (like the 2nd floor).
"""
import argparse, os
import numpy as np, cv2, yaml
from PIL import Image
from scipy import ndimage as ndi, optimize
from scipy.interpolate import splprep, splev

# ======================= TUNING (edit here) =======================
# --- track extraction ---
FREE_THRESH      = 250    # pgm value >= this counts as free space
OPEN_KERNEL_PX   = 13     # removes thin free slivers (laser leaking through glass/doors)
FILL_HOLES_PX    = 2000   # free-space holes smaller than this (chairs, people, noise) get filled
RING_DILATE_PX   = 70     # how far out from the central block the corridor ring may extend (70 px = 3.5 m)
RING_HOLE_PX     = 3000   # holes inside the ring smaller than this get filled
RING_OPEN_PX     = 15     # final smoothing of the ring
WIDTH_MINFILT    = 15     # width = min over this many neighbouring points (conservative)
# --- racing line ---
CAR_WIDTH        = 0.30   # m
SAFETY_MARGIN    = 0.25   # m on EACH side
KAPPA_MAX        = 1.0    # 1/m  (car max is about tan(0.42)/0.3175 = 1.4)
SMOOTH_S         = 1.0    # smoothing of the reference before optimisation
STEP_OPT         = 0.25   # m between optimiser points
STEP_OUT         = 0.10   # m between output points
# --- speed profile ---
V_MAX            = 7.0    # m/s
AY_MAX           = 4.0    # m/s^2 lateral grip
AX_ACCEL         = 3.0    # m/s^2
AX_DECEL         = 4.0    # m/s^2
SLOW_FACTOR      = 0.5
# ==================================================================

import trajectory_planning_helpers as tph

# --- compatibility fixes for newer scipy / numpy (no need to edit the installed library) ---
def _dist_to_p(t_glob, path, p):
    s = np.array(splev(float(np.ravel(t_glob)[0]), path)).ravel()
    return float(np.hypot(*(np.asarray(p) - s)))
class _OptShim:
    @staticmethod
    def fmin(*a, **k):
        return float(np.ravel(optimize.fmin(*a, **k))[0])
tph.spline_approximation.dist_to_p = _dist_to_p
tph.spline_approximation.optimize = _OptShim


def load_map(yaml_path):
    with open(yaml_path) as f:
        m = yaml.safe_load(f)
    img_path = m['image'] if os.path.isabs(m['image']) else os.path.join(os.path.dirname(yaml_path), m['image'])
    img = np.array(Image.open(img_path).convert('L'))
    if m.get('negate', 0):
        img = 255 - img
    return img, float(m['resolution']), float(m['origin'][0]), float(m['origin'][1])


def largest(mask):
    lab, n = ndi.label(mask)
    if n == 0:
        raise SystemExit('no free space found')
    return lab == (np.argmax(ndi.sum(mask, lab, range(1, n + 1))) + 1)


def fill_small_holes(mask, max_px):
    holes, n = ndi.label(~mask)
    sizes = ndi.sum(~mask, holes, range(1, n + 1))
    small = np.isin(holes, np.where(sizes < max_px)[0] + 1)
    return mask | small


def disk(px):
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (px, px))


# =============================== 1. EXTRACT TRACK ===============================
def extract_track(img, res, ox, oy):
    H = img.shape[0]
    free = img >= FREE_THRESH
    free = cv2.morphologyEx(free.astype(np.uint8), cv2.MORPH_OPEN, disk(OPEN_KERNEL_PX)).astype(bool)
    free = fill_small_holes(largest(free), FILL_HOLES_PX)

    # central block = biggest non-free region that doesn't touch the image border
    nf = ~free
    lab, n = ndi.label(nf)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    sizes = ndi.sum(nf, lab, range(1, n + 1))
    cand = [(s, i + 1) for i, s in enumerate(sizes) if (i + 1) not in border]
    if not cand:
        raise SystemExit('no enclosed central block found - is the loop closed in the map?')
    inner = lab == max(cand)[1]

    # corridor ring around the block
    ring = free & ndi.binary_dilation(inner, iterations=RING_DILATE_PX) & ~inner
    ring = largest(ring)
    holes, hn = ndi.label(~ring)
    hs = ndi.sum(~ring, holes, range(1, hn + 1))
    for i, s in enumerate(hs):
        if s < RING_HOLE_PX and not inner[holes == i + 1].any():
            ring[holes == i + 1] = True
    ring = cv2.morphologyEx(ring.astype(np.uint8), cv2.MORPH_OPEN, disk(RING_OPEN_PX)).astype(bool)
    ring = largest(ring)

    # centerline = boundary of the points closer to the inner wall than the outer wall
    d_in = ndi.distance_transform_edt(~inner)
    d_out = ndi.distance_transform_edt(ring | inner)
    half = inner | (ring & (d_in <= d_out))
    half = cv2.morphologyEx(half.astype(np.uint8), cv2.MORPH_CLOSE, disk(9))
    cs, _ = cv2.findContours(half, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cnt = max(cs, key=cv2.contourArea)[:, 0, :]          # (col,row)
    rows, cols = cnt[:, 1], cnt[:, 0]
    xs = ox + (cols + 0.5) * res
    ys = oy + (H - 1 - rows + 0.5) * res
    w = np.minimum(d_in, d_out)[rows, cols] * res

    tck, _ = splprep([xs, ys, w], s=len(xs) * 0.02, per=True)
    L = np.sum(np.hypot(np.diff(xs), np.diff(ys)))
    N = int(L / 0.1)
    X, Y, Wd = splev(np.linspace(0, 1, N, endpoint=False), tck)
    Wd = ndi.minimum_filter1d(Wd, size=WIDTH_MINFILT, mode='wrap')
    if 0.5 * np.sum(X * np.roll(Y, -1) - np.roll(X, -1) * Y) < 0:   # make it counter-clockwise
        X, Y, Wd = X[::-1], Y[::-1], Wd[::-1]
    center = np.column_stack([X, Y, Wd, Wd])
    print(f'[track] loop {L:.1f} m, {N} pts, half-width min {Wd.min():.2f} / median {np.median(Wd):.2f} m')
    return center, ring


# =============================== 2. RACING LINE ===============================
def make_raceline(center, ring, res, ox, oy):
    H, Wimg = ring.shape

    def to_px(p):
        return int((p[0] - ox) / res - 0.5), int(H - 1 - ((p[1] - oy) / res - 0.5))

    def ray(p, d):                      # distance to wall along direction d in the cleaned map
        for r in np.arange(0, 3.0, 0.01):
            c, w = to_px(p + d * r)
            if not (0 <= w < H and 0 <= c < Wimg) or not ring[w, c]:
                return r
        return 3.0

    ref = tph.spline_approximation.spline_approximation(
        track=center, k_reg=3, s_reg=SMOOTH_S, stepsize_prep=1.0, stepsize_reg=STEP_OPT, debug=False)
    cl = np.vstack([ref[:, :2], ref[:1, :2]])
    cx, cy, A, nv = tph.calc_splines.calc_splines(path=cl)

    # real wall distance along each normal, then the min of neighbours (covers the chord between points)
    ref[:, 2] = [ray(ref[i, :2], nv[i]) for i in range(len(ref))]
    ref[:, 3] = [ray(ref[i, :2], -nv[i]) for i in range(len(ref))]
    for k in (2, 3):
        ref[:, k] = np.min([np.roll(ref[:, k], j) for j in (-2, -1, 0, 1, 2)], axis=0)

    spl_len = tph.calc_spline_lengths.calc_spline_lengths(cx, cy)
    psi, kap, dkap = tph.calc_head_curv_an.calc_head_curv_an(
        cx, cy, np.arange(len(ref)), np.zeros(len(ref)), calc_dcurv=True)
    alpha, reft, nvt = tph.iqp_handler.iqp_handler(
        reftrack=ref, normvectors=nv, A=A, spline_len=spl_len, psi=psi, kappa=kap, dkappa=dkap,
        kappa_bound=KAPPA_MAX, w_veh=CAR_WIDTH + 2 * SAFETY_MARGIN, print_debug=False, plot_debug=False,
        stepsize_interp=STEP_OPT, iters_min=5, curv_error_allowed=0.05)[:3]

    line, _, cxo, cyo, si, ti, *_ = tph.create_raceline.create_raceline(
        refline=reft[:, :2], normvectors=nvt, alpha=alpha, stepsize_interp=STEP_OUT)
    _, kappa = tph.calc_head_curv_an.calc_head_curv_an(cxo, cyo, si, ti)
    el = np.hypot(*np.diff(np.vstack([line, line[:1]]), axis=0).T)

    ggv = np.array([[0, AX_DECEL, AY_MAX], [V_MAX + 1, AX_DECEL, AY_MAX]])
    axm = np.array([[0, AX_ACCEL], [V_MAX + 1, AX_ACCEL]])
    v = tph.calc_vel_profile.calc_vel_profile(ggv=ggv, ax_max_machines=axm, v_max=V_MAX, kappa=kappa,
                                              el_lengths=el, closed=True, drag_coeff=0.0, m_veh=3.5)

    # check: clearance of the car centre from the walls of the cleaned map
    dist = ndi.distance_transform_edt(ring) * res
    clear = np.array([dist[to_px(p)[1], to_px(p)[0]] for p in line])
    lap = np.sum(el / np.maximum(v, 0.1))
    print(f'[raceline] {el.sum():.1f} m, max curvature {np.abs(kappa).max():.2f} 1/m, '
          f'speed {v.min():.1f}-{v.max():.1f} m/s, est lap {lap:.1f} s')
    print(f'[raceline] wall clearance from car centre: min {clear.min():.2f} m '
          f'(target {CAR_WIDTH / 2 + SAFETY_MARGIN:.2f})')
    if clear.min() < CAR_WIDTH / 2 + 0.05:
        print('[raceline] WARNING: line comes within 5 cm of a wall - check the preview')
    return line, v, lap


def preview(img, line, v, res, ox, oy, lap, path):
    H = img.shape[0]
    vis = cv2.cvtColor(np.where(img >= FREE_THRESH, 255, np.where(img < 100, 0, 200)).astype(np.uint8),
                       cv2.COLOR_GRAY2BGR)
    px = np.column_stack([(line[:, 0] - ox) / res - 0.5, H - 1 - ((line[:, 1] - oy) / res - 0.5)]).astype(int)
    cm = cv2.applyColorMap(((v - v.min()) / (np.ptp(v) + 1e-9) * 255).astype(np.uint8).reshape(-1, 1),
                           cv2.COLORMAP_TURBO)
    for i in range(len(px)):
        cv2.line(vis, tuple(px[i]), tuple(px[(i + 1) % len(px)]), tuple(int(c) for c in cm[i, 0]), 3)
    cv2.circle(vis, tuple(px[0]), 8, (0, 160, 0), -1)
    x0, y0 = np.maximum(px.min(0) - 40, 0)
    x1, y1 = px.max(0) + 40
    crop = cv2.resize(vis[y0:y1, x0:x1], None, fx=1.5, fy=1.5, interpolation=cv2.INTER_NEAREST)
    cv2.putText(crop, f'blue {v.min():.1f} m/s -> red {v.max():.1f} m/s | est lap {lap:.1f} s | green = start',
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.imwrite(path, crop)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('map_yaml')
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    name = os.path.splitext(os.path.basename(a.map_yaml))[0]
    out = os.path.expanduser(a.out or os.path.dirname(os.path.abspath(a.map_yaml)))
    os.makedirs(out, exist_ok=True)
    P = lambda s: os.path.join(out, f'{name}_{s}')

    img, res, ox, oy = load_map(a.map_yaml)
    center, ring = extract_track(img, res, ox, oy)
    np.savetxt(P('centerline.csv'), center, delimiter=',', fmt='%.4f',
               header='x_m,y_m,w_tr_right_m,w_tr_left_m', comments='# ')
    Image.fromarray(np.where(ring, 254, 0).astype(np.uint8)).save(P('track.pgm'))
    with open(P('track.yaml'), 'w') as f:
        f.write(f'image: {name}_track.pgm\nmode: trinary\nresolution: {res}\norigin: [{ox}, {oy}, 0]\n'
                f'negate: 0\noccupied_thresh: 0.65\nfree_thresh: 0.25\n')

    line, v, lap = make_raceline(center, ring, res, ox, oy)
    np.savetxt(P('raceline.csv'), np.column_stack([line, v]), delimiter=',', fmt='%.4f')
    np.savetxt(P('raceline_slow.csv'), np.column_stack([line, v * SLOW_FACTOR]), delimiter=',', fmt='%.4f')
    preview(img, line, v, res, ox, oy, lap, P('preview.png'))
    print(f'[done] start point x={line[0, 0]:.2f} y={line[0, 1]:.2f}; files in {out}')


if __name__ == '__main__':
    main()
