# -*- coding: utf-8 -*-
"""泳池清洁机器人覆盖率分析核心

纯像素口径：池面面积按画面中用户标定的水域区域（矩形/椭圆）统计，
机器人每帧检测框大小即其"扫过"的像素范围，覆盖率 = 扫过并集 / 池面像素。
离线运行，不依赖网络与外部模型。
"""
import os
import sys
import json
import math
import time
import subprocess
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2

VERSION = "0.1.0"
SHAPE_RECT = "rect"
SHAPE_ELLIPSE = "ellipse"

SWATH_FACTOR = 0.85      # 有效清洗宽度 = 检测框等效直径 × 该系数
GAP_FRAMES_MAX = 12      # 相邻检测点允许连线的最大间隔帧
MAX_GRID = 1000          # 覆盖栅格最长边
MIN_DET_RATIO = 0.0004   # 候选目标面积下限（占池面积）
MAX_DET_RATIO = 0.05     # 候选目标面积上限（占池面积）


class CancelledError(Exception):
    """用户取消分析"""


@dataclass
class Calibration:
    shape: str
    points: list


@dataclass
class AnalyzeResult:
    video: str
    out_dir: str
    frames: int
    fps: float
    width: int
    height: int
    detected: int
    coverage: float
    uncovered: float
    avg_passes: float
    pool_area_px: int
    path_len_px: float
    overlay_path: str
    heatmap_path: str
    trajectory_csv: str
    summary_json: str
    curve: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


# ---------------------------------------------------------------- 基础工具
def _base_dir():
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def _find_ffmpeg():
    name = "ffmpeg.exe" if os.name == "nt" else "ffmpeg"
    cands = []
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands += [os.path.join(meipass, name), os.path.join(meipass, "_internal", name)]
    cands += [
        os.path.join(_base_dir(), name),
        os.path.join(_base_dir(), "_internal", name),
        os.path.join(os.getcwd(), name),
    ]
    for c in cands:
        if os.path.isfile(c):
            return c
    from shutil import which
    return which(name)


def _progress(progress, frac, msg):
    if progress is None:
        return
    try:
        progress(max(0.0, min(1.0, float(frac))), str(msg))
    except TypeError:
        progress(max(0.0, min(1.0, float(frac))))


def _check_cancel(cancel):
    if cancel is not None:
        try:
            if cancel():
                raise CancelledError("用户取消分析")
        except TypeError:
            if cancel():
                raise CancelledError("用户取消分析")


def _odd(v, lo=3):
    v = max(lo, int(round(v)))
    return v if v % 2 == 1 else v + 1


# ---------------------------------------------------------------- 视频读取
def video_info(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件：%s" % path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if width <= 0 or height <= 0:
        raise RuntimeError("无法读取视频分辨率，文件可能损坏")
    return {"fps": float(fps), "frames": frames, "width": width, "height": height}


def probe_frame(path, frame_index=0):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件：%s" % path)
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, int(frame_index)))
        ok, frame = cap.read()
        if not ok or frame is None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = cap.read()
        if not ok or frame is None:
            raise RuntimeError("无法读取视频帧")
        return frame
    finally:
        cap.release()


# ---------------------------------------------------------------- 标定
def _order_rect_points(pts):
    pts = [tuple(map(float, p)) for p in pts]
    if len(pts) != 4:
        raise ValueError("矩形需要 4 个点")
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    ordered = sorted(pts, key=lambda p: math.atan2(p[1] - cy, p[0] - cx))
    # 把最靠左上的点作为起点（保持顺时针/逆时针顺序不变）
    i = min(range(4), key=lambda j: ordered[j][0] + ordered[j][1])
    return ordered[i:] + ordered[:i]


def _ellipse_params(calib):
    """由 中心 + 长轴端点 + 短轴端点 计算 cv2.ellipse 参数"""
    pts = calib.points
    if len(pts) < 3:
        raise ValueError("圆形池需要 中心、长轴端点、短轴端点 共 3 个点")
    c = np.array(pts[0], dtype=float)
    v1 = np.array(pts[1], dtype=float) - c
    v2 = np.array(pts[2], dtype=float) - c
    a = float(np.linalg.norm(v1))
    b = float(np.linalg.norm(v2))
    if a < 5 or b < 5:
        raise ValueError("圆形池标定点距离过近，请重新点选")
    angle = float(math.degrees(math.atan2(v1[1], v1[0])))
    return (float(c[0]), float(c[1])), (2.0 * a, 2.0 * b), angle


def build_pool_mask(shape, size, calib):
    """返回 uint8 池面掩膜（池内 255）"""
    w, h = int(size[0]), int(size[1])
    mask = np.zeros((h, w), np.uint8)
    if calib.shape == SHAPE_RECT:
        pts = np.array(_order_rect_points(calib.points), np.int32)
        cv2.fillPoly(mask, [pts], 255)
    elif calib.shape == SHAPE_ELLIPSE:
        center, axes, angle = _ellipse_params(calib)
        cv2.ellipse(mask, (int(round(center[0])), int(round(center[1]))),
                    (max(3, int(round(axes[0] / 2))), max(3, int(round(axes[1] / 2)))),
                    angle, 0, 360, 255, -1)
    else:
        raise ValueError("未知池型：%s" % calib.shape)
    return mask


def _water_mask(frame):
    """尝试自动识别水面区域（用于自动标定）"""
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hh, ss, vv = cv2.split(hsv)
    b = frame[:, :, 0].astype(np.int16)
    r = frame[:, :, 2].astype(np.int16)
    masks = [
        ((b - r > 45) & (ss > 105) & (vv > 115)),
        ((hh > 80) & (hh < 120) & (ss > 90) & (vv > 90)),
        ((b - r > 30) & (ss > 80) & (vv > 100)),
    ]
    fh, fw = frame.shape[:2]
    for m in masks:
        m = (m.astype(np.uint8) * 255)
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            continue
        c = max(cnts, key=cv2.contourArea)
        area = cv2.contourArea(c)
        if area < 0.12 * fh * fw:
            continue
        hull = cv2.convexHull(c)
        return hull, area, fh * fw
    return None, 0, fh * fw


def auto_calibrate(path, shape=SHAPE_RECT):
    """自动识别池面轮廓；失败抛 RuntimeError"""
    info = video_info(path)
    frame = probe_frame(path, 0)
    hull, area, total = _water_mask(frame)
    if hull is None or len(hull) < 3:
        raise RuntimeError("自动识别失败，请手动点选池边")
    if shape == SHAPE_ELLIPSE:
        ell = cv2.fitEllipse(hull)
        (cx, cy), (MA, ma), ang = ell
        ca, sa = math.cos(math.radians(ang)), math.sin(math.radians(ang))
        p1 = (cx + MA / 2.0 * ca, cy + MA / 2.0 * sa)
        p2 = (cx - ma / 2.0 * sa, cy + ma / 2.0 * ca)
        return Calibration(SHAPE_ELLIPSE, [[float(cx), float(cy)],
                                           [float(p1[0]), float(p1[1])],
                                           [float(p2[0]), float(p2[1])]])
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.02, 0.03, 0.04, 0.05, 0.06):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = ap.reshape(-1, 2)
            break
    if quad is None:
        rect = cv2.minAreaRect(hull)
        quad = cv2.boxPoints(rect)
    pts = _order_rect_points(quad)
    # 轻微内缩，避开池沿白边
    cx = sum(p[0] for p in pts) / 4.0
    cy = sum(p[1] for p in pts) / 4.0
    pts = [[float(cx + (p[0] - cx) * 0.99), float(cy + (p[1] - cy) * 0.99)] for p in pts]
    return Calibration(SHAPE_RECT, pts)


# ---------------------------------------------------------------- 检测
def _detect_candidates(gray, bg_gray, roi, mode, thr, pool_area, kern):
    diff = cv2.absdiff(gray, bg_gray)
    base = ((diff > thr) & (roi > 0)).astype(np.uint8) * 255
    cond = (gray < 115) if mode != "bright" else (gray > 190)
    conds = [cond]
    amin = max(80, int(pool_area * MIN_DET_RATIO))
    amax = max(amin * 4, int(pool_area * MAX_DET_RATIO))
    out = []
    for cond in conds:
        m = (base > 0).astype(np.uint8) & cond.astype(np.uint8)
        m = (m * 255).astype(np.uint8)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kern["open"])
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kern["close"])
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            a = cv2.contourArea(c)
            if a < amin or a > amax:
                continue
            x, y, w, h = cv2.boundingRect(c)
            if w < 4 or h < 4:
                continue
            ar = w / float(max(h, 1))
            if not (0.3 < ar < 3.0):
                continue
            fill = a / float(w * h)
            if fill < 0.25:
                continue
            out.append({"cx": x + w / 2.0, "cy": y + h / 2.0, "w": float(w),
                        "h": float(h), "area": float(a), "fill": float(fill)})
    out.sort(key=lambda d: -d["area"])
    return out


def _pick_candidate(cands, prev, gate, size_ref, pred=None):
    """从候选里挑机器人：先按尺寸一致性过滤，再按预测位置门限匹配。
    有上一帧但找不到合理候选时返回 None（视为该帧漏检），避免跳到反光/漂浮物上。"""
    if not cands:
        return None
    if size_ref and len(size_ref) >= 5:
        med = float(np.median(size_ref))
        cands = [c for c in cands if 0.3 * med <= c["area"] <= 3.0 * med]
        if not cands:
            return None
    if prev is not None:
        tgt = pred if pred is not None else prev
        near = [c for c in cands
                if math.hypot(c["cx"] - tgt["cx"], c["cy"] - tgt["cy"]) <= gate]
        if not near:
            return None
        return min(near, key=lambda c: math.hypot(c["cx"] - tgt["cx"], c["cy"] - tgt["cy"]))
    return cands[0]


# ---------------------------------------------------------------- 覆盖率
def _resample_polyline(points, step):
    if len(points) < 2:
        return list(points)
    step = max(1e-6, step)
    out = [points[0]]
    prev = points[0]
    acc = 0.0
    for p in points[1:]:
        d = math.hypot(p[0] - prev[0], p[1] - prev[1])
        if d <= 0:
            continue
        t = 0.0
        while True:
            need = step - acc
            if t + need <= d:
                t += need
                out.append((prev[0] + (p[0] - prev[0]) * t / d,
                            prev[1] + (p[1] - prev[1]) * t / d))
                acc = 0.0
            else:
                acc += d - t
                break
        prev = p
    return out


def _runs_from_detections(dets, gap_max):
    runs, cur, last_k = [], [], None
    for k in range(len(dets)):
        if dets[k] is None:
            if len(cur) > 1:
                runs.append(cur)
            cur = []
            continue
        if cur and (k - last_k) > gap_max:
            runs.append(cur)
            cur = []
        cur.append((k, dets[k]["cx"], dets[k]["cy"]))
        last_k = k
    if len(cur) > 1:
        runs.append(cur)
    return runs


def _heat_colors(pass_cnt):
    heat = np.zeros(pass_cnt.shape + (3,), np.uint8)
    lut = [(0, 0, 180), (180, 80, 0), (0, 170, 255), (0, 220, 120), (245, 245, 245)]
    heat[pass_cnt == 0] = lut[0]
    heat[pass_cnt == 1] = lut[1]
    heat[pass_cnt == 2] = lut[2]
    heat[pass_cnt == 3] = lut[3]
    heat[pass_cnt >= 4] = lut[4]
    return heat


def _save_heatmap(path, pass_cnt, mask_small, calib, pts, crop, ds, out_w=1000):
    heat = _heat_colors(pass_cnt)
    heat[~mask_small.astype(bool)] = (25, 25, 25)
    gh, gw = heat.shape[:2]
    out_w = max(360, min(int(out_w), int(gw * 2)))
    r = out_w / float(gw)
    out_h = max(60, int(round(gh * r)))
    heat = cv2.resize(heat, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    def mp(x, y):
        return (int(round((x - crop[0]) / ds * r)), int(round((y - crop[1]) / ds * r)))

    if calib.shape == SHAPE_RECT:
        cv2.polylines(heat, [np.array([mp(*p) for p in _order_rect_points(calib.points)], np.int32)],
                      True, (255, 255, 255), 2)
    else:
        center, axes, angle = _ellipse_params(calib)
        c = mp(*center)
        cv2.ellipse(heat, c, (max(2, int(axes[0] / 2 / ds * r)),
                              max(2, int(axes[1] / 2 / ds * r))),
                    angle, 0, 360, (255, 255, 255), 2)
    last_k = None
    prev_pt = None
    for (k, x, y) in pts:
        pt = mp(x, y)
        if prev_pt is not None and last_k is not None and k - last_k <= GAP_FRAMES_MAX:
            cv2.line(heat, prev_pt, pt, (255, 255, 255), 1)
        prev_pt = pt
        last_k = k
    lut = [(0, 0, 180), (180, 80, 0), (0, 170, 255), (0, 220, 120), (245, 245, 245)]
    legend_h = 62
    legend = np.zeros((legend_h, out_w, 3), np.uint8)
    labels = ["uncovered", "1x", "2x", "3x", "4x+"]
    for i, lab in enumerate(labels):
        x0 = 14 + i * max(1, (out_w - 20) // 5)
        col = tuple(int(v) for v in lut[i])
        cv2.rectangle(legend, (x0, 16), (x0 + 26, 44), col, -1)
        cv2.putText(legend, lab, (x0 + 32, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (240, 240, 240), 1, cv2.LINE_AA)
    canvas = np.vstack([heat, legend])
    cv2.imwrite(path, canvas)


# ---------------------------------------------------------------- 叠加视频
def _open_writer(path, size, fps, ffmpeg):
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % size, "-r", "%.3f" % fps, "-i", "-",
               "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
               "-pix_fmt", "yuv420p", "-movflags", "+faststart", path]
        try:
            pp = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                  stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            return ("ffmpeg", pp), True
        except Exception:
            pass
    for ext, fourcc in ((".avi", "MJPG"), (".mp4", "mp4v")):
        p = os.path.splitext(path)[0] + ext
        vw = cv2.VideoWriter(p, cv2.VideoWriter_fourcc(*fourcc), fps, size)
        if vw.isOpened():
            return ("cv2", vw, p), True
    return None, False


def _write_frame(writer, frame):
    kind = writer[0]
    if kind == "ffmpeg":
        writer[1].stdin.write(frame.tobytes())
    elif kind == "cv2":
        writer[1].write(frame)


def _close_writer(writer):
    kind = writer[0]
    if kind == "ffmpeg":
        try:
            writer[1].stdin.close()
        except Exception:
            pass
        try:
            writer[1].wait(timeout=120)
        except Exception:
            pass
        return getattr(writer[1], "returncode", 0) == 0
    elif kind == "cv2":
        writer[1].release()
        return True
    return False


# ---------------------------------------------------------------- 主分析
def analyze(video_path, calib, out_dir, *, target="auto", sensitivity=0.5,
            progress=None, cancel=None, overlay_width=1280):
    if target not in ("auto", "dark", "bright"):
        raise ValueError("未知目标类型：%s" % target)
    if not os.path.isfile(video_path):
        raise RuntimeError("视频文件不存在：%s" % video_path)
    if not isinstance(calib, Calibration) or not calib.points:
        raise RuntimeError("尚未完成池面标定")
    os.makedirs(out_dir, exist_ok=True)

    info = video_info(video_path)
    fps, n_frames = info["fps"] or 25.0, info["frames"]
    vw, vh = info["width"], info["height"]
    if n_frames <= 0:
        raise RuntimeError("视频帧数为 0，文件可能损坏")

    pool_mask = build_pool_mask(calib.shape, (vw, vh), calib)
    pool_area = int(np.count_nonzero(pool_mask))
    if pool_area < 0.02 * vw * vh:
        raise RuntimeError("标定的池面区域过小（不到画面 2%），请重新标定")

    # 检测 ROI：池面外扩一点，容忍机器人贴边
    pad = max(8, int(0.01 * math.hypot(vw, vh)))
    roi = cv2.dilate(pool_mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
    roi = cv2.bitwise_and(roi, roi)

    s = min(vw, vh) / 1080.0
    kern = {
        "open": np.ones((_odd(5 * max(s, 0.5)), _odd(5 * max(s, 0.5))), np.uint8),
        "close": np.ones((_odd(25 * max(s, 0.5)), _odd(25 * max(s, 0.5))), np.uint8),
    }
    # 灵敏度 -> 差分阈值（0 保守 28，1 灵敏 2）
    sens = min(1.0, max(0.0, float(sensitivity)))
    thr = int(round(28 - 26 * sens))

    _progress(progress, 0.0, "读取视频背景…")
    # 背景：均匀抽帧取中值
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")
    n_bg = int(min(61, max(9, n_frames)))
    idxs = np.linspace(0, max(0, n_frames - 1), n_bg).astype(int)
    frames_bg = []
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, fr = cap.read()
        if ok and fr is not None:
            frames_bg.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
    if not frames_bg:
        cap.release()
        raise RuntimeError("无法读取视频帧")
    bg_gray = np.median(np.stack(frames_bg), axis=0).astype(np.uint8)

    # 第一遍：逐帧检测 + 跟踪
    warnings = []
    gate = 0.15 * math.hypot(vw, vh)

    def run_track_pass(mode, p0=0.05, p1=0.60):
        cap2 = cv2.VideoCapture(video_path)
        if not cap2.isOpened():
            raise RuntimeError("无法打开视频文件")
        dets_local = [None] * n_frames
        prev, prev2, size_ref, misses = None, None, [], 0
        cnt = 0
        for k in range(n_frames):
            ok, frame = cap2.read()
            if not ok:
                break
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            cands = _detect_candidates(gray, bg_gray, roi, mode, thr, pool_area, kern)
            pred = None
            if prev is not None:
                if prev2 is not None:
                    pred = {"cx": 2 * prev["cx"] - prev2["cx"],
                            "cy": 2 * prev["cy"] - prev2["cy"]}
                else:
                    pred = prev
            chosen = _pick_candidate(cands, prev, gate, size_ref, pred)
            if chosen is not None:
                dets_local[k] = chosen
                prev2, prev = prev, chosen
                size_ref.append(chosen["area"])
                if len(size_ref) > 60:
                    size_ref.pop(0)
                cnt += 1
                misses = 0
            else:
                misses += 1
                if misses >= 30:          # 连续漏检：允许重新捕获
                    prev, prev2 = None, None
            if k % 10 == 0:
                _check_cancel(cancel)
                _progress(progress, p0 + (p1 - p0) * k / max(1, n_frames),
                          "检测机器人… 第 %d/%d 帧" % (k, n_frames))
        cap2.release()
        return dets_local, cnt

    if target == "auto":
        _progress(progress, 0.05, "检测机器人（暗色目标）…")
        dets, detected = run_track_pass("dark", 0.05, 0.55)
        if detected < 0.25 * n_frames:
            _progress(progress, 0.05, "暗色目标较少，尝试亮色目标…")
            dets2, detected2 = run_track_pass("bright", 0.10, 0.60)
            if detected2 > detected:
                dets, detected = dets2, detected2
                warnings.append("已自动切换为亮色目标检测。")
    else:
        dets, detected = run_track_pass(target, 0.05, 0.60)
    _check_cancel(cancel)
    if detected < 5:
        raise RuntimeError("未检测到机器人。可调高灵敏度、切换目标类型，或检查池面标定是否准确。")

    # 第二遍：像素覆盖率（栅格近似）
    _progress(progress, 0.62, "计算覆盖面积…")
    ys, xs = np.where(pool_mask > 0)
    x0, x1 = int(xs.min()), int(xs.max()) + 1
    y0, y1 = int(ys.min()), int(ys.max()) + 1
    ds = max(1, int(round(max(x1 - x0, y1 - y0) / float(MAX_GRID))))
    gw = max(8, int(round((x1 - x0) / ds)))
    gh = max(8, int(round((y1 - y0) / ds)))
    mask_small = cv2.resize(pool_mask[y0:y1, x0:x1], (gw, gh),
                            interpolation=cv2.INTER_NEAREST) > 127
    pool_area_small = float(mask_small.sum())

    cov = np.zeros((gh, gw), np.uint8)
    pass_cnt = np.zeros((gh, gw), np.uint16)
    pts_plane = []
    curve = []
    prev_idx = None
    for k in range(n_frames):
        d = dets[k]
        if d is not None:
            r = math.sqrt(d["w"] * d["h"]) / 2.0 * SWATH_FACTOR
            r_px = max(1, int(round(r / ds)))
            cx = (d["cx"] - x0) / ds
            cy = (d["cy"] - y0) / ds
            tmp = np.zeros((gh, gw), np.uint8)
            if prev_idx is not None and (k - prev_idx) <= GAP_FRAMES_MAX:
                pd = dets[prev_idx]
                ptx = int(round((pd["cx"] - x0) / ds))
                pty = int(round((pd["cy"] - y0) / ds))
                cv2.line(tmp, (ptx, pty), (int(round(cx)), int(round(cy))), 1,
                         thickness=max(1, 2 * r_px))
            else:
                cv2.circle(tmp, (int(round(cx)), int(round(cy))), r_px, 1, -1)
            tmp = (tmp > 0).astype(np.uint8) * mask_small.astype(np.uint8)
            cov |= tmp
            pass_cnt += tmp.astype(np.uint16)
            pts_plane.append((k, float(d["cx"]), float(d["cy"])))
            prev_idx = k
        curve.append((k / fps, float(cov.sum()) / max(1e-6, pool_area_small)))
        if k % 60 == 0:
            _check_cancel(cancel)
    _check_cancel(cancel)
    coverage = float(cov.sum()) / max(1e-6, pool_area_small)

    # 平均经过次数：轨迹简化后逐段栅格化
    runs = _runs_from_detections(dets, GAP_FRAMES_MAX)
    pc2 = np.zeros((gh, gw), np.uint16)
    for run in runs:
        pts_small = [(int(round((r[1] - x0) / ds)), int(round((r[2] - y0) / ds))) for r in run]
        arr = np.array(pts_small, np.int32).reshape(-1, 1, 2)
        if len(arr) < 2:
            continue
        simple = cv2.approxPolyDP(arr, max(1.0, 4.0), False).reshape(-1, 2)
        for a, b in zip(simple[:-1], simple[1:]):
            tmp = np.zeros((gh, gw), np.uint8)
            r_px = max(1, int(round(math.sqrt(dets[run[0][0]]["w"] * dets[run[0][0]]["h"])
                                    / 2.0 * SWATH_FACTOR / ds)))
            cv2.line(tmp, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1,
                     thickness=max(1, 2 * r_px))
            tmp = (tmp > 0).astype(np.uint8) * mask_small.astype(np.uint8)
            pc2 += tmp.astype(np.uint16)
    pcm = pc2 > 0
    avg_passes = float(pc2[pcm].mean()) if pcm.any() else 0.0

    path_len_px = 0.0
    for run in runs:
        for a, b in zip(run[:-1], run[1:]):
            path_len_px += math.hypot(b[1] - a[1], b[2] - a[2])

    # 热力图
    _progress(progress, 0.68, "生成热力图…")
    heatmap_path = os.path.join(out_dir, "heatmap.png")
    try:
        _save_heatmap(heatmap_path, pc2, mask_small, calib, pts_plane, (x0, y0), ds)
    except Exception:
        heatmap_path = ""

    # 轨迹 CSV
    traj_csv = os.path.join(out_dir, "trajectory.csv")
    with open(traj_csv, "w", encoding="utf-8") as f:
        f.write("frame,time_s,cx_px,cy_px,w_px,h_px\n")
        for k, d in enumerate(dets):
            if d is None:
                f.write("%d,%.3f,,,,\n" % (k, k / fps))
            else:
                f.write("%d,%.3f,%.1f,%.1f,%.1f,%.1f\n"
                        % (k, k / fps, d["cx"], d["cy"], d["w"], d["h"]))

    # 叠加视频
    _progress(progress, 0.72, "渲染轨迹视频…")
    ffmpeg = _find_ffmpeg()
    out_w = min(int(overlay_width), vw)
    out_w -= out_w % 2
    out_h = int(round(vh * out_w / float(vw)))
    out_h -= out_h % 2
    ov_path = os.path.join(out_dir, "overlay.mp4")
    writer, ok_writer = _open_writer(ov_path, (out_w, out_h), fps, ffmpeg)
    if not ok_writer:
        raise RuntimeError("无法创建输出视频（缺少视频编码器）")
    if ffmpeg is None:
        warnings.append("未找到 ffmpeg，轨迹视频已用 MJPG/AVI 编码，兼容性略低。")
    if calib.shape == SHAPE_RECT:
        rect_pts = np.array(_order_rect_points(calib.points), np.int32)
    else:
        center, axes, angle = _ellipse_params(calib)
    cap = cv2.VideoCapture(video_path)
    trail = []
    last_trail_k = None
    for k in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        vis = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        sx = out_w / float(vw)
        if calib.shape == SHAPE_RECT:
            cv2.polylines(vis, [(rect_pts * sx).astype(np.int32)], True, (0, 255, 255), 2)
        else:
            cv2.ellipse(vis, (int(center[0] * sx), int(center[1] * sx)),
                        (max(2, int(axes[0] / 2 * sx)), max(2, int(axes[1] / 2 * sx))),
                        angle, 0, 360, (0, 255, 255), 2)
        d = dets[k]
        if d is not None:
            x1p = int((d["cx"] - d["w"] / 2) * sx)
            y1p = int((d["cy"] - d["h"] / 2) * sx)
            x2p = int((d["cx"] + d["w"] / 2) * sx)
            y2p = int((d["cy"] + d["h"] / 2) * sx)
            cv2.rectangle(vis, (x1p, y1p), (x2p, y2p), (0, 255, 0), 2)
            pt = (int(d["cx"] * sx), int(d["cy"] * sx))
            if trail and last_trail_k is not None and k - last_trail_k <= GAP_FRAMES_MAX:
                cv2.line(vis, trail[-1], pt, (0, 220, 255), 2)
            trail.append(pt)
            last_trail_k = k
        cov_now = curve[k][1]
        cv2.putText(vis, "coverage %.1f%%" % (cov_now * 100),
                    (max(8, out_w - 260), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "t=%.1fs" % (k / fps), (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
        _write_frame(writer, vis)
        if k % 30 == 0:
            _check_cancel(cancel)
            _progress(progress, 0.72 + 0.26 * k / max(1, n_frames),
                      "渲染轨迹视频… 第 %d/%d 帧" % (k, n_frames))
    cap.release()
    ok_close = _close_writer(writer)
    if not ok_close:
        warnings.append("视频编码可能未完整结束，若播放异常可改用无轨迹视频。")
    if not os.path.isfile(ov_path):
        base = os.path.splitext(ov_path)[0]
        for ext in (".avi", ".mp4"):
            if os.path.isfile(base + ext):
                ov_path = base + ext
                break

    _progress(progress, 0.99, "保存结果…")
    summary_path = os.path.join(out_dir, "summary.json")
    result = AnalyzeResult(
        video=video_path, out_dir=os.path.abspath(out_dir), frames=int(n_frames),
        fps=float(fps), width=int(vw), height=int(vh), detected=int(detected),
        coverage=float(coverage), uncovered=float(1.0 - coverage),
        avg_passes=float(avg_passes), pool_area_px=int(pool_area),
        path_len_px=float(path_len_px), overlay_path=os.path.abspath(ov_path),
        heatmap_path=os.path.abspath(heatmap_path) if heatmap_path else "",
        trajectory_csv=os.path.abspath(traj_csv), summary_json=os.path.abspath(summary_path),
        curve=[[float(t), float(c)] for t, c in curve],
        warnings=list(warnings),
    )
    payload = asdict(result)
    payload.update({
        "version": VERSION,
        "shape": calib.shape,
        "calib_points": [[float(x), float(y)] for x, y in calib.points],
        "params": {"target": target, "sensitivity": sens, "swath_factor": SWATH_FACTOR,
                   "gap_frames": GAP_FRAMES_MAX, "pixel_basis": True},
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    })
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    _progress(progress, 1.0, "完成")
    return result


# ---------------------------------------------------------------- 历史
def list_history(root_dir):
    out = []
    if not root_dir or not os.path.isdir(root_dir):
        return out
    for name in os.listdir(root_dir):
        d = os.path.join(root_dir, name)
        sp = os.path.join(d, "summary.json")
        if not os.path.isfile(sp):
            continue
        try:
            with open(sp, "r", encoding="utf-8") as f:
                s = json.load(f)
            out.append({
                "out_dir": os.path.abspath(d),
                "video": s.get("video", ""),
                "coverage": float(s.get("coverage", 0.0)),
                "detected": int(s.get("detected", 0)),
                "frames": int(s.get("frames", 0)),
                "shape": s.get("shape", ""),
                "mtime": os.path.getmtime(sp),
                "created": s.get("created", ""),
            })
        except Exception:
            continue
    out.sort(key=lambda x: -x["mtime"])
    return out
