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
import re
import time
import subprocess
from dataclasses import dataclass, field, asdict

import numpy as np
import cv2

try:
    from PIL import Image, ImageDraw, ImageFont
except Exception:  # pragma: no cover
    Image = ImageDraw = ImageFont = None

VERSION = "0.7.3"
SHAPE_RECT = "rect"
SHAPE_ELLIPSE = "ellipse"
SHAPE_POLY = "poly"

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
    coverage_path: str = ""
    curve_path: str = ""
    trajectory_image_path: str = ""
    shape: str = ""


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


# ---------------------------------------------------------------- 通用绘图工具
_CJK_FONT_CACHE = []


def _find_cjk_font():
    """找一个可用的中文字体（Windows 用微软雅黑；Linux/安卓用 Noto/文泉驿）"""
    if _CJK_FONT_CACHE:
        return _CJK_FONT_CACHE[0]
    cands = []
    if os.name == "nt":
        win = os.environ.get("WINDIR", "C:/Windows")
        cands += [os.path.join(win, "Fonts", n) for n in
                  ("msyh.ttc", "msyhbd.ttc", "simhei.ttf", "simsun.ttc", "Deng.ttf")]
    cands += [
        "/system/fonts/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
        "/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf",
    ]
    meipass = getattr(sys, "_MEIPASS", None)
    if meipass:
        cands += [os.path.join(meipass, "msyh.ttc"),
                  os.path.join(meipass, "NotoSansCJK-Regular.ttc")]
    cands.append(os.path.join(_base_dir(), "msyh.ttc"))
    for c in cands:
        if os.path.isfile(c):
            _CJK_FONT_CACHE.append(c)
            return c
    _CJK_FONT_CACHE.append(None)
    return None


def _font(size):
    path = _find_cjk_font()
    if not path or ImageFont is None:
        return None
    try:
        return ImageFont.truetype(path, int(max(9, size)))
    except Exception:
        return None


def _text(img_bgr, xy, text, size=22, color=(255, 255, 255)):
    """在 BGR ndarray 上写文字（有中文字体用 Pillow，否则退回 OpenCV 英文）"""
    f = _font(size)
    if f is None or Image is None:
        cv2.putText(img_bgr, str(text), (int(xy[0]), int(xy[1])),
                    cv2.FONT_HERSHEY_SIMPLEX, size / 28.0, color, 1, cv2.LINE_AA)
        return img_bgr
    pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
    ImageDraw.Draw(pil).text((int(xy[0]), int(xy[1])), str(text), font=f,
                             fill=(int(color[2]), int(color[1]), int(color[0])))
    return cv2.cvtColor(np.array(pil), cv2.COLOR_RGB2BGR)


def _imwrite(path, img):
    """写图片：cv2.imwrite 在 Windows 上不支持中文路径，这里用 imencode + 文件流，
    中文/特殊字符路径也能写出来。返回是否成功。"""
    ext = os.path.splitext(str(path))[1].lower() or ".png"
    if ext not in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"):
        ext = ".png"
    try:
        ok, buf = cv2.imencode(ext, img)
        if ok:
            with open(path, "wb") as f:
                f.write(buf.tobytes())
            return os.path.isfile(path) and os.path.getsize(path) > 0
    except Exception:
        pass
    try:
        return bool(cv2.imwrite(path, img))
    except Exception:
        return False


def _parse_hms(text):
    """解析 HH:MM 或 HH:MM:SS；失败返回 None"""
    if text is None:
        return None
    s = str(text).strip().replace("：", ":")
    if not s:
        return None
    m = re.match(r"^(\d{1,2}):(\d{1,2})(?::(\d{1,2}))?$", s)
    if not m:
        return None
    h, mi, se = int(m.group(1)), int(m.group(2)), int(m.group(3) or 0)
    if h > 23 or mi > 59 or se > 59:
        return None
    return h * 3600 + mi * 60 + se


def _fmt_clock(sec):
    sec = int(round(sec)) % 86400
    return "%02d:%02d:%02d" % (sec // 3600, (sec // 60) % 60, sec % 60)


def _fmt_min(sec):
    sec = int(round(sec))
    return "+%d分%02d秒" % (sec // 60, sec % 60)


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


def _poly_points(points):
    """把任意点列表整理成 (N,2) float32 闭合轮廓点（去掉重复点）"""
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 2)
    if len(pts) < 3:
        raise ValueError("轮廓至少需要 3 个点")
    keep = [pts[0]]
    for p in pts[1:]:
        if np.linalg.norm(p - keep[-1]) > 1e-6:
            keep.append(p)
    pts = np.array(keep, np.float32)
    if len(pts) < 3:
        raise ValueError("轮廓点过于集中")
    return pts


def _resample_closed(points, step):
    """把闭合轮廓按弧长等距重采样"""
    pts = _poly_points(points)
    closed = np.vstack([pts, pts[0]])
    seg = np.linalg.norm(np.diff(closed, axis=0), axis=1)
    perim = float(seg.sum())
    if perim <= 1e-3:
        raise ValueError("轮廓周长过小")
    step = max(1e-3, float(step))
    n = int(max(8, min(800, round(perim / step))))
    targets = np.linspace(0.0, perim, n, endpoint=False)
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    out = []
    j = 0
    for t in targets:
        while j < len(seg) and cum[j + 1] < t:
            j += 1
        j = min(j, len(seg) - 1)
        r = (t - cum[j]) / max(1e-9, seg[j])
        out.append(closed[j] + (closed[j + 1] - closed[j]) * r)
    return np.array(out, np.float32)


def _chaikin_closed(points, iters=1):
    pts = np.asarray(points, np.float32)
    for _ in range(max(0, int(iters))):
        out = []
        n = len(pts)
        for i in range(n):
            p, q = pts[i], pts[(i + 1) % n]
            out.append(0.75 * p + 0.25 * q)
            out.append(0.25 * p + 0.75 * q)
        pts = np.array(out, np.float32)
    return pts


def smooth_contour(points, strength=1.0):
    """把用户手绘的折线轮廓变平滑（返回稠密闭合轮廓点列表）

    strength: 0=只重采样不圆滑, 1=标准, 2=更强；越大越圆润但会略微收缩。
    """
    try:
        strength = max(0.0, min(3.0, float(strength)))
    except (TypeError, ValueError):
        strength = 1.0
    pts = _poly_points(points)
    if strength <= 0.05:
        return _resample_closed(pts, 3.0).tolist()
    # 先粗采样 -> Chaikin 平滑 -> 再细采样
    coarse = _resample_closed(pts, 12.0)
    iters = max(1, int(round(strength * 2)))
    sm = _chaikin_closed(coarse, iters)
    sm = _resample_closed(sm, 4.0)
    return [[float(x), float(y)] for x, y in sm]


def snap_contour(frame, points, band_px=None):
    """可选：把轮廓点沿法线方向吸附到画面里梯度最强的位置（池壁/池底交界）。

    默认只在很小的范围内搜索并强阻尼，不会大幅改变轮廓；失败时返回原轮廓。
    """
    try:
        pts = _resample_closed(points, 4.0)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        gray = cv2.GaussianBlur(gray, (5, 5), 0).astype(np.float32)
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)
        h, w = grad.shape
        if band_px is None:
            band_px = max(4.0, 0.012 * min(h, w))
        band_px = float(max(2.0, min(band_px, 0.05 * min(h, w))))
        offsets = np.arange(-band_px, band_px + 0.5, 1.0)
        cur = pts.copy()
        for _ in range(3):
            nxt = []
            n = len(cur)
            for i in range(n):
                a, b = cur[i - 1], cur[(i + 1) % n]
                t = b - a
                nt = np.linalg.norm(t)
                if nt < 1e-6:
                    nxt.append(cur[i]); continue
                t = t / nt
                nrm = np.array([-t[1], t[0]], np.float32)
                q = cur[i][None, :] + nrm[None, :] * offsets[:, None]
                xs = np.clip(np.round(q[:, 0]).astype(int), 0, w - 1)
                ys = np.clip(np.round(q[:, 1]).astype(int), 0, h - 1)
                vals = grad[ys, xs]
                # 轻微偏向近处的惩罚，避免乱跳
                vals = vals * (1.0 - 0.15 * np.abs(offsets) / band_px)
                k = int(np.argmax(vals))
                target = cur[i] + nrm * offsets[k]
                nxt.append(cur[i] + (target - cur[i]) * 0.5)
            cur = np.array(nxt, np.float32)
            cur = _resample_closed(_chaikin_closed(cur, 1), 4.0)
        return [[float(x), float(y)] for x, y in cur]
    except Exception:
        return [[float(x), float(y)] for x, y in np.asarray(points, np.float32).reshape(-1, 2)]


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
    elif calib.shape == SHAPE_POLY:
        pts = _poly_points(calib.points)
        if len(pts) < 3:
            raise ValueError("手绘轮廓至少需要 3 个点")
        cv2.fillPoly(mask, [pts.astype(np.int32)], 255)
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
    if shape == SHAPE_POLY:
        pts = smooth_contour(hull.reshape(-1, 2).tolist(), 1.0)
        return Calibration(SHAPE_POLY, pts)
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
def _detect_candidates(gray, bg_gray, roi, mode, thr, pool_area, kern,
                       amin_ratio=MIN_DET_RATIO, amax_ratio=MAX_DET_RATIO):
    diff = cv2.absdiff(gray, bg_gray)
    base = ((diff > thr) & (roi > 0)).astype(np.uint8) * 255
    cond = (gray < 115) if mode != "bright" else (gray > 190)
    conds = [cond]
    amin = max(24, int(pool_area * amin_ratio))
    amax = max(amin * 6, int(pool_area * amax_ratio))
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
            dia = math.sqrt(float(w) * float(h))
            out.append({"cx": x + w / 2.0, "cy": y + h / 2.0, "w": float(w),
                        "h": float(h), "area": float(a), "fill": float(fill),
                        "_dia": dia})
    out.sort(key=lambda d: -d["area"])
    return out


def _estimate_robot_area(cap, bg_gray, roi, mode, thr, pool_area, kern, n_frames,
                         samples=25, det_scale=1.0):
    """抽样若干帧，用每帧最大候选的面积中位数估计机器人大小（机器人通常是最大的运动目标）"""
    areas = []
    idxs = np.linspace(0, max(0, n_frames - 1), max(6, int(samples))).astype(int)
    for i in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
        ok, frame = cap.read()
        if not ok:
            continue
        if det_scale != 1.0:
            frame = cv2.resize(frame, (bg_gray.shape[1], bg_gray.shape[0]),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = _detect_candidates(gray, bg_gray, roi, mode, thr, pool_area, kern)
        if cands:
            areas.append(cands[0]["area"])
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    if not areas:
        return None
    return float(np.median(areas))


def _template_track(video_path, seed, n_frames, fps, pool_mask, progress, cancel,
                    sensitivity=0.5, p0=0.05, p1=0.75):
    """手动框选一次机器人，之后用模板匹配逐帧跟踪（适合小目标/浅色/反光干扰大的机位）。
    seed: {"frame": 帧号, "box": [x, y, w, h]}（原视频像素坐标）"""
    frame_idx = int(seed.get("frame", 0))
    bx, by, bw, bh = [int(round(v)) for v in seed.get("box", [])]
    if bw < 6 or bh < 6:
        raise RuntimeError("框选的机器人区域太小，请重新框选")
    info = video_info(video_path)
    vw, vh = info["width"], info["height"]
    bx = max(0, min(bx, vw - bw)); by = max(0, min(by, vh - bh))
    seed_frame = probe_frame(video_path, frame_idx)
    gray0 = cv2.cvtColor(seed_frame, cv2.COLOR_BGR2GRAY)
    template = gray0[by:by + bh, bx:bx + bw].copy()
    if float(template.std()) < 4.0:
        raise RuntimeError("框选区域几乎没有纹理，请把机器人完整框进去")
    try:
        sens = max(0.0, min(1.0, float(sensitivity)))
    except (TypeError, ValueError):
        sens = 0.5
    thr = 0.62 - 0.22 * sens
    margin = int(max(bw, bh) * 1.5) + 8
    dets = [None] * n_frames
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")
    # 搜索范围限制在池面附近，避免匹配到池外的白墙/设备
    pad = int(0.01 * max(vw, vh)) + max(bw, bh)
    roi = cv2.dilate(pool_mask, np.ones((pad), np.uint8))
    prev = (float(bx + bw / 2.0), float(by + bh / 2.0))
    vel = (0.0, 0.0)
    miss = 0
    last_k = None
    for k in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        if k < frame_idx:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        found = None
        px = int(round(prev[0] + vel[0])); py = int(round(prev[1] + vel[1]))
        x0 = max(0, px - margin - bw // 2); x1 = min(vw, px + margin + bw // 2 + 1)
        y0 = max(0, py - margin - bh // 2); y1 = min(vh, py + margin + bh // 2 + 1)
        if x1 - x0 >= bw and y1 - y0 >= bh:
            crop = gray[y0:y1, x0:x1]
            res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
            _, mx, _, mloc = cv2.minMaxLoc(res)
            if mx >= thr:
                cx = x0 + mloc[0] + bw / 2.0
                cy = y0 + mloc[1] + bh / 2.0
                found = (cx, cy, float(mx))
        if found is None and miss >= 8:
            # 跟丢后整池搜索一次（阈值略高，避免乱匹配）
            ys, xs = np.where(roi > 0)
            if len(xs):
                x0, x1 = int(xs.min()), int(xs.max()) + 1
                y0, y1 = int(ys.min()), int(ys.max()) + 1
                crop = gray[y0:y1, x0:x1]
                if crop.shape[0] >= bh and crop.shape[1] >= bw:
                    res = cv2.matchTemplate(crop, template, cv2.TM_CCOEFF_NORMED)
                    _, mx, _, mloc = cv2.minMaxLoc(res)
                    mask_patch = cv2.resize(roi[y0:y1, x0:x1], (res.shape[1], res.shape[0]),
                                            interpolation=cv2.INTER_NEAREST) > 0
                    if mx >= thr + 0.08 and mask_patch[mloc[1], mloc[0]]:
                        found = (x0 + mloc[0] + bw / 2.0, y0 + mloc[1] + bh / 2.0, float(mx))
        if found is not None:
            cx, cy, score = found
            if last_k is not None and k - last_k > 0:
                vel = (0.5 * vel[0] + 0.5 * (cx - prev[0]) / (k - last_k),
                       0.5 * vel[1] + 0.5 * (cy - prev[1]) / (k - last_k))
            dets[k] = {"cx": float(cx), "cy": float(cy), "w": float(bw), "h": float(bh),
                       "area": float(bw * bh), "fill": 1.0,
                       "_dia": math.sqrt(float(bw * bh))}
            prev = (cx, cy); last_k = k; miss = 0
            if score >= 0.80 and k % 4 == 0:   # 高分时缓慢更新模板，适应旋转/水光
                nx = int(round(cx - bw / 2.0)); ny = int(round(cy - bh / 2.0))
                if 0 <= nx and 0 <= ny and nx + bw <= vw and ny + bh <= vh:
                    patch = gray[ny:ny + bh, nx:nx + bw].copy()
                    if float(patch.std()) > 3.0:
                        template = cv2.addWeighted(template, 0.7, patch, 0.3, 0)
        else:
            miss += 1
            if miss >= 40:
                prev = (prev[0] + vel[0] * 40, prev[1] + vel[1] * 40)
                vel = (0.0, 0.0)
        if miss > 90:
            break
        if k % 10 == 0:
            _check_cancel(cancel)
            _progress(progress, p0 + (p1 - p0) * k / max(1, n_frames),
                      "模板跟踪机器人… 第 %d/%d 帧" % (k, n_frames))
    cap.release()
    return dets


def _gather_candidates(video_path, bg_gray, roi, modes, thr, pool_area, kern,
                       n_frames, stride, progress, cancel, p0, p1, pool_mask=None):
    """逐帧收集所有候选目标（不做“锁定”），供后面多轨迹选择用"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("无法打开视频文件")
    out = [None] * n_frames
    for k in range(n_frames):
        if stride > 1 and (k % stride):
            if not cap.grab():
                break
            if k % 30 == 0:
                _progress(progress, p0 + (p1 - p0) * k / max(1, n_frames),
                          "检测机器人… 第 %d/%d 帧" % (k, n_frames))
            continue
        ok, frame = cap.read()
        if not ok:
            break
        if bg_gray.shape[1] != frame.shape[1]:
            frame = cv2.resize(frame, (bg_gray.shape[1], bg_gray.shape[0]),
                               interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        cands = []
        for mode in modes:
            cands.extend(_detect_candidates(gray, bg_gray, roi, mode, thr, pool_area, kern,
                                            amin_ratio=0.00006, amax_ratio=0.06))
        if pool_mask is not None and cands:
            # 候选框与“池面（外扩一点）”有重叠才算；既能排除池外的固定物体，
            # 又不会把贴着池壁走的机器人误删（标定偏内时尤其重要）。
            h, w = pool_mask.shape[:2]
            pad = max(5, int(round(0.02 * min(h, w))))
            mask_soft = cv2.dilate(pool_mask, np.ones((pad * 2 + 1, pad * 2 + 1), np.uint8))
            keep = []
            for c in cands:
                x1 = max(0, int(round(c["cx"] - c["w"] / 2.0)))
                x2 = min(w, int(round(c["cx"] + c["w"] / 2.0)) + 1)
                y1 = max(0, int(round(c["cy"] - c["h"] / 2.0)))
                y2 = min(h, int(round(c["cy"] + c["h"] / 2.0)) + 1)
                if x2 <= x1 or y2 <= y1:
                    continue
                inside = int(np.count_nonzero(mask_soft[y1:y2, x1:x2]))
                if inside >= 0.12 * float((x2 - x1) * (y2 - y1)):
                    keep.append(c)
            cands = keep
        out[k] = cands
        if k % 10 == 0:
            _check_cancel(cancel)
            _progress(progress, p0 + (p1 - p0) * k / max(1, n_frames),
                      "检测机器人… 第 %d/%d 帧" % (k, n_frames))
    cap.release()
    return out


def _select_robot_track(cands_per_frame, n_frames):
    """多轨迹跟踪：把候选连成若干条轨迹，挑“命中多、尺寸稳定、确实在移动”的那条当机器人。
    反光/水花一般原地闪烁（位移很小），会被移动距离条件筛掉。"""
    max_gap = 12
    tracks = []
    for k in range(n_frames):
        cands = cands_per_frame[k] or []
        if cands:
            for t in tracks:
                t["_assigned"] = None
            for c in cands:
                best, best_d = None, None
                for t in tracks:
                    if t["_assigned"] is not None:
                        continue
                    if k - t["last_k"] > max_gap:
                        continue
                    med = t["med"]
                    if med > 0 and not (0.3 * med <= c["_dia"] <= 3.3 * med):
                        continue
                    d = math.hypot(c["cx"] - t["x"], c["cy"] - t["y"])
                    if d > max(24.0, 3.0 * med):
                        continue
                    if best_d is None or d < best_d:
                        best, best_d = t, d
                if best is not None:
                    best["_assigned"] = c
                    best["pts"].append((k, c))
                    best["hits"] += 1
                    best["last_k"] = k
                    best["x"], best["y"] = c["cx"], c["cy"]
                    best["sizes"].append(c["_dia"])
                    if len(best["sizes"]) > 30:
                        best["sizes"].pop(0)
                    best["med"] = float(np.median(best["sizes"]))
                else:
                    tracks.append({"hits": 1, "pts": [(k, c)], "last_k": k,
                                   "x": c["cx"], "y": c["cy"],
                                   "sizes": [c["_dia"]], "med": c["_dia"],
                                   "_assigned": c})
        # 定期清理长时间未更新的轨迹
        if k % 30 == 0 and len(tracks) > 400:
            tracks = [t for t in tracks if k - t["last_k"] <= max_gap]
    if not tracks:
        return [None] * n_frames, None
    best = None
    for t in tracks:
        pts = t["pts"]
        if t["hits"] < 3:
            continue
        span = max(1, pts[-1][0] - pts[0][0] + 1)
        hit_rate = t["hits"] / float(span)
        path = 0.0
        for (k1, c1), (k2, c2) in zip(pts[:-1], pts[1:]):
            if k2 - k1 <= max_gap:
                path += math.hypot(c2["cx"] - c1["cx"], c2["cy"] - c1["cy"])
        t["path"] = path
        sizes = np.asarray(t["sizes"], dtype=float)
        med = float(np.median(sizes)) if len(sizes) else 0.0
        size_cv = float(sizes.std() / med) if med > 0 else 9.0
        ok_moving = path >= 2.0 * max(1.0, med)
        ok_hits = t["hits"] >= max(6, int(0.05 * n_frames))
        if not (ok_moving and ok_hits):
            continue
        # 命中率高、尺寸稳定、移动里程长为佳
        score = hit_rate * (t["hits"] ** 0.25) + path / max(1.0, med) - 1.2 * size_cv
        if best is None or score > best[0]:
            best = (score, t)
    if best is None:
        return [None] * n_frames, None
    track = best[1]
    dets = [None] * n_frames
    for k, c in track["pts"]:
        dets[k] = c
    return dets, track


def _pick_candidate(cands, prev, gate, size_ref, pred=None):
    """从候选里挑机器人：先按尺寸一致性过滤，再按预测位置门限匹配。
    有上一帧但找不到合理候选时返回 None（视为该帧漏检），避免跳到反光/漂浮物上。"""
    if not cands:
        return None
    if size_ref and len(size_ref) >= 5:
        med = float(np.median(size_ref))
        cands = [c for c in cands if 0.2 * med <= c["area"] <= 3.5 * med]
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
def _raster_sweep(dets, runs, x0, y0, ds, gh, gw, mask_small, n_frames):
    """统一栅格化：沿轨迹画“固定宽度的清洗带”（粗线段，不是一个个圆盘），
    边缘是直边；同一条带子同时生成覆盖并集与经过次数，两张地图白色区域完全一致。"""
    cov = np.zeros((gh, gw), np.uint8)
    pc = np.zeros((gh, gw), np.uint16)
    recent = np.zeros((gh, gw), np.uint8)   # 最近几次扫过的并集，用于合并“同一次经过”
    curve = [None] * n_frames
    pool_area = float(mask_small.sum()) if mask_small.any() else 1.0

    def sweep_segment(x1, y1, x2, y2, r, k):
        """画一条宽度 2r 的清洗带（平头线段），并更新覆盖/经过次数"""
        nonlocal recent
        ri = max(1, int(round(r)))
        xa = max(0, int(min(x1, x2)) - ri - 1)
        xb = min(gw, int(max(x1, x2)) + ri + 2)
        ya = max(0, int(min(y1, y2)) - ri - 1)
        yb = min(gh, int(max(y1, y2)) + ri + 2)
        if xa >= xb or ya >= yb:
            return
        tmp = np.zeros((yb - ya, xb - xa), np.uint8)
        cv2.line(tmp,
                 (int(round(x1)) - xa, int(round(y1)) - ya),
                 (int(round(x2)) - xa, int(round(y2)) - ya),
                 1, thickness=max(1, 2 * ri), lineType=cv2.LINE_8)
        tmp &= mask_small[ya:yb, xa:xb]
        cov[ya:yb, xa:xb] |= tmp
        rec_sub = recent[ya:yb, xa:xb]
        new = tmp & (rec_sub == 0)
        pc[ya:yb, xa:xb] += new
        recent -= (recent > 0).astype(np.uint8)
        rec_sub[:] |= tmp
        cf = float(cov.sum()) / pool_area
        if curve[k] is None or cf > curve[k]:
            curve[k] = cf

    for run in runs:
        pts = []
        for (k, cx, cy) in run:
            d = dets[k]
            r = math.sqrt(max(1.0, d["w"] * d["h"])) / 2.0 * SWATH_FACTOR / ds
            pts.append((float((cx - x0) / ds), float((cy - y0) / ds), max(1.0, r), int(k)))
        if len(pts) < 2:
            continue
        radii = sorted(p[2] for p in pts)
        # 采样间距取半个带宽，保证带子连续、拐角不掉块
        step = max(1.5, radii[len(radii) // 2])
        for (x1, y1, r1, k1), (x2, y2, r2, k2) in zip(pts[:-1], pts[1:]):
            dist = math.hypot(x2 - x1, y2 - y1)
            n = max(1, int(round(dist / step)))
            for i in range(n):
                t = i / float(n)
                t2 = (i + 1) / float(n)
                sweep_segment(x1 + (x2 - x1) * t, y1 + (y2 - y1) * t,
                              x1 + (x2 - x1) * t2, y1 + (y2 - y1) * t2,
                              r1 + (r2 - r1) * t, int(round(k1 + (k2 - k1) * t)))

    last = 0.0
    curve = [last if c is None else c for c in curve]
    for i, c in enumerate(curve):
        if c < last:
            curve[i] = last
        last = curve[i]
    return cov, pc, curve


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


def _runs_from_detections(dets, gap_max, jump_factor=4.0):
    """把检测点按“相邻检测间隔不超过 gap_max 帧、且位移不超过 N 个机身”切成若干段轨迹。
    注意：跳帧/加速视频里，两帧之间机器人可能已经走出很远，不能盲目连线。"""
    runs, cur, last_k = [], [], None
    for k in range(len(dets)):
        d = dets[k]
        if d is None:
            continue
        if cur:
            pd = cur[-1]
            dia = math.sqrt(max(1.0, float(d.get("w", 1.0)) * float(d.get("h", 1.0))))
            dist = math.hypot(d["cx"] - pd[1], d["cy"] - pd[2])
            if (k - last_k) > gap_max or dist > jump_factor * max(1.0, dia):
                if len(cur) > 1:
                    runs.append(cur)
                cur = []
        cur.append((k, d["cx"], d["cy"]))
        last_k = k
    if len(cur) > 1:
        runs.append(cur)
    return runs


def _compose_map(colors, mask_small, calib, crop, ds, legend_items, title, out_w=1100):
    """生成带中文标题/图例的覆盖类地图（返回 BGR ndarray）"""
    arr = colors.copy()
    arr[~mask_small.astype(bool)] = (238, 238, 238)
    gh, gw = arr.shape[:2]
    out_w = int(max(420, min(out_w, gw * 2)))
    ratio = out_w / float(gw)
    out_h = max(60, int(round(gh * ratio)))
    big = cv2.resize(arr, (out_w, out_h), interpolation=cv2.INTER_NEAREST)

    def mp(x, y):
        return (int(round((x - crop[0]) / ds * ratio)), int(round((y - crop[1]) / ds * ratio)))

    if calib.shape == SHAPE_RECT:
        pts = np.array([mp(*p) for p in _order_rect_points(calib.points)], np.int32)
        cv2.polylines(big, [pts], True, (110, 110, 110), 2)
    elif calib.shape == SHAPE_POLY:
        pts = np.array([mp(*p) for p in calib.points], np.int32)
        cv2.polylines(big, [pts], True, (110, 110, 110), 2)
    else:
        center, axes, angle = _ellipse_params(calib)
        cv2.ellipse(big, mp(*center),
                    (max(2, int(axes[0] / 2 / ds * ratio)), max(2, int(axes[1] / 2 / ds * ratio))),
                    angle, 0, 360, (110, 110, 110), 2)

    strip_h = 96
    strip = np.full((strip_h, out_w, 3), 255, np.uint8)
    strip = _text(strip, (16, 10), title, 22, (35, 35, 35))
    n = max(1, len(legend_items))
    slot = (out_w - 32) / float(n)
    for i, (label, bgr) in enumerate(legend_items):
        x = int(16 + slot * i)
        cv2.rectangle(strip, (x, 50), (x + 28, 76), tuple(int(v) for v in bgr), -1)
        cv2.rectangle(strip, (x, 50), (x + 28, 76), (130, 130, 130), 1)
        strip = _text(strip, (x + 36, 52), label, 19, (45, 45, 45))
    return np.vstack([big, strip])


def _draw_coverage_badge(img_bgr, coverage):
    """在地图左上角画"覆盖率 xx.x%"标签（深色底 + 白字）"""
    try:
        txt = "覆盖率 %.1f%%" % (float(coverage) * 100.0)
        f = _font(34)
        pad = 14
        if f is not None and Image is not None:
            pil = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
            d = ImageDraw.Draw(pil)
            box = d.textbbox((0, 0), txt, font=f)
            tw, th = box[2] - box[0], box[3] - box[1]
        else:
            (tw, th), _ = cv2.getTextSize(txt.encode("utf-8") if False else "coverage %.1f%%" % (coverage * 100),
                                          cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
            tw += 8
        x0, y0 = 14, 14
        x1, y1 = x0 + tw + pad * 2, y0 + th + pad * 2
        # 纯色底（不做半透明叠加），保证两张图的徽标区域完全一致
        cv2.rectangle(img_bgr, (x0, y0), (x1, y1), (38, 38, 38), -1)
        return _text(img_bgr, (x0 + pad, y0 + pad - 6), txt, 34, (255, 255, 255))
    except Exception:
        return img_bgr


def _save_maps(out_dir, cov, pass_cnt, mask_small, calib, crop, ds, coverage=0.0):
    """输出两张图：coverage_map.png（两色覆盖图）与 heatmap.png（次数热力图）"""
    # 覆盖图：绿=已清洗，白=未清洗
    colors = np.full(pass_cnt.shape + (3,), 255, np.uint8)
    colors[cov > 0] = (46, 160, 67)
    img = _compose_map(colors, mask_small, calib, crop, ds,
                       [("已清洗", (46, 160, 67)), ("未清洗", (255, 255, 255))],
                       "覆盖图（绿色=已清洗，白色=未清洗）")
    img = _draw_coverage_badge(img, coverage)
    cov_path = os.path.join(out_dir, "coverage_map.png")
    if not _imwrite(cov_path, img):
        cov_path = ""

    # 热力图：白=未清洗，蓝/黄/红/黑 = 1/2/3/4次以上
    colors = np.full(pass_cnt.shape + (3,), 255, np.uint8)
    colors[pass_cnt == 1] = (230, 120, 40)
    colors[pass_cnt == 2] = (0, 215, 255)
    colors[pass_cnt == 3] = (40, 40, 220)
    colors[pass_cnt >= 4] = (0, 0, 0)
    img = _compose_map(colors, mask_small, calib, crop, ds,
                       [("未清洗", (255, 255, 255)), ("1次", (230, 120, 40)),
                        ("2次", (0, 215, 255)), ("3次", (40, 40, 220)),
                        ("4次及以上", (0, 0, 0))],
                       "热力图（白=未清洗；蓝/黄/红/黑=1/2/3/4次以上）")
    img = _draw_coverage_badge(img, coverage)
    heat_path = os.path.join(out_dir, "heatmap.png")
    if not _imwrite(heat_path, img):
        heat_path = ""
    return cov_path, heat_path


def _save_curve_chart(path, curve_real, start_sec, speed, duration_real):
    """输出覆盖率-真实时间曲线图 coverage_curve.png"""
    W, H = 1100, 540
    L, R, T, B = 92, 44, 96, 78
    pw, ph = W - L - R, H - T - B
    t0, t1 = curve_real[0][0], curve_real[-1][0]
    span = max(1.0, t1 - t0)
    x_of = lambda t: L + (t - t0) / span * pw
    y_of = lambda c: T + (1.0 - max(0.0, min(1.0, c))) * ph

    img = None
    if Image is not None:
        img = Image.new("RGB", (W, H), (255, 255, 255))
        d = ImageDraw.Draw(img)
        f_title, f_sub, f_tick, f_note = _font(26), _font(16), _font(15), _font(18)
        # 网格
        for i in range(5):
            frac = i / 4.0
            y = y_of(frac)
            for x in range(int(L), int(W - R), 8):
                d.line([(x, y), (x + 4, y)], fill=(215, 215, 215), width=1)
            if f_tick:
                d.text((L - 14, y - 9), "%d%%" % int(frac * 100), font=f_tick,
                       fill=(110, 110, 110), anchor="ra")
        # x 轴刻度
        for i in range(7):
            frac = i / 6.0
            t = t0 + span * frac
            x = x_of(t)
            d.line([(x, T), (x, T + ph)], fill=(235, 235, 235), width=1)
            label = _fmt_clock(t) if start_sec is not None else _fmt_min(t)
            if f_tick:
                d.text((x, H - B + 20), label, font=f_tick, fill=(110, 110, 110), anchor="ma")
        # 面积 + 折线
        pts = [(x_of(t), y_of(c)) for t, c in curve_real[::max(1, len(curve_real) // 900)]] or [(L, T + ph)]
        poly = [(L, T + ph)] + pts + [(W - R, T + ph)]
        d.polygon(poly, fill=(219, 234, 252))
        d.line(pts, fill=(37, 99, 235), width=3, joint="curve")
        # 阈值线
        for thr, label in ((0.5, "50%"), (0.8, "80%"), (0.9, "90%")):
            y = y_of(thr)
            d.line([(L, y), (W - R, y)], fill=(190, 210, 190), width=1)
            d.text((W - R - 4, y - 20), label, font=f_tick, fill=(120, 150, 120), anchor="ra")
        # 终点标注
        ex, ey = x_of(t1), y_of(curve_real[-1][1])
        d.ellipse([ex - 5, ey - 5, ex + 5, ey + 5], fill=(37, 99, 235))
        d.text((min(ex + 10, W - R - 4), ey - 24), "%.1f%%" % (curve_real[-1][1] * 100),
               font=f_note, fill=(37, 99, 235), anchor="ra" if ex > W - 160 else "la")
        # 标题
        d.text((L, 26), "覆盖率随时间变化", font=f_title, fill=(25, 25, 25))
        if f_sub:
            span_txt = ("视频 %.1f 秒 × %.0f 倍速 = 真实 %.1f 分钟" % (duration_real / max(1e-6, speed),
                                                                  speed, duration_real / 60.0))
            if start_sec is not None:
                span_txt += "  ·  %s → %s" % (_fmt_clock(t0), _fmt_clock(t1))
            d.text((L, 62), span_txt, font=f_sub, fill=(110, 110, 110))
        try:
            img.save(path)
            if os.path.isfile(path) and os.path.getsize(path) > 0:
                return path
        except Exception:
            pass

    # 无 Pillow 时的退路：OpenCV 画英文版
    canvas = np.full((H, W, 3), 255, np.uint8)
    for i in range(5):
        frac = i / 4.0
        y = int(y_of(frac))
        cv2.line(canvas, (L, y), (W - R, y), (220, 220, 220), 1)
        cv2.putText(canvas, "%d%%" % int(frac * 100), (L - 46, y + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (110, 110, 110), 1)
    pts = np.array([[int(x_of(t)), int(y_of(c))] for t, c in curve_real[::max(1, len(curve_real) // 900)]]
                   or [[L, T + ph]], np.int32)
    cv2.polylines(canvas, [pts], False, (235, 99, 37), 3)
    cv2.putText(canvas, "coverage vs real time", (L, 40), cv2.FONT_HERSHEY_SIMPLEX,
                0.9, (30, 30, 30), 2)
    if _imwrite(path, canvas):
        return path
    return ""


def _save_trajectory_image(path, frame, calib, pts, coverage, start_sec, speed, fps):
    """输出完整轨迹图 trajectory.png（全分辨率，含起点/终点）"""
    vis = frame.copy()
    if calib.shape == SHAPE_RECT:
        cv2.polylines(vis, [np.array(_order_rect_points(calib.points), np.int32)],
                      True, (0, 255, 255), 3)
    elif calib.shape == SHAPE_POLY:
        cv2.polylines(vis, [np.array(calib.points, np.int32)], True, (0, 255, 255), 3)
    else:
        center, axes, angle = _ellipse_params(calib)
        cv2.ellipse(vis, (int(center[0]), int(center[1])),
                    (max(3, int(axes[0] / 2)), max(3, int(axes[1] / 2))), angle, 0, 360,
                    (0, 255, 255), 3)
    last_k = None
    for (k, x, y) in pts:
        p = (int(round(x)), int(round(y)))
        if last_k is not None and k - last_k <= GAP_FRAMES_MAX:
            cv2.line(vis, prev_p, p, (0, 150, 255), 4, cv2.LINE_AA)
        prev_p = p
        last_k = k
    if pts:
        sx, sy = int(round(pts[0][1])), int(round(pts[0][2]))
        ex, ey = int(round(pts[-1][1])), int(round(pts[-1][2]))
        cv2.circle(vis, (sx, sy), 12, (0, 200, 0), -1)
        cv2.circle(vis, (sx, sy), 12, (255, 255, 255), 2)
        cv2.circle(vis, (ex, ey), 12, (0, 0, 230), -1)
        cv2.circle(vis, (ex, ey), 12, (255, 255, 255), 2)
        vis = _text(vis, (max(8, sx + 18), max(26, sy - 18)), "起点", 26, (255, 255, 255))
        vis = _text(vis, (max(8, ex + 18), max(26, ey - 18)), "终点", 26, (255, 255, 255))
    title = "完整轨迹 · 覆盖率 %.1f%%" % (coverage * 100)
    vis = _text(vis, (20, 138), title, 32, (0, 0, 0))
    vis = _text(vis, (18, 136), title, 32, (255, 255, 255))
    if start_sec is not None and pts:
        t_end = pts[-1][0] / max(1e-6, fps) * speed
        sub = "真实时间 %s → %s" % (_fmt_clock(start_sec), _fmt_clock(start_sec + t_end))
        vis = _text(vis, (20, 184), sub, 26, (0, 0, 0))
        vis = _text(vis, (18, 182), sub, 26, (255, 255, 255))
    return path if _imwrite(path, vis) else ""


# ---------------------------------------------------------------- 叠加视频# ---------------------------------------------------------------- 叠加视频
def _open_writer(path, size, fps, ffmpeg):
    if ffmpeg:
        cmd = [ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
               "-s", "%dx%d" % size, "-r", "%.3f" % fps, "-i", "-",
               "-c:v", "libx264", "-preset", "ultrafast", "-crf", "24",
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
def _render_overlay(video_path, dets, n_frames, fps, vw, vh, calib, curve,
                    out_dir, overlay_width, start_sec, speed, stride,
                    progress, cancel, warnings):
    """渲染轨迹叠加视频（独立函数：支持按需跳过）"""
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
    poly_pts = None
    if calib.shape == SHAPE_RECT:
        rect_pts = np.array(_order_rect_points(calib.points), np.int32)
    elif calib.shape == SHAPE_POLY:
        poly_pts = np.array(calib.points, np.float32).reshape(-1, 2)
    else:
        center, axes, angle = _ellipse_params(calib)
    cap = cv2.VideoCapture(video_path)
    # 增量轨迹层：每帧只画新增的一小段，避免长视频 O(n²) 重绘
    trail_mask = np.zeros((out_h, out_w), np.uint8)
    start_mask = np.zeros((out_h, out_w), np.uint8)
    trail_prev = None
    trail_last_k = None
    for k in range(n_frames):
        ok, frame = cap.read()
        if not ok:
            break
        vis = cv2.resize(frame, (out_w, out_h), interpolation=cv2.INTER_AREA)
        sx = out_w / float(vw)
        if calib.shape == SHAPE_RECT:
            cv2.polylines(vis, [(rect_pts * sx).astype(np.int32)], True, (0, 255, 255), 2)
        elif calib.shape == SHAPE_POLY:
            cv2.polylines(vis, [(poly_pts * sx).astype(np.int32)], True, (0, 255, 255), 2)
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
            if trail_prev is not None:
                dia = math.sqrt(max(1.0, d["w"] * d["h"])) * sx
                far = math.hypot(pt[0] - trail_prev[0], pt[1] - trail_prev[1]) > 4.0 * max(6.0, dia)
                if (k - trail_last_k) <= max(GAP_FRAMES_MAX, stride * 2) and not far:
                    cv2.line(trail_mask, trail_prev, pt, 255, 3, cv2.LINE_AA)
            if trail_prev is None:
                cv2.circle(start_mask, pt, 7, 255, -1)
            trail_prev, trail_last_k = pt, k
        # 合成完整累计轨迹（掩膜索引，速度快）
        if trail_prev is not None:
            vis[trail_mask > 0] = (0, 165, 255)
            vis[start_mask > 0] = (0, 200, 0)
        cov_now = curve[k][1]
        cv2.putText(vis, "coverage %.1f%%" % (cov_now * 100),
                    (max(8, out_w - 260), 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        if start_sec is not None:
            t_txt = _fmt_clock(start_sec + k / fps * speed)
        else:
            t_txt = "T+%.1fs" % (k / fps * speed)
        cv2.putText(vis, t_txt, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (255, 255, 255), 2, cv2.LINE_AA)
        _write_frame(writer, vis)
        if k % 30 == 0:
            _check_cancel(cancel)
            _progress(progress, 0.72 + 0.26 * k / max(1, n_frames),
                      "渲染轨迹视频… 第 %d/%d 帧" % (k, n_frames))
    # 结尾定格：完整轨迹展示
    try:
        if trail_prev is not None:
            freeze = vis.copy()
            cv2.putText(freeze, "END  full trajectory", (max(10, out_w // 2 - 150), out_h // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
            for _ in range(int(round(2.5 * fps))):
                _write_frame(writer, freeze)
    except Exception:
        pass
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

    return ov_path


def analyze(video_path, calib, out_dir, *, target="auto", sensitivity=0.5,
            progress=None, cancel=None, overlay_width=1280, make_video=True,
            start_time=None, speed_factor=1.0, seed=None):
    if target not in ("auto", "dark", "bright"):
        raise ValueError("未知目标类型：%s" % target)
    if not os.path.isfile(video_path):
        raise RuntimeError("视频文件不存在：%s" % video_path)
    if not isinstance(calib, Calibration) or not calib.points:
        raise RuntimeError("尚未完成池面标定")
    os.makedirs(out_dir, exist_ok=True)
    start_sec = _parse_hms(start_time)
    try:
        speed = float(speed_factor or 1.0)
    except (TypeError, ValueError):
        speed = 1.0
    if not (speed > 0):
        speed = 1.0

    info = video_info(video_path)
    fps, n_frames = info["fps"] or 25.0, info["frames"]
    vw, vh = info["width"], info["height"]
    if n_frames <= 0:
        raise RuntimeError("视频帧数为 0，文件可能损坏")

    pool_mask = build_pool_mask(calib.shape, (vw, vh), calib)
    pool_area = int(np.count_nonzero(pool_mask))
    if pool_area < 0.02 * vw * vh:
        raise RuntimeError("标定的池面区域过小（不到画面 2%），请重新标定")

    # 检测 ROI：池面外扩一圈（按池子尺寸），容忍机器人贴边/标定略小于实际活动范围
    try:
        _ys, _xs = np.where(pool_mask > 0)
        _bw = int(_xs.max() - _xs.min() + 1); _bh = int(_ys.max() - _ys.min() + 1)
    except Exception:
        _bw, _bh = vw, vh
    pad = max(12, int(0.01 * math.hypot(vw, vh)), int(0.10 * min(_bw, _bh)))
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

    det_scale = 0.5 if max(vw, vh) >= 2560 else 1.0
    stride = 2 if n_frames >= 1500 else 1
    if seed:
        # 手动框选模式：用模板跟踪（适合小目标/浅色/反光干扰大的机位）
        warnings = []
        if start_time and start_sec is None:
            warnings.append("起始时间格式不正确（应为 HH:MM:SS），曲线图按相对时间显示。")
        _progress(progress, 0.04, "模板跟踪机器人…")
        dets = _template_track(video_path, seed, n_frames, fps, pool_mask, progress, cancel,
                               sensitivity=sens, p0=0.06, p1=0.66)
        detected = sum(1 for d in dets if d is not None)
        warnings.append("已使用手动框选模板跟踪（第 %d 帧框选）。" % int(seed.get("frame", 0)))
        if detected < 5:
            raise RuntimeError("模板跟踪没有跟上机器人：请重新框选（框住机器人本体、选机器人清晰的一帧）。")
    else:
        # 检测分辨率：只有 2K/4K 画面才降采样；1080p 保持全分辨率（小目标需要）
        dw = int(round(vw * det_scale)); dh = int(round(vh * det_scale))
        if det_scale != 1.0:
            roi_det = cv2.resize(roi, (dw, dh), interpolation=cv2.INTER_NEAREST)
            pool_det = cv2.resize(pool_mask, (dw, dh), interpolation=cv2.INTER_NEAREST)
            pool_area_det = int(np.count_nonzero(pool_det))
            s2 = min(dw, dh) / 1080.0
            kern_det = {
                "open": np.ones((_odd(5 * max(s2, 0.5)), _odd(5 * max(s2, 0.5))), np.uint8),
                "close": np.ones((_odd(25 * max(s2, 0.5)), _odd(25 * max(s2, 0.5))), np.uint8),
            }
        else:
            roi_det, pool_area_det, kern_det = roi, pool_area, kern

        _progress(progress, 0.0, "读取视频背景…")
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
                if det_scale != 1.0:
                    fr = cv2.resize(fr, (dw, dh), interpolation=cv2.INTER_AREA)
                frames_bg.append(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY))
        cap.release()
        if not frames_bg:
            raise RuntimeError("无法读取视频帧")
        bg_gray = np.median(np.stack(frames_bg), axis=0).astype(np.uint8)

        warnings = []
        if det_scale != 1.0 or stride > 1:
            warnings.append("已启用加速模式：检测分辨率 ×%.2f、隔 %d 帧检测（不影响覆盖率统计口径）。"
                            % (det_scale, stride))
        if start_time and start_sec is None:
            warnings.append("起始时间格式不正确（应为 HH:MM:SS），曲线图按相对时间显示。")

        if target == "dark":
            modes = ["dark"]
        elif target == "bright":
            modes = ["bright"]
        else:
            modes = ["dark", "bright"]
        _progress(progress, 0.04, "检测目标（多轨迹跟踪）…")
        _pm = pool_mask if det_scale == 1.0 else cv2.resize(
            pool_mask, (bg_gray.shape[1], bg_gray.shape[0]), interpolation=cv2.INTER_NEAREST)
        cands_all = _gather_candidates(video_path, bg_gray, roi_det, modes, thr, pool_area_det,
                                       kern_det, n_frames, stride, progress, cancel, 0.06, 0.58,
                                       pool_mask=_pm)
        dets, robot_track = _select_robot_track(cands_all, n_frames)
        detected = sum(1 for d in dets if d is not None)
        if detected < 5:
            raise RuntimeError("未检测到机器人。可调高灵敏度、切换目标类型，或检查池面标定是否准确。")
        if detected < 0.15 * n_frames:
            warnings.append("机器人可见帧数较少（%d/%d），覆盖率可能偏低；可调高灵敏度或改用亮色/暗色目标类型。"
                            % (detected, n_frames))


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

    pts_plane = [(k, float(d["cx"]), float(d["cy"])) for k, d in enumerate(dets) if d is not None]
    # 轨迹平滑：检测框中心有 1~3 像素抖动，平滑后再栅格化，避免把抖动画成小圆弧
    dets_cov = [dict(d) if d else None for d in dets]
    for k in range(n_frames):
        if dets[k] is None:
            continue
        xs, ys, ws, hs = [], [], [], []
        for j in range(max(0, k - 2), min(n_frames, k + 3)):
            dj = dets[j]
            if dj is None:
                continue
            xs.append(dj["cx"]); ys.append(dj["cy"])
            ws.append(dj["w"]); hs.append(dj["h"])
        if xs:
            dets_cov[k]["cx"] = sum(xs) / len(xs)
            dets_cov[k]["cy"] = sum(ys) / len(ys)
            dets_cov[k]["w"] = sum(ws) / len(ws)
            dets_cov[k]["h"] = sum(hs) / len(hs)
    runs = _runs_from_detections(dets_cov, GAP_FRAMES_MAX)
    cov, pass_cnt, curve_cov = _raster_sweep(dets_cov, runs, x0, y0, ds, gh, gw,
                                             mask_small, n_frames)
    _check_cancel(cancel)
    coverage = float(cov.sum()) / max(1e-6, pool_area_small)
    curve = [(k / fps, float(c)) for k, c in enumerate(curve_cov)]
    pcm = pass_cnt > 0
    avg_passes = float(pass_cnt[pcm].mean()) if pcm.any() else 0.0
    repeat_frac = float((pass_cnt[pcm] > 1).mean()) if pcm.any() else 0.0

    path_len_px = 0.0
    for run in runs:
        for a, b in zip(run[:-1], run[1:]):
            path_len_px += math.hypot(b[1] - a[1], b[2] - a[2])

    # 覆盖图 + 热力图（两张）
    _progress(progress, 0.68, "生成覆盖图/热力图…")
    cov_path, heatmap_path = "", ""
    try:
        cov_path, heatmap_path = _save_maps(out_dir, cov, pass_cnt, mask_small, calib, (x0, y0), ds, coverage)
    except Exception as exc:
        cov_path = heatmap_path = ""
        warnings.append("覆盖图/热力图生成失败：%s" % exc)
    for _label, _p in (("覆盖图", cov_path), ("热力图", heatmap_path)):
        if not _p:
            warnings.append("%s未能生成（请检查输出目录是否可写）。" % _label)

    # 覆盖率-真实时间 曲线图
    curve_real = [((start_sec + t * speed) if start_sec is not None else t * speed, c)
                  for t, c in curve]
    duration_real = (n_frames - 1) / fps * speed
    curve_path = os.path.join(out_dir, "coverage_curve.png")
    try:
        _saved = _save_curve_chart(curve_path, curve_real, start_sec, speed, duration_real)
        if not _saved or not os.path.isfile(_saved):
            curve_path = ""
    except Exception as exc:
        curve_path = ""
        warnings.append("覆盖率曲线图生成失败：%s" % exc)
    if not curve_path:
        warnings.append("覆盖率曲线图未能生成（请检查输出目录是否可写）。")

    # 完整轨迹图
    trajectory_image = os.path.join(out_dir, "trajectory.png")
    try:
        base_frame = probe_frame(video_path, 0)
        _saved = _save_trajectory_image(trajectory_image, base_frame, calib, pts_plane,
                                        coverage, start_sec, speed, fps)
        if not _saved or not os.path.isfile(_saved):
            trajectory_image = ""
    except Exception as exc:
        trajectory_image = ""
        warnings.append("轨迹图生成失败：%s" % exc)

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

    # 叠加视频（可跳过）
    ov_path = ""
    if make_video:
        ov_path = _render_overlay(video_path, dets, n_frames, fps, vw, vh, calib, curve,
                                  out_dir, int(overlay_width), start_sec, speed, stride,
                                  progress, cancel, warnings)
    else:
        warnings.append("已按设置跳过轨迹视频生成（提速）。")

    _progress(progress, 0.99, "保存结果…")
    result = AnalyzeResult(
        video=video_path, out_dir=os.path.abspath(out_dir), frames=int(n_frames),
        fps=float(fps), width=int(vw), height=int(vh), detected=int(detected),
        coverage=float(coverage), uncovered=float(1.0 - coverage),
        avg_passes=float(avg_passes), pool_area_px=int(pool_area),
        path_len_px=float(path_len_px), overlay_path=(os.path.abspath(ov_path) if ov_path else ""),
        heatmap_path=os.path.abspath(heatmap_path) if heatmap_path else "",
        trajectory_csv=os.path.abspath(traj_csv), summary_json="",
        coverage_path=os.path.abspath(cov_path) if cov_path else "",
        curve_path=os.path.abspath(curve_path) if curve_path else "",
        trajectory_image_path=os.path.abspath(trajectory_image) if trajectory_image else "",
        curve=[[float(t), float(c)] for t, c in curve],
        warnings=list(warnings),
        shape=calib.shape,
    )
    _progress(progress, 1.0, "完成")
    return result


# ---------------------------------------------------------------- 机位模板
def _safe_template_name(name):
    name = re.sub(r'[\\/:*?"<>|\s]+', "_", str(name or "")).strip("._")
    return name[:60] or "template"


def save_template(templates_dir, name, calib, source_video=""):
    """把当前标定保存为机位模板，返回 json 路径"""
    if not isinstance(calib, Calibration):
        raise RuntimeError("没有可保存的标定")
    os.makedirs(templates_dir, exist_ok=True)
    fname = _safe_template_name(name) + ".json"
    path = os.path.join(templates_dir, fname)
    payload = {
        "version": 1,
        "name": str(name),
        "shape": calib.shape,
        "points": [[float(x), float(y)] for x, y in calib.points],
        "source_video": str(source_video or ""),
        "created": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=1)
    return path


def list_templates(templates_dir):
    """列出模板目录里的机位模板，按创建时间倒序"""
    out = []
    if not templates_dir or not os.path.isdir(templates_dir):
        return out
    for fn in os.listdir(templates_dir):
        if not fn.lower().endswith(".json"):
            continue
        path = os.path.join(templates_dir, fn)
        try:
            with open(path, "r", encoding="utf-8") as f:
                d = json.load(f)
            shape = d.get("shape")
            pts = d.get("points") or []
            if shape not in (SHAPE_RECT, SHAPE_ELLIPSE, SHAPE_POLY) or len(pts) < 3:
                continue
            out.append({
                "name": d.get("name") or os.path.splitext(fn)[0],
                "path": os.path.abspath(path),
                "shape": shape,
                "points": [[float(p[0]), float(p[1])] for p in pts],
                "source_video": d.get("source_video", ""),
                "created": d.get("created", ""),
                "mtime": os.path.getmtime(path),
            })
        except Exception:
            continue
    out.sort(key=lambda x: -x["mtime"])
    return out


def load_template(path):
    """读取机位模板，返回 Calibration"""
    with open(path, "r", encoding="utf-8") as f:
        d = json.load(f)
    shape = d.get("shape")
    pts = d.get("points") or []
    if shape not in (SHAPE_RECT, SHAPE_ELLIPSE, SHAPE_POLY):
        raise RuntimeError("模板格式不正确：未知池型")
    if len(pts) < (4 if shape == SHAPE_RECT else 3):
        raise RuntimeError("模板格式不正确：标定点不足")
    return Calibration(shape, [[float(p[0]), float(p[1])] for p in pts])


# ---------------------------------------------------------------- 历史
HISTORY_FILE = "_history.jsonl"


def record_history(root_dir, result):
    """把一次分析记录追加到输出根目录的内部索引（不再在结果目录里生成 summary.json）"""
    try:
        if not root_dir:
            return ""
        os.makedirs(root_dir, exist_ok=True)
        curve = list(getattr(result, "curve", []) or [])
        step = max(1, len(curve) // 200)
        small = [[round(float(t), 3), round(float(c), 4)] for t, c in curve[::step]]
        rec = {
            "video": result.video,
            "out_dir": os.path.abspath(result.out_dir),
            "coverage": float(result.coverage),
            "uncovered": float(getattr(result, "uncovered", 1.0 - result.coverage)),
            "detected": int(result.detected),
            "frames": int(result.frames),
            "avg_passes": float(getattr(result, "avg_passes", 0.0)),
            "shape": str(getattr(result, "shape", "")),
            "fps": float(getattr(result, "fps", 0.0)),
            "created": time.strftime("%Y-%m-%d %H:%M:%S"),
            "mtime": time.time(),
            "curve": small,
            "overlay_path": getattr(result, "overlay_path", ""),
            "coverage_path": getattr(result, "coverage_path", ""),
            "heatmap_path": getattr(result, "heatmap_path", ""),
            "curve_path": getattr(result, "curve_path", ""),
            "trajectory_image_path": getattr(result, "trajectory_image_path", ""),
            "warnings": list(getattr(result, "warnings", []) or []),
        }
        path = os.path.join(root_dir, HISTORY_FILE)
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        return path
    except Exception:
        return ""


def list_history(root_dir):
    """读取历史记录：优先内部索引 _history.jsonl；旧版结果目录回退读 summary.json"""
    out = []
    if not root_dir or not os.path.isdir(root_dir):
        return out
    path = os.path.join(root_dir, HISTORY_FILE)
    if os.path.isfile(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict):
                        continue
                    if not os.path.isdir(rec.get("out_dir", "")):
                        continue
                    rec.setdefault("mtime", 0.0)
                    out.append(rec)
        except Exception:
            out = []
    if not out:  # 兼容旧版：扫描 summary.json
        for name in os.listdir(root_dir):
            d = os.path.join(root_dir, name)
            sp = os.path.join(d, "summary.json")
            if not os.path.isfile(sp):
                continue
            try:
                with open(sp, "r", encoding="utf-8") as f:
                    sm = json.load(f)
                out.append({
                    "out_dir": os.path.abspath(d),
                    "video": sm.get("video", ""),
                    "coverage": float(sm.get("coverage", 0.0)),
                    "detected": int(sm.get("detected", 0)),
                    "frames": int(sm.get("frames", 0)),
                    "shape": sm.get("shape", ""),
                    "mtime": os.path.getmtime(sp),
                    "created": sm.get("created", ""),
                    "curve": sm.get("curve", []),
                })
            except Exception:
                continue
    out.sort(key=lambda x: -float(x.get("mtime") or 0.0))
    return out
