# -*- coding: utf-8 -*-
"""泳池机器人覆盖率分析 v0.1 —— Tkinter 桌面界面。

分析算法在 core.py（接口契约见 INTERFACE.md）。本模块遵守两条硬约束：

1. import 本模块不会创建 Tk 根窗口、不会启动线程；窗口只在 main() 里创建。
2. 分析在后台线程执行，绝不直接操作 Tk 控件；进度 / 结果 / 异常全部经
   queue 交回主线程，由 root.after() 轮询刷新界面。

运行：python app.py
"""

from __future__ import annotations

import base64
import json
import math
import os
import queue
import re
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import filedialog
from tkinter import font as tkfont
from tkinter import messagebox, ttk

try:  # 缺少 core.py 时也允许 import 成功，main() 会给出中文提示
    import core
except Exception:  # pragma: no cover - 仅在文件缺失/依赖损坏时触发
    core = None

APP_TITLE = "泳池机器人覆盖率分析 v0.1"
MIN_WINDOW_WIDTH = 900
MIN_WINDOW_HEIGHT = 680

SHAPE_RECT = "rect"
SHAPE_ELLIPSE = "ellipse"
SHAPE_LABELS = {SHAPE_RECT: "矩形", SHAPE_ELLIPSE: "圆形（椭圆）"}

TARGET_LABELS = ("自动", "暗色机器人", "亮色机器人")
TARGET_MAP = {"自动": "auto", "暗色机器人": "dark", "亮色机器人": "bright"}

MARK_COLORS = ("#ff5252", "#ffd740", "#69f0ae", "#40c4ff")
CURVE_COLORS = ("#1565c0", "#c62828")
VIDEO_TYPES = [
    ("视频文件", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv *.m4v *.mpg *.mpeg"),
    ("所有文件", "*.*"),
]


# ============================================================ 通用小工具
def app_dir() -> str:
    """程序所在目录（打包后是 exe 所在目录，源码运行时是工程目录）。"""
    if getattr(sys, "frozen", False):
        return os.path.dirname(os.path.abspath(sys.executable))
    return os.path.dirname(os.path.abspath(__file__))


def default_reports_root() -> str:
    """默认输出根目录：<程序目录>/PoolCoverageReports。"""
    return os.path.join(app_dir(), "PoolCoverageReports")


def format_duration(seconds: float) -> str:
    try:
        seconds = max(0.0, float(seconds))
    except (TypeError, ValueError):
        return "未知"
    total = int(round(seconds))
    return "%d 分 %02d 秒" % (total // 60, total % 60)


def format_clock(mtime) -> str:
    try:
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(float(mtime)))
    except Exception:
        return "未知时间"


def percent_text(value, digits: int = 1) -> str:
    try:
        return ("%." + str(digits) + "f%%") % (float(value) * 100.0)
    except Exception:
        return "—"


def open_path(path: str) -> None:
    """用系统默认程序打开文件/文件夹；Windows 上即 os.startfile。"""
    if not path:
        raise FileNotFoundError("路径为空")
    path = os.path.abspath(str(path))
    if not os.path.exists(path):
        raise FileNotFoundError("文件不存在：\n%s" % path)
    if hasattr(os, "startfile"):
        os.startfile(path)  # type: ignore[attr-defined]
        return
    import subprocess

    opener = "open" if sys.platform == "darwin" else "xdg-open"
    subprocess.Popen([opener, path])


def _rgb_to_photo_slow(rgb) -> "tk.PhotoImage":
    """纯 Tk 的兜底实现：逐行 put 像素（较慢，仅当 PPM 方式不可用时使用）。"""
    height, width = rgb.shape[:2]
    photo = tk.PhotoImage(width=width, height=height)
    chunk = 16
    for top in range(0, height, chunk):
        block = rgb[top:top + chunk]
        lines = []
        for row in block:
            lines.append("{" + " ".join(
                "#%02x%02x%02x" % (int(r), int(g), int(b)) for r, g, b in row) + "}")
        photo.put("\n".join(lines), to=(0, top))
    return photo


def _rgb_to_photo(rgb) -> "tk.PhotoImage":
    """numpy RGB 数组 -> tk.PhotoImage（优先 PPM/base64，速度快）。"""
    height, width = rgb.shape[:2]
    header = ("P6\n%d %d\n255\n" % (width, height)).encode("ascii")
    payload = base64.b64encode(header + rgb.tobytes()).decode("ascii")
    for kwargs in ({"data": payload, "format": "ppm"}, {"data": payload}):
        try:
            return tk.PhotoImage(**kwargs)
        except tk.TclError:
            continue
    return _rgb_to_photo_slow(rgb)


def frame_to_photo(frame, max_w: int, max_h: int):
    """把 core.probe_frame 返回的 BGR 帧缩放到画布内，返回 (photo, 宽, 高, 缩放比)。"""
    import numpy as np

    arr = np.asarray(frame)
    if arr.ndim == 2:
        arr = np.stack((arr, arr, arr), axis=-1)
    if arr.ndim != 3 or arr.shape[0] < 1 or arr.shape[1] < 1:
        raise ValueError("画面数据为空")
    if arr.shape[2] > 3:
        arr = arr[:, :, :3]
    height, width = arr.shape[:2]
    scale = min(max_w / float(width), max_h / float(height), 1.0)
    disp_w = max(1, int(round(width * scale)))
    disp_h = max(1, int(round(height * scale)))
    if disp_w != width or disp_h != height:
        yi = np.clip((np.arange(disp_h) * (height / float(disp_h))).astype(np.int64), 0, height - 1)
        xi = np.clip((np.arange(disp_w) * (width / float(disp_w))).astype(np.int64), 0, width - 1)
        small = arr[yi][:, xi]
    else:
        small = arr
    rgb = np.ascontiguousarray(small[:, :, 2::-1])
    if rgb.dtype != np.uint8:
        rgb = rgb.clip(0, 255).astype(np.uint8)
    return _rgb_to_photo(rgb), disp_w, disp_h, scale


def load_curve(out_dir: str, fallback_coverage: float = 0.0):
    """读取某次历史记录的 summary.json 里的覆盖率曲线，返回 (点列表, 提示文字)。"""
    points = []
    note = ""
    path = os.path.join(out_dir or "", "summary.json")
    data = None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
    except Exception:
        note = "读取 summary.json 失败，按最终覆盖率画平线"
    raw = None
    if isinstance(data, dict):
        for key in ("curve", "coverage_curve", "coverageCurve"):
            value = data.get(key)
            if isinstance(value, list) and value:
                raw = value
                break
    if raw:
        for item in raw:
            t = c = None
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                t, c = item[0], item[1]
            elif isinstance(item, dict):
                for key in ("t", "time", "time_s", "sec", "second", "seconds"):
                    if key in item:
                        t = item[key]
                        break
                for key in ("coverage", "cov", "c"):
                    if key in item:
                        c = item[key]
                        break
            try:
                t = float(t)
                c = float(c)
            except (TypeError, ValueError):
                continue
            if math.isnan(t) or math.isnan(c):
                continue
            points.append((t, min(1.0, max(0.0, c))))
        points.sort(key=lambda p: p[0])
    if not points:
        points = [(0.0, float(fallback_coverage or 0.0))]
        if not note:
            note = "该记录没有曲线数据，按最终覆盖率画平线"
    return points, note


# ============================================================ 标定窗口
class CalibrationDialog(tk.Toplevel):
    """画面标定窗口：Canvas 显示缩放后的首帧，鼠标点选池底边界。"""

    MAX_CANVAS_W = 1000
    MAX_CANVAS_H = 620

    def __init__(self, master, video_path: str, shape: str, frame, initial=None):
        super().__init__(master)
        self.withdraw()
        self.result = None
        self._video_path = video_path
        self._shape = shape
        self._required = 4 if shape == SHAPE_RECT else 3
        self._points = []  # 原视频像素坐标
        self._photo = None
        self._scale = 1.0
        self._busy = False

        screen_w = self.winfo_screenwidth()
        screen_h = self.winfo_screenheight()
        max_w = max(480, min(self.MAX_CANVAS_W, int(screen_w * 0.9)))
        max_h = max(360, min(self.MAX_CANVAS_H, int(screen_h * 0.72)))
        self._photo, disp_w, disp_h, self._scale = frame_to_photo(frame, max_w, max_h)

        self.title("画面标定 —— %s" % SHAPE_LABELS.get(shape, shape))
        self.transient(master)
        self.resizable(False, False)

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        self.canvas = tk.Canvas(outer, width=disp_w, height=disp_h, bg="#101010",
                                highlightthickness=1, highlightbackground="#666666",
                                cursor="crosshair")
        self.canvas.pack()
        self.canvas.create_image(0, 0, anchor="nw", image=self._photo)
        self.canvas.bind("<Button-1>", self._on_click)

        self.var_hint = tk.StringVar(value="")
        ttk.Label(outer, textvariable=self.var_hint, foreground="#333333").pack(
            fill="x", pady=(8, 6))

        buttons = ttk.Frame(outer)
        buttons.pack()
        self.btn_clear = ttk.Button(buttons, text="清除", width=10, command=self._clear)
        self.btn_auto = ttk.Button(buttons, text="自动识别", width=12,
                                   command=self._auto_detect)
        self.btn_ok = ttk.Button(buttons, text="确定", width=10, command=self._confirm)
        self.btn_cancel = ttk.Button(buttons, text="取消", width=10, command=self.destroy)
        for i, btn in enumerate((self.btn_clear, self.btn_auto, self.btn_ok, self.btn_cancel)):
            btn.grid(row=0, column=i, padx=6)

        if initial is not None and getattr(initial, "shape", None) == shape:
            pts = [(float(p[0]), float(p[1])) for p in getattr(initial, "points", [])]
            if len(pts) == self._required:
                self._points = pts

        self.bind("<Escape>", lambda _e: self.destroy())
        self._redraw()

        self.update_idletasks()
        try:
            px, py = master.winfo_rootx(), master.winfo_rooty()
            pw, ph = master.winfo_width(), master.winfo_height()
            x = max(0, px + (pw - self.winfo_reqwidth()) // 2)
            y = max(0, py + (ph - self.winfo_reqheight()) // 3)
            self.geometry("+%d+%d" % (x, y))
        except tk.TclError:
            pass
        self.deiconify()
        self.grab_set()
        self.focus_set()

    # -------------------------------------------------- 坐标换算
    def _to_original(self, x: float, y: float):
        if self._scale <= 0:
            return float(x), float(y)
        return float(x) / self._scale, float(y) / self._scale

    def _to_display(self, x: float, y: float):
        return float(x) * self._scale, float(y) * self._scale

    # -------------------------------------------------- 交互
    def _on_click(self, event):
        if self._busy:
            return
        if len(self._points) >= self._required:
            self._update_hint()
            return
        self._points.append(self._to_original(event.x, event.y))
        self._redraw()

    def _clear(self):
        if self._busy:
            return
        self._points = []
        self._redraw()

    def _set_busy(self, busy: bool):
        self._busy = busy
        try:
            self.configure(cursor="watch" if busy else "")
        except tk.TclError:
            pass
        state = "disabled" if busy else "normal"
        for btn in (self.btn_clear, self.btn_auto, self.btn_ok, self.btn_cancel):
            btn.configure(state=state)
        if busy:
            self.update_idletasks()

    def _auto_detect(self):
        if self._busy:
            return
        if core is None:
            messagebox.showerror("无法分析", "缺少分析核心 core.py，程序不完整。", parent=self)
            return
        self._set_busy(True)
        try:
            calib = core.auto_calibrate(self._video_path, self._shape)
            points = [(float(p[0]), float(p[1])) for p in getattr(calib, "points", [])]
            if not points:
                raise RuntimeError("没有识别到池面")
        except Exception:
            self._set_busy(False)
            messagebox.showwarning("自动识别失败", "自动识别失败，请手动点选。", parent=self)
            return
        self._set_busy(False)
        self._points = points[: self._required]
        self._redraw()
        if len(self._points) < self._required:
            messagebox.showwarning(
                "自动识别不完整",
                "自动识别只得到 %d 个点，需要 %d 个，请手动补充。"
                % (len(self._points), self._required), parent=self)

    def _confirm(self):
        if self._busy:
            return
        if len(self._points) != self._required:
            messagebox.showwarning(
                "标定未完成",
                "需要点选 %d 个点，当前只有 %d 个。" % (self._required, len(self._points)),
                parent=self)
            return
        points = [(int(round(p[0])), int(round(p[1]))) for p in self._points]
        if self._shape == SHAPE_ELLIPSE:
            cx, cy = points[0]
            for px, py in points[1:]:
                if math.hypot(px - cx, py - cy) < 5:
                    messagebox.showwarning(
                        "标定点过近",
                        "中心点、长轴端点、短轴端点距离太近，请重新点选。", parent=self)
                    return
        try:
            self.result = core.Calibration(shape=self._shape, points=points)
        except Exception as exc:
            messagebox.showerror("标定失败", str(exc), parent=self)
            return
        self.destroy()

    # -------------------------------------------------- 绘制
    def _redraw(self):
        self.canvas.delete("mk")
        if self._shape == SHAPE_RECT:
            self._draw_rect_marks()
        else:
            self._draw_ellipse_marks()
        self._update_hint()

    def _draw_rect_marks(self):
        disp = [self._to_display(p[0], p[1]) for p in self._points]
        if len(disp) >= 2:
            coords = []
            for x, y in disp:
                coords.extend([x, y])
            self.canvas.create_line(coords, fill="#ffd740", width=2, dash=(6, 3), tags="mk")
        if len(disp) == 4:
            coords = []
            for x, y in disp:
                coords.extend([x, y])
            self.canvas.create_polygon(coords, outline="#69f0ae", fill="", width=2,
                                       tags="mk")
        self._draw_points(disp)

    def _draw_ellipse_marks(self):
        disp = [self._to_display(p[0], p[1]) for p in self._points]
        if len(disp) >= 1:
            x, y = disp[0]
            self.canvas.create_line(x - 10, y, x + 10, y, fill="#40c4ff", width=2, tags="mk")
            self.canvas.create_line(x, y - 10, x, y + 10, fill="#40c4ff", width=2, tags="mk")
        if len(disp) >= 2:
            self.canvas.create_line(disp[0][0], disp[0][1], disp[1][0], disp[1][1],
                                    fill="#ffd740", width=2, dash=(6, 3), tags="mk")
        if len(disp) >= 3:
            self.canvas.create_line(disp[0][0], disp[0][1], disp[2][0], disp[2][1],
                                    fill="#ffd740", width=2, dash=(6, 3), tags="mk")
            coords = self._ellipse_polygon(disp[0], disp[1], disp[2])
            if coords:
                self.canvas.create_polygon(coords, outline="#69f0ae", fill="",
                                           width=2, tags="mk")
        self._draw_points(disp)

    @staticmethod
    def _ellipse_polygon(center, p_major, p_minor, steps: int = 96):
        cx, cy = center
        v1x, v1y = p_major[0] - cx, p_major[1] - cy
        v2x, v2y = p_minor[0] - cx, p_minor[1] - cy
        a = math.hypot(v1x, v1y)
        b = math.hypot(v2x, v2y)
        angle = math.atan2(v1y, v1x)
        ca, sa = math.cos(angle), math.sin(angle)
        coords = []
        for i in range(steps):
            t = 2.0 * math.pi * i / steps
            x = a * math.cos(t)
            y = b * math.sin(t)
            coords.extend([cx + x * ca - y * sa, cy + x * sa + y * ca])
        return coords

    def _draw_points(self, disp):
        for index, (x, y) in enumerate(disp):
            color = MARK_COLORS[index % len(MARK_COLORS)]
            r = 6
            self.canvas.create_oval(x - r, y - r, x + r, y + r, fill=color,
                                    outline="#ffffff", width=1, tags="mk")
            self.canvas.create_text(x + 12, y - 12, text=str(index + 1), fill=color,
                                    font=("TkDefaultFont", 10, "bold"), tags="mk")

    def _update_hint(self):
        n = len(self._points)
        if self._shape == SHAPE_RECT:
            if n == 0:
                text = "请依次点击泳池的 4 个角（顺序任意）——0/4"
            elif n < 4:
                text = "已点 %d/4 个角点，请继续点击（顺序任意）" % n
            else:
                text = "已点 4/4 个角点，可点“确定”保存标定"
        else:
            if n == 0:
                text = "第 1 步：点击池底中心 —— 0/3"
            elif n == 1:
                text = "第 2 步：点击长轴端点（池子较长的方向）—— 1/3"
            elif n == 2:
                text = "第 3 步：点击短轴端点（与长轴垂直的方向）—— 2/3"
            else:
                text = "已点 3/3，可点“确定”保存标定"
        self.var_hint.set(text)


# ============================================================ 历史对比窗口
class CompareDialog(tk.Toplevel):
    """两条历史记录的覆盖率曲线对比。"""

    def __init__(self, master, records):
        super().__init__(master)
        self.title("历史记录对比 —— 覆盖率曲线")
        self.transient(master)
        self.geometry("940x620")
        self.minsize(780, 500)
        self._records = list(records)[:2]
        self._curves = []
        self._notes = []
        for rec in self._records:
            pts, note = load_curve(str(rec.get("out_dir") or ""), rec.get("coverage") or 0.0)
            self._curves.append(pts)
            self._notes.append(note)
        self._build()
        self.bind("<Escape>", lambda _e: self.destroy())

    def _build(self):
        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        info = ttk.Frame(outer)
        info.pack(fill="x", pady=(0, 8))
        for idx, rec in enumerate(self._records):
            info.columnconfigure(idx, weight=1)
            frame = ttk.LabelFrame(info, padding=8,
                                   text=" 记录 %s " % ("AB"[idx]))
            frame.grid(row=0, column=idx, sticky="nsew", padx=(0, 6 if idx == 0 else 0))
            frames_total = int(rec.get("frames") or 0)
            detected = int(rec.get("detected") or 0)
            hit = (detected / float(frames_total)) if frames_total else 0.0
            rows = (
                ("视频", os.path.basename(str(rec.get("video") or "（未知）"))),
                ("时间", str(rec.get("created") or format_clock(rec.get("mtime")))),
                ("覆盖率", percent_text(rec.get("coverage") or 0.0)),
                ("检测命中率", "%s（%d/%d 帧）" % (percent_text(hit), detected, frames_total)),
            )
            for r, (key, value) in enumerate(rows):
                color = CURVE_COLORS[idx] if key == "覆盖率" else "#222222"
                ttk.Label(frame, text=key + "：").grid(row=r, column=0, sticky="w")
                ttk.Label(frame, text=value, foreground=color).grid(
                    row=r, column=1, sticky="w", padx=(4, 0))
            if self._notes[idx]:
                ttk.Label(frame, text="提示：" + self._notes[idx],
                          foreground="#b26a00").grid(
                    row=len(rows), column=0, columnspan=2, sticky="w", pady=(4, 0))

        self.canvas = tk.Canvas(outer, bg="#ffffff", highlightthickness=1,
                                highlightbackground="#bbbbbb")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _e: self._draw())
        self._draw()

    def _draw(self):
        canvas = self.canvas
        canvas.delete("all")
        width = max(canvas.winfo_width(), 240)
        height = max(canvas.winfo_height(), 160)
        left, top, right, bottom = 64, 26, 24, 46
        plot_w = max(10, width - left - right)
        plot_h = max(10, height - top - bottom)

        max_t = 1.0
        for pts in self._curves:
            if pts:
                max_t = max(max_t, pts[-1][0])
        max_t = max(max_t, 1.0)

        # 网格 + 纵轴（覆盖率 %）
        for i in range(5):
            y = top + plot_h * i / 4.0
            canvas.create_line(left, y, left + plot_w, y, fill="#e0e0e0")
            canvas.create_text(left - 8, y, anchor="e", text="%d%%" % int(round(100 - 25 * i)),
                               fill="#555555", font=("TkDefaultFont", 9))
        # 横轴（视频秒）
        for i in range(5):
            x = left + plot_w * i / 4.0
            canvas.create_line(x, top, x, top + plot_h, fill="#f0f0f0")
            label = "%.1f" % (max_t * i / 4.0)
            canvas.create_text(x, top + plot_h + 12, text=label, fill="#555555",
                               font=("TkDefaultFont", 9))
        canvas.create_line(left, top, left, top + plot_h, fill="#888888")
        canvas.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="#888888")
        canvas.create_text(left + plot_w / 2.0, top + plot_h + 32,
                           text="视频时间（秒）", fill="#333333")
        canvas.create_text(left - 8, top - 12, anchor="e", text="覆盖率",
                           fill="#333333", font=("TkDefaultFont", 9, "bold"))

        # 两条曲线
        for idx, pts in enumerate(self._curves):
            if not pts:
                continue
            if len(pts) == 1:
                pts = [(0.0, pts[0][1]), (max_t, pts[0][1])]
            coords = []
            for t, c in pts:
                x = left + plot_w * (min(t, max_t) / max_t)
                y = top + plot_h * (1.0 - c)
                coords.extend([x, y])
            if len(coords) >= 4:
                canvas.create_line(coords, fill=CURVE_COLORS[idx], width=2)
            last_t, last_c = pts[-1]
            if last_t <= max_t:
                x = left + plot_w * (last_t / max_t)
                y = top + plot_h * (1.0 - last_c)
                canvas.create_oval(x - 3, y - 3, x + 3, y + 3,
                                   fill=CURVE_COLORS[idx], outline="")

        # 图例
        legend_x = left + 14
        for idx, rec in enumerate(self._records):
            label = "记录 %s：%s" % (
                "AB"[idx],
                os.path.basename(str(rec.get("video") or "（未知）")))
            canvas.create_line(legend_x, top + 14, legend_x + 24, top + 14,
                               fill=CURVE_COLORS[idx], width=3)
            canvas.create_text(legend_x + 30, top + 14, anchor="w", text=label,
                               fill=CURVE_COLORS[idx], font=("TkDefaultFont", 9))
            legend_x += 30 + max(90, 7 * len(label)) + 24
            if legend_x > width - 160:
                break


# ============================================================ 主窗口
class PoolCoverageApp:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title(APP_TITLE)
        self.root.minsize(MIN_WINDOW_WIDTH, MIN_WINDOW_HEIGHT)
        self.root.geometry("%dx%d" % (1040, 780))
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        self.reports_root = default_reports_root()
        self.video_path = ""
        self._calib = None
        self._result = None
        self._analyzing = False
        self._poll_active = False
        self._poll_id = None
        self._queue = queue.Queue()
        self._cancel_event = threading.Event()
        self._worker = None
        self._history_records = []
        self._history_by_iid = {}

        self._setup_fonts()
        self._build_ui()
        self._refresh_history()

    # -------------------------------------------------- 外观
    def _setup_fonts(self):
        preferred = ("Microsoft YaHei UI", "Microsoft YaHei", "PingFang SC",
                     "Hiragino Sans GB", "Noto Sans CJK SC", "Source Han Sans SC",
                     "WenQuanYi Micro Hei", "SimHei", "Arial Unicode MS")
        try:
            available = set(tkfont.families(self.root))
        except tk.TclError:
            available = set()
        family = next((name for name in preferred if name in available), "")
        if family:
            for name in ("TkDefaultFont", "TkTextFont", "TkMenuFont", "TkHeadingFont",
                         "TkCaptionFont", "TkTooltipFont"):
                try:
                    tkfont.nametofont(name).configure(family=family, size=10)
                except tk.TclError:
                    pass
            style = ttk.Style(self.root)
            style.configure("Value.TLabel", font=(family, 13, "bold"), foreground="#0b57d0")
            style.configure("Hint.TLabel", font=(family, 9), foreground="#666666")
            style.configure("Section.TLabel", font=(family, 10, "bold"))

    # -------------------------------------------------- 界面
    def _build_ui(self):
        main = ttk.Frame(self.root, padding=12)
        main.pack(fill="both", expand=True)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(5, weight=1)

        row = 0
        # ---- 1. 视频
        video_box = ttk.LabelFrame(main, text=" 1. 选择视频 ", padding=8)
        video_box.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        video_box.columnconfigure(1, weight=1)
        ttk.Label(video_box, text="视频文件：").grid(row=0, column=0, sticky="w")
        self.var_video = tk.StringVar(value="")
        ttk.Entry(video_box, textvariable=self.var_video, state="readonly").grid(
            row=0, column=1, sticky="ew", padx=8)
        self.btn_browse = ttk.Button(video_box, text="浏览…", width=10,
                                     command=self._choose_video)
        self.btn_browse.grid(row=0, column=2, sticky="e")
        self.var_video_info = tk.StringVar(value="尚未选择视频。支持 mp4 / avi / mov / mkv 等常见格式。")
        ttk.Label(video_box, textvariable=self.var_video_info, style="Hint.TLabel").grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        row += 1

        # ---- 2. 池型与标定
        pool_box = ttk.LabelFrame(main, text=" 2. 选择池型并标定 ", padding=8)
        pool_box.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        pool_box.columnconfigure(3, weight=1)
        ttk.Label(pool_box, text="池型：").grid(row=0, column=0, sticky="w")
        shape_frame = ttk.Frame(pool_box)
        shape_frame.grid(row=0, column=1, columnspan=2, sticky="w", padx=(0, 12))
        self.var_shape = tk.StringVar(value=SHAPE_RECT)
        self.rb_rect = ttk.Radiobutton(shape_frame, text="矩形", value=SHAPE_RECT,
                                       variable=self.var_shape,
                                       command=self._on_shape_changed)
        self.rb_ellipse = ttk.Radiobutton(shape_frame, text="圆形（椭圆）",
                                          value=SHAPE_ELLIPSE,
                                          variable=self.var_shape,
                                          command=self._on_shape_changed)
        self.rb_rect.grid(row=0, column=0, padx=(0, 14))
        self.rb_ellipse.grid(row=0, column=1)
        self.btn_calibrate = ttk.Button(pool_box, text="打开画面标定…", width=16,
                                        command=self._open_calibration)
        self.btn_calibrate.grid(row=0, column=4, sticky="e")
        self.var_calib_status = tk.StringVar(value="未标定")
        self.lbl_calib = ttk.Label(pool_box, textvariable=self.var_calib_status,
                                   foreground="#b26a00")
        self.lbl_calib.grid(row=1, column=0, columnspan=5, sticky="w", pady=(6, 0))
        ttk.Label(pool_box, text="矩形池：依次点 4 个池角；圆形池：依次点 池中心 → 长轴端点 → 短轴端点。",
                  style="Hint.TLabel").grid(row=2, column=0, columnspan=5, sticky="w")
        row += 1

        # ---- 3. 参数
        param_box = ttk.LabelFrame(main, text=" 3. 分析参数 ", padding=8)
        param_box.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        param_box.columnconfigure(5, weight=1)
        ttk.Label(param_box, text="目标类型：").grid(row=0, column=0, sticky="w")
        self.var_target = tk.StringVar(value=TARGET_LABELS[0])
        self.combo_target = ttk.Combobox(param_box, textvariable=self.var_target,
                                         values=TARGET_LABELS, state="readonly", width=13)
        self.combo_target.grid(row=0, column=1, sticky="w", padx=(0, 18))
        ttk.Label(param_box, text="灵敏度：").grid(row=0, column=2, sticky="w")
        self.var_sensitivity = tk.DoubleVar(value=0.5)
        self.scale_sens = ttk.Scale(param_box, from_=0.0, to=1.0, length=200,
                                    variable=self.var_sensitivity,
                                    command=self._on_sensitivity)
        self.scale_sens.grid(row=0, column=3, sticky="w")
        self.var_sens_text = tk.StringVar(value="0.50")
        ttk.Label(param_box, textvariable=self.var_sens_text, width=5).grid(
            row=0, column=4, sticky="w", padx=(6, 12))
        ttk.Label(param_box, text="0 = 保守（少误检）　1 = 灵敏（暗光/小机器人）",
                  style="Hint.TLabel").grid(row=1, column=0, columnspan=6, sticky="w",
                                            pady=(4, 0))
        row += 1

        # ---- 4. 开始/取消 + 进度
        run_box = ttk.Frame(main)
        run_box.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        run_box.columnconfigure(2, weight=1)
        self.btn_start = ttk.Button(run_box, text="开始分析", width=12,
                                    command=self._start_analysis)
        self.btn_start.grid(row=0, column=0, sticky="w")
        self.btn_cancel = ttk.Button(run_box, text="取消", width=8, state="disabled",
                                     command=self._cancel_analysis)
        self.btn_cancel.grid(row=0, column=1, sticky="w", padx=(8, 12))
        self.progress = ttk.Progressbar(run_box, mode="determinate", maximum=100.0)
        self.progress.grid(row=0, column=2, sticky="ew")
        self.var_status = tk.StringVar(value="就绪：请选择视频并完成标定。")
        ttk.Label(run_box, textvariable=self.var_status).grid(
            row=1, column=0, columnspan=3, sticky="w", pady=(6, 0))
        row += 1

        # ---- 5. 结果
        result_box = ttk.LabelFrame(main, text=" 4. 分析结果 ", padding=8)
        result_box.grid(row=row, column=0, sticky="ew", pady=(0, 8))
        for col in range(4):
            result_box.columnconfigure(col * 2 + 1, weight=1)
        self.var_cov = tk.StringVar(value="—")
        self.var_uncovered = tk.StringVar(value="—")
        self.var_passes = tk.StringVar(value="—")
        self.var_hit = tk.StringVar(value="—")
        metrics = (
            ("覆盖率", self.var_cov),
            ("漏洗面积", self.var_uncovered),
            ("平均经过次数", self.var_passes),
            ("检测命中率", self.var_hit),
        )
        for index, (title, var) in enumerate(metrics):
            ttk.Label(result_box, text=title + "：").grid(
                row=0, column=index * 2, sticky="e")
            ttk.Label(result_box, textvariable=var, style="Value.TLabel").grid(
                row=0, column=index * 2 + 1, sticky="w", padx=(2, 18))

        btn_frame = ttk.Frame(result_box)
        btn_frame.grid(row=1, column=0, columnspan=8, sticky="w", pady=(8, 0))
        self.btn_open_dir = ttk.Button(btn_frame, text="打开输出文件夹", width=14,
                                       state="disabled",
                                       command=lambda: self._open_result("out_dir",
                                                                         "输出文件夹"))
        self.btn_play = ttk.Button(btn_frame, text="播放轨迹视频", width=14,
                                   state="disabled",
                                   command=lambda: self._open_result("overlay_path",
                                                                     "轨迹视频"))
        self.btn_heatmap = ttk.Button(btn_frame, text="查看热力图", width=12,
                                      state="disabled",
                                      command=lambda: self._open_result("heatmap_path",
                                                                        "热力图"))
        self.btn_summary = ttk.Button(btn_frame, text="打开 summary.json", width=17,
                                      state="disabled",
                                      command=lambda: self._open_result("summary_json",
                                                                        "summary.json"))
        for index, btn in enumerate((self.btn_open_dir, self.btn_play,
                                     self.btn_heatmap, self.btn_summary)):
            btn.grid(row=0, column=index, padx=(0, 8))

        self.var_result_note = tk.StringVar(value="输出目录：%s" % self.reports_root)
        ttk.Label(result_box, textvariable=self.var_result_note, style="Hint.TLabel").grid(
            row=2, column=0, columnspan=8, sticky="w", pady=(6, 0))
        row += 1

        # ---- 6. 历史记录
        history_box = ttk.LabelFrame(main, text=" 5. 历史记录（按住 Ctrl 选两条 → 点“对比”） ",
                                     padding=8)
        history_box.grid(row=row, column=0, sticky="nsew")
        history_box.columnconfigure(0, weight=1)
        history_box.rowconfigure(1, weight=1)
        toolbar = ttk.Frame(history_box)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 6))
        toolbar.columnconfigure(0, weight=1)
        self.var_history_hint = tk.StringVar(value="共 0 条历史记录")
        ttk.Label(toolbar, textvariable=self.var_history_hint,
                  style="Hint.TLabel").grid(row=0, column=0, sticky="w")
        self.btn_refresh = ttk.Button(toolbar, text="刷新", width=8,
                                      command=self._refresh_history)
        self.btn_refresh.grid(row=0, column=1, padx=(0, 8))
        self.btn_compare = ttk.Button(toolbar, text="对比", width=8, state="disabled",
                                      command=self._compare_selected)
        self.btn_compare.grid(row=0, column=2)

        columns = ("time", "video", "coverage", "hit")
        self.tree = ttk.Treeview(history_box, columns=columns, show="headings",
                                 selectmode="extended", height=5)
        self.tree.heading("time", text="时间")
        self.tree.heading("video", text="视频")
        self.tree.heading("coverage", text="覆盖率")
        self.tree.heading("hit", text="检测命中率")
        self.tree.column("time", width=140, anchor="center", stretch=False)
        self.tree.column("video", width=520, anchor="w", stretch=True)
        self.tree.column("coverage", width=90, anchor="center", stretch=False)
        self.tree.column("hit", width=150, anchor="center", stretch=False)
        scrollbar = ttk.Scrollbar(history_box, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.grid(row=1, column=0, sticky="nsew")
        scrollbar.grid(row=1, column=1, sticky="ns")
        self.tree.bind("<<TreeviewSelect>>", self._on_history_selected)
        row += 1

        # ---- 底部状态
        ttk.Label(main, text="输出目录：%s" % self.reports_root,
                  style="Hint.TLabel").grid(row=row, column=0, sticky="w", pady=(6, 0))

    # -------------------------------------------------- 小回调
    def _on_sensitivity(self, _value=None):
        try:
            self.var_sens_text.set("%.2f" % float(self.var_sensitivity.get()))
        except (tk.TclError, ValueError):
            pass

    def _set_status(self, text: str):
        self.var_status.set(text)

    def _set_calibrated(self, calib):
        self._calib = calib
        shape = getattr(calib, "shape", "")
        count = len(getattr(calib, "points", []) or [])
        self.var_calib_status.set("已标定（%s，%d 个点）" % (SHAPE_LABELS.get(shape, shape),
                                                         count))
        self.lbl_calib.configure(foreground="#2e7d32")

    def _clear_calibration(self, reason: str = "未标定"):
        self._calib = None
        self.var_calib_status.set(reason)
        self.lbl_calib.configure(foreground="#b26a00")

    # -------------------------------------------------- 选择视频
    def _choose_video(self):
        path = filedialog.askopenfilename(title="选择泳池监控视频",
                                          filetypes=VIDEO_TYPES)
        if not path:
            return
        self.video_path = path
        self.var_video.set(path)
        self._clear_calibration()
        self._update_video_info(path)
        self._set_status("已选择视频，请完成池面标定。")

    def _update_video_info(self, path: str):
        if core is None:
            self.var_video_info.set("缺少 core.py，无法读取视频信息。")
            return
        try:
            info = core.video_info(path)
            fps = float(info.get("fps") or 0.0)
            frames = int(info.get("frames") or 0)
            width = int(info.get("width") or 0)
            height = int(info.get("height") or 0)
            duration = format_duration(frames / fps) if fps > 0 else "未知"
            self.var_video_info.set("分辨率 %d×%d　·　%.2f fps　·　共 %d 帧　·　时长 %s"
                                    % (width, height, fps, frames, duration))
        except Exception as exc:
            self.var_video_info.set("已选择视频（读取视频信息失败：%s）" % exc)

    # -------------------------------------------------- 池型
    def _on_shape_changed(self):
        if self._calib is not None:
            self._clear_calibration("未标定（池型已更改，请重新标定）")

    def _current_shape(self) -> str:
        value = self.var_shape.get()
        return value if value in (SHAPE_RECT, SHAPE_ELLIPSE) else SHAPE_RECT

    # -------------------------------------------------- 标定
    def _open_calibration(self):
        if self._analyzing:
            return
        if not self.video_path or not os.path.isfile(self.video_path):
            messagebox.showinfo("请先选择视频", "请先在“1. 选择视频”里选择一个视频文件。")
            return
        if core is None:
            messagebox.showerror("无法标定", "缺少分析核心 core.py，程序不完整。")
            return
        try:
            frame = core.probe_frame(self.video_path)
        except Exception as exc:
            messagebox.showerror("读取画面失败",
                                 "无法读取视频首帧：\n%s" % exc)
            return
        shape = self._current_shape()
        try:
            dialog = CalibrationDialog(self.root, self.video_path, shape, frame,
                                       initial=self._calib)
        except Exception as exc:
            messagebox.showerror("打开标定失败", "无法显示标定画面：\n%s" % exc)
            return
        self.root.wait_window(dialog)
        if dialog.result is not None:
            self._set_calibrated(dialog.result)
            self._set_status("标定完成，可以点击“开始分析”。")

    # -------------------------------------------------- 分析
    def _start_analysis(self):
        if self._analyzing:
            return
        if core is None:
            messagebox.showerror("无法分析", "缺少分析核心 core.py，程序不完整。")
            return
        video = self.video_path
        if not video:
            messagebox.showinfo("请先选择视频", "请先选择要分析的视频文件。")
            return
        if not os.path.isfile(video):
            messagebox.showerror("视频不存在", "视频文件不存在：\n%s" % video)
            return
        if self._calib is None:
            messagebox.showinfo("请先标定", "请先点击“打开画面标定…”完成池面标定。")
            return
        if getattr(self._calib, "shape", None) != self._current_shape():
            self._clear_calibration("未标定（池型已更改，请重新标定）")
            messagebox.showinfo("请重新标定", "池型已更改，请重新完成池面标定。")
            return
        target = TARGET_MAP.get(self.var_target.get(), "auto")
        try:
            sensitivity = min(1.0, max(0.0, float(self.var_sensitivity.get())))
        except (tk.TclError, ValueError):
            sensitivity = 0.5
        try:
            out_dir = self._make_out_dir(video)
        except OSError as exc:
            messagebox.showerror("无法创建输出目录",
                                 "无法创建输出目录：\n%s\n\n%s"
                                 % (self.reports_root, exc))
            return

        # ---- 重置界面状态
        self._result = None
        self.var_cov.set("—")
        self.var_uncovered.set("—")
        self.var_passes.set("—")
        self.var_hit.set("—")
        self.var_result_note.set("输出目录：%s" % out_dir)
        for btn in (self.btn_open_dir, self.btn_play, self.btn_heatmap, self.btn_summary):
            btn.configure(state="disabled")
        self.progress.configure(value=0.0)
        self.var_status.set("准备分析…")
        self._set_analyzing(True)

        self._queue = queue.Queue()
        self._cancel_event.clear()
        self._worker = threading.Thread(
            target=self._worker_run,
            args=(video, self._calib, out_dir, target, sensitivity),
            name="PoolCoverageAnalyze",
            daemon=True,
        )
        self._worker.start()

        self._poll_active = True
        self._schedule_poll()

    def _make_out_dir(self, video: str) -> str:
        os.makedirs(self.reports_root, exist_ok=True)
        stem = os.path.splitext(os.path.basename(video))[0]
        stem = re.sub(r'[\\/:*?"<>|\s]+', "_", stem).strip("._")[:40] or "video"
        stamp = time.strftime("%Y%m%d-%H%M%S")
        base = os.path.join(self.reports_root, "%s_%s" % (stamp, stem))
        out_dir = base
        index = 2
        while os.path.isdir(out_dir):
            out_dir = "%s_%d" % (base, index)
            index += 1
        os.makedirs(out_dir, exist_ok=True)
        return out_dir

    def _worker_run(self, video, calib, out_dir, target, sensitivity):
        """后台线程：只调用 core，不碰任何 Tk 控件，全部结果走 queue。"""
        try:
            result = core.analyze(
                video, calib, out_dir,
                target=target,
                sensitivity=sensitivity,
                progress=lambda frac, msg: self._queue.put(("progress", frac, msg)),
                cancel=self._cancel_event.is_set,
            )
            self._queue.put(("result", result))
        except BaseException as exc:  # noqa: BLE001 - 线程里必须兜住一切
            cancel_cls = getattr(core, "CancelledError", None)
            if cancel_cls is not None and isinstance(exc, cancel_cls):
                self._queue.put(("cancelled", str(exc)))
            else:
                self._queue.put(("error", str(exc) or type(exc).__name__,
                                 traceback.format_exc()))
        finally:
            self._queue.put(("finished", None))

    def _schedule_poll(self):
        try:
            self._poll_id = self.root.after(120, self._poll_queue)
        except tk.TclError:
            self._poll_active = False

    def _poll_queue(self):
        self._poll_id = None
        try:
            while True:
                message = self._queue.get_nowait()
                try:
                    done = self._handle_message(message)
                except tk.TclError:
                    raise
                except Exception:
                    # 单个回调出错不应卡死界面：记录后继续处理后续消息
                    sys.stderr.write(traceback.format_exc() + "\n")
                    done = False
                if done:
                    return
        except queue.Empty:
            pass
        except tk.TclError:
            self._poll_active = False
            return
        if self._poll_active:
            self._schedule_poll()

    def _handle_message(self, message) -> bool:
        """处理一条来自分析线程的消息；返回 True 表示线程已结束。"""
        kind = message[0]
        if kind == "progress":
            self._on_progress(message[1], message[2])
        elif kind == "result":
            self._on_result(message[1])
        elif kind == "error":
            self._on_error(message[1], message[2])
        elif kind == "cancelled":
            self._on_cancelled()
        elif kind == "finished":
            self._on_finished()
            return True
        return False

    def _on_progress(self, fraction, message):
        try:
            value = max(0.0, min(1.0, float(fraction))) * 100.0
        except (TypeError, ValueError):
            value = 0.0
        self.progress.configure(value=value)
        self._set_status("%s（%d%%）" % (message, int(round(value))))

    def _on_result(self, result):
        self._result = result
        coverage = float(getattr(result, "coverage", 0.0) or 0.0)
        uncovered = float(getattr(result, "uncovered", max(0.0, 1.0 - coverage)) or 0.0)
        frames = int(getattr(result, "frames", 0) or 0)
        detected = int(getattr(result, "detected", 0) or 0)
        hit = (detected / float(frames)) if frames > 0 else 0.0
        self.var_cov.set(percent_text(coverage))
        self.var_uncovered.set(percent_text(uncovered))
        try:
            passes = float(getattr(result, "avg_passes", 0.0) or 0.0)
        except (TypeError, ValueError):
            passes = 0.0
        self.var_passes.set("%.2f 次" % passes)
        self.var_hit.set("%s（%d/%d 帧）" % (percent_text(hit), detected, frames))

        notes = ["输出目录：%s" % str(getattr(result, "out_dir", "") or "")]
        warnings = list(getattr(result, "warnings", []) or [])
        if warnings:
            notes.append("提示：" + "；".join(str(w) for w in warnings))
        self.var_result_note.set("\n".join(notes))
        for btn in (self.btn_open_dir, self.btn_play, self.btn_heatmap, self.btn_summary):
            btn.configure(state="normal")
        self.progress.configure(value=100.0)
        self._set_status("分析完成：覆盖率 %s，漏洗 %s。"
                         % (percent_text(coverage), percent_text(uncovered)))
        self._refresh_history()

    def _on_error(self, message, traceback_text=""):
        self._set_status("分析失败：%s" % message)
        if traceback_text:
            try:
                sys.stderr.write(traceback_text + "\n")
            except Exception:
                pass
        messagebox.showerror("分析失败", str(message))

    def _on_cancelled(self):
        self._set_status("已取消分析。")
        self.progress.configure(value=0.0)
        messagebox.showinfo("已取消", "分析已取消。")

    def _on_finished(self):
        self._poll_active = False
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        self._set_analyzing(False)

    def _set_analyzing(self, analyzing: bool):
        self._analyzing = analyzing
        state = "disabled" if analyzing else "normal"
        for widget in (self.btn_browse, self.rb_rect, self.rb_ellipse,
                       self.btn_calibrate, self.btn_start):
            widget.configure(state=state)
        self.combo_target.configure(state="disabled" if analyzing else "readonly")
        self.scale_sens.configure(state="disabled" if analyzing else "normal")
        self.btn_cancel.configure(state="normal" if analyzing else "disabled")

    def _cancel_analysis(self):
        if not self._analyzing:
            return
        self._cancel_event.set()
        self.btn_cancel.configure(state="disabled")
        self._set_status("正在取消…（等待当前步骤结束）")

    # -------------------------------------------------- 结果按钮
    def _open_result(self, attribute: str, label: str):
        if self._result is None:
            return
        path = str(getattr(self._result, attribute, "") or "")
        if not path:
            messagebox.showinfo("暂无文件", "本次结果没有生成%s。" % label)
            return
        try:
            open_path(path)
        except Exception as exc:
            messagebox.showerror("打开失败", "无法打开%s：\n%s" % (label, exc))

    # -------------------------------------------------- 历史
    def _refresh_history(self):
        records = []
        if os.path.isdir(self.reports_root) and core is not None:
            try:
                records = list(core.list_history(self.reports_root))
            except Exception as exc:
                self.var_history_hint.set("读取历史记录失败：%s" % exc)
                records = []
        self._history_records = [r for r in records if isinstance(r, dict)]
        self._history_by_iid = {}
        try:
            children = self.tree.get_children()
            if children:
                self.tree.delete(*children)
        except tk.TclError:
            return
        for record in self._history_records:
            frames = int(record.get("frames") or 0)
            detected = int(record.get("detected") or 0)
            hit = (detected / float(frames)) if frames else 0.0
            video_name = os.path.basename(str(record.get("video") or "（未知）"))
            try:
                iid = self.tree.insert(
                    "", "end",
                    values=(format_clock(record.get("mtime")), video_name,
                            percent_text(record.get("coverage") or 0.0),
                            "%s（%d/%d）" % (percent_text(hit), detected, frames)))
            except tk.TclError:
                break
            self._history_by_iid[iid] = record
        self.var_history_hint.set("共 %d 条历史记录（按住 Ctrl 点选两条后“对比”）"
                                  % len(self._history_records))
        self.btn_compare.configure(state="disabled")

    def _on_history_selected(self, _event=None):
        selected = self.tree.selection()
        self.btn_compare.configure(state="normal" if len(selected) == 2 else "disabled")

    def _compare_selected(self):
        selected = [iid for iid in self.tree.selection() if iid in self._history_by_iid]
        if len(selected) != 2:
            messagebox.showinfo("请选择两条记录",
                                "请按住 Ctrl 键，在历史记录里点选恰好两条记录再点“对比”。")
            return
        records = [self._history_by_iid[iid] for iid in selected]
        try:
            CompareDialog(self.root, records)
        except Exception as exc:
            messagebox.showerror("对比失败", "无法打开对比窗口：\n%s" % exc)

    # -------------------------------------------------- 关闭
    def _on_close(self):
        if self._analyzing:
            if not messagebox.askyesno("确认退出", "分析仍在进行，退出会中止分析。确定退出吗？"):
                return
            self._cancel_event.set()
        self._poll_active = False
        if self._poll_id is not None:
            try:
                self.root.after_cancel(self._poll_id)
            except tk.TclError:
                pass
            self._poll_id = None
        try:
            self.root.destroy()
        except tk.TclError:
            pass


# ============================================================ 入口
def main() -> int:
    if sys.platform.startswith("win"):
        try:
            import ctypes

            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            pass

    root = tk.Tk()
    root.withdraw()
    if core is None:
        messagebox.showerror(
            APP_TITLE,
            "缺少分析核心 core.py（或它依赖的库未安装），程序无法启动。\n"
            "请确认 core.py 与本程序在同一目录，并已安装 requirements.txt 中的依赖。")
        try:
            root.destroy()
        except tk.TclError:
            pass
        return 1
    try:
        PoolCoverageApp(root)
    except Exception:
        messagebox.showerror(APP_TITLE, "程序启动失败：\n" + traceback.format_exc())
        try:
            root.destroy()
        except tk.TclError:
            pass
        return 1
    root.deiconify()
    root.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
