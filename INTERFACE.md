# 项目约定（Lead 与 GUI 开发的接口契约）

工程目录：/root/workspace/pool-coverage/
```
core.py                 # 分析核心（Lead 负责，勿改）
app.py                  # Tkinter 桌面程序（GUI 负责）
requirements.txt
.github/workflows/build-windows.yml
README.md
INTERFACE.md
```

## core.py 对外 API（必须严格按此调用）

```python
VERSION = "0.1.0"
SHAPE_RECT = "rect"
SHAPE_ELLIPSE = "ellipse"

class CancelledError(Exception): ...

@dataclass
class Calibration:
    shape: str                 # "rect" | "ellipse"
    points: list               # rect: [(x,y)]*4 任意顺序; ellipse: [中心, 长轴端点, 短轴端点]

@dataclass
class AnalyzeResult:
    video: str
    out_dir: str
    frames: int
    fps: float
    width: int
    height: int
    detected: int              # 检测到机器人的帧数
    coverage: float            # 0..1
    uncovered: float           # 0..1
    avg_passes: float          # 覆盖区平均经过次数
    pool_area_px: int
    path_len_px: float
    overlay_path: str
    heatmap_path: str
    trajectory_csv: str
    summary_json: str
    curve: list                # [(视频秒, 覆盖率0..1), ...]
    warnings: list             # 字符串列表

def video_info(path) -> dict          # {"fps":float,"frames":int,"width":int,"height":int}
def probe_frame(path, frame_index=0) -> numpy.ndarray  # BGR 全分辨率
def auto_calibrate(path, shape=SHAPE_RECT) -> Calibration   # 失败抛 RuntimeError
def analyze(video_path, calib, out_dir, *, target="auto", sensitivity=0.5,
            progress=None, cancel=None, overlay_width=1280) -> AnalyzeResult
def list_history(root_dir) -> list[dict]
    # 扫描 root_dir 下的历史输出目录，返回 [{"out_dir","video","mtime","coverage","detected","frames"}...]，按时间倒序
```

约定：
- `progress(fraction: float, message: str)` 在分析线程里被调用，fraction 0..1
- `cancel()` 返回 True 时 analyze 抛 `CancelledError`
- `target` ∈ {"auto","dark","bright"}
- `sensitivity` 0..1（0=保守，1=灵敏）
- 异常：普通失败抛 `RuntimeError`，附中文说明
- 所有输出文件写到 `out_dir`（由 GUI 创建）

## GUI（app.py）要求
- 纯 tkinter/ttk，无第三方 GUI 依赖；中文界面；Windows 用微软雅黑。
- 窗口标题 "泳池机器人覆盖率分析 v0.1"，最小 900x680，可缩放。
- 顶部：视频文件选择（浏览…）→ 路径显示
- 池型：单选 矩形 / 圆形（椭圆）
- 标定按钮 "打开画面标定…"：弹出 Toplevel，Canvas 显示 core.probe_frame 缩放图
    - 矩形：依次点 4 个池角（顺序任意），显示 n/4；按钮：清除 / 自动识别 / 确定
    - 圆形：依次点 池中心 → 长轴端点 → 短轴端点，显示 n/3；同样的按钮
    - 自动识别调用 core.auto_calibrate，失败 MessageBox 提示"自动识别失败，请手动点选"
    - 确定后生成 core.Calibration，主窗口显示"已标定"
- 参数区：目标类型 Combobox（自动/暗色机器人/亮色机器人）、灵敏度 Scale(0..1, 默认0.5)
- "开始分析"按钮：禁用输入，进度条 + 状态文字，可取消；分析在 threading.Thread 里跑，用 queue 回传进度/结果/异常
- 结果区：覆盖率、漏洗面积、平均经过次数、检测命中率（detected/frames）
- 结果按钮：打开输出文件夹（os.startfile）、播放轨迹视频（os.startfile overlay_path）、查看热力图、打开 summary.json
- 历史对比：底部列出 list_history(输出根目录) 最近记录；选两条 → "对比" 弹出窗口：Canvas 画两条覆盖率曲线（视频秒为横轴）+ 两侧指标
- 默认输出根目录：<exe 所在目录>/PoolCoverageReports（冻结时用 sys.executable 所在目录，源码运行时用工程目录）
- 不得在 import 时创建窗口或启动线程；入口 `if __name__ == "__main__": main()`
- 分析失败/取消要恢复 UI 状态并提示。

## 构建与 CI
- requirements.txt: opencv-python-headless、numpy、pyinstaller（CI 里装）
- workflow: .github/workflows/build-windows.yml，windows-latest，python 3.12，pip 安装，下载 ffmpeg.exe（BtbN win64 gpl 或 gyan essentials），
  `pyinstaller --noconfirm --onedir --windowed --name PoolCoverage --add-binary "ffmpeg.exe;." app.py`
  然后 Compress-Archive 打包 dist/PoolCoverage 为 zip 并 upload-artifact。
- README.md：给小白的分步说明：创建 GitHub 仓库 → 上传文件 → Actions 运行 → 下载 artifact → 解压 → 双击 PoolCoverage.exe；使用步骤（选视频/池型/标定/分析/看结果）；常见问题。

## 测试要求（GUI 侧）
- `python -m py_compile app.py` 通过
- `python -c "import app"` 不弹窗、不报错（无显示器环境也要能 import）
- 不得引用 core 中不存在的符号

## 阶段2 新增 core API（手绘轮廓 + 机位模板）

```python
SHAPE_POLY = "poly"           # 手绘/自定义闭合轮廓，points 为边界顺序点列

core.smooth_contour(points, strength=1.0) -> [[x,y], ...]
    # 输入任意轮廓点（>=3，边界顺序），返回稠密平滑闭合轮廓；
    # strength 0=只重采样, 1=标准, 2=更强（更圆润、略收缩）

core.snap_contour(frame, points, band_px=None) -> [[x,y], ...]
    # 可选的边缘吸附（沿法线找梯度最大处，带阻尼）；失败返回原轮廓

core.save_template(templates_dir, name, calib, source_video="") -> path
core.list_templates(templates_dir) -> [{"name","path","shape","points","source_video","created","mtime"}]
core.load_template(path) -> Calibration

core.auto_calibrate(video, "poly") -> Calibration("poly", 初始平滑轮廓)
```

GUI 侧要求：主窗口池型三选一（矩形/圆形/手绘自定义）；标定窗口 poly 模式为鼠标拖动描边 + 平滑预览（平滑强度、吸附开关默认关、≤12 个可拖动手柄）；机位模板区（Combobox + 套用/保存/删除，目录 <exe目录>/PoolCoverageTemplates）。

## 阶段3 新增（真实时间 + 新输出）

```python
core.analyze(video, calib, out_dir, *, target="auto", sensitivity=0.5,
             progress=None, cancel=None, overlay_width=1280,
             start_time=None,      # "HH:MM:SS"，可为 None/""；传入则曲线横轴为真实时钟时间
             speed_factor=1.0)     # 加速倍数；真实时间 = 起始时间 + 视频秒 × 倍数
```

AnalyzeResult 新增字段（可能为空字符串）：
- `coverage_path`    覆盖图 PNG（绿=已清洗/白=未清洗）
- `curve_path`       覆盖率-真实时间曲线图 PNG
- `trajectory_image_path` 完整轨迹图 PNG（含起点/终点/真实时间）

输出文件变化：
- `heatmap.png`：白=未清洗，蓝/黄/红/黑=1/2/3/4 次以上
- `coverage_map.png`：绿=已清洗，白=未清洗
- `coverage_curve.png`：覆盖率随时间变化（真实时间轴）
- `trajectory.png`：完整轨迹
- `overlay.mp4`：轨迹加粗、起点标记、结尾定格 2.5 秒展示完整轨迹（时长 +2.5s）

## 阶段4（v0.5.0）变更

```python
core.analyze(video, calib, out_dir, *, target="auto", sensitivity=0.5,
             progress=None, cancel=None,
             overlay_width=1280,   # 轨迹视频宽度：1280/960/640/480
             make_video=True,      # False 时不生成轨迹视频（提速），overlay_path 返回 ""
             start_time=None, speed_factor=1.0)
```

- 不再生成 `summary.json`；`AnalyzeResult.summary_json` 恒为 `""`。
- `AnalyzeResult` 新增 `shape`（"rect"/"ellipse"/"poly"）。
- 覆盖图与热力图左上角自带「覆盖率 xx.x%」标签。
- 新增历史索引：
  - `core.record_history(root_dir, result) -> path`：追加到 `<root_dir>/_history.jsonl`
  - `core.list_history(root_dir)`：优先读该索引，旧目录回退扫 summary.json；
    返回记录含 `curve`（约 200 点）、`avg_passes`、`overlay_path`、`coverage_path`、
    `heatmap_path`、`curve_path`、`trajectory_image_path`、`created`、`shape` 等。
- 输出目录完全由 GUI 决定（core 只往传入的 out_dir 写文件）。
