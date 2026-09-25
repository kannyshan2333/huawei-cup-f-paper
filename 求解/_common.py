# 公共代码模板：求解/_common.py（系统自动播种，每个任务只写一次，不要覆写覆盖函数）
import sys, os
try:
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')
except Exception:
    pass
# 问题目录 = 运行脚本所在目录（兼容 cd 求解/问题X 与 求解/ 根运行两种方式）
_script_dir = os.path.dirname(os.path.abspath(sys.argv[0]))
# 在 Jupyter / ipykernel 下 sys.argv[0] 指向内核启动脚本，须改用工作目录（notebook 所在目录）
if 'ipykernel' in sys.modules or 'ipykernel_launcher' in os.path.basename(sys.argv[0]):
    _script_dir = os.getcwd()
if not _script_dir or not os.path.isdir(_script_dir):
    _script_dir = os.getcwd()
BASE_DIR = _script_dir

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm
import warnings, logging
warnings.filterwarnings('ignore')
logging.getLogger('matplotlib').setLevel(logging.ERROR)  # 抑制findfont警告
logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)  # 抑制字体管理器警告

# ★ 统一绘图类导入中枢：一次性提供常用 matplotlib 类/子模块，
#   使脚本只写 `from _common import *` 即可直接用，无需各自重复 import（防止遗漏导致 NameError）。
import matplotlib.ticker as mticker
import scipy.stats as stats
from matplotlib.patches import (Rectangle, FancyBboxPatch, FancyArrowPatch, FancyArrow,
                                Circle, RegularPolygon, Ellipse, Wedge, Polygon, Arrow, Arc)
from matplotlib.collections import PatchCollection, LineCollection, PolyCollection
from matplotlib.path import Path
from matplotlib.spines import Spine
try:
    from matplotlib.projections.polar import PolarAxes
    from matplotlib.projections import register_projection
    from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
except Exception:
    pass

# 字体：仅首次运行时重建缓存（marker 防止每次都全量扫描字体拖慢）
def _find_font_dir():
    starts = [BASE_DIR, os.getcwd()]
    for start in starts:
        d = start
        for _ in range(6):
            candidate = os.path.join(d, 'fonts')
            if os.path.isdir(candidate) and os.path.isfile(os.path.join(candidate, 'SourceHanSerifCN-Regular.otf')):
                return candidate
            parent = os.path.dirname(d)
            if parent == d: break
            d = parent
    return None
_SOLVE_DIR = os.path.dirname(BASE_DIR)
# 求解/ 根目录（脚本所在 问题X 的上一级），跨问题共享中间文件统一放这里
SOLVE_DIR = _SOLVE_DIR
_FONT_MARKER = os.path.join(_SOLVE_DIR, '_fonts_ready')
if not os.path.exists(_FONT_MARKER):
    try:
        import glob as _glb
        for _fc in _glb.glob(os.path.join(matplotlib.get_cachedir(), 'font*')):
            try: os.remove(_fc)
            except: pass
        _fm._load_fontmanager(try_read_cache=False)
        open(_FONT_MARKER, 'w').close()
    except Exception:
        pass
_FONT_DIR = _find_font_dir()
if _FONT_DIR:
    for _f in ['SourceHanSerifCN-Regular.otf', 'SourceHanSerifCN-Bold.otf']:
        _p = os.path.join(_FONT_DIR, _f)
        if os.path.exists(_p): _fm.fontManager.addfont(_p)
_SYS_FONT = os.path.join(os.environ.get('WINDIR', 'C:\\Windows'), 'Fonts')
for _f in ['times.ttf', 'timesbd.ttf', 'timesi.ttf', 'simsun.ttc']:
    _p = os.path.join(_SYS_FONT, _f)
    if os.path.exists(_p): _fm.fontManager.addfont(_p)
plt.rcParams['font.family'] = ['Times New Roman', 'Source Han Serif CN', 'SimSun', 'serif']
plt.rcParams['font.weight'] = 'normal'
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 220
plt.rcParams['savefig.bbox'] = 'tight'
plt.rcParams['pdf.fonttype'] = 42

# ★ 兼容层：monkey-patch 旧 API，防止 agent 用 plt.cm.get_cmap() 报错
if not hasattr(matplotlib.cm, 'get_cmap'):
    matplotlib.cm.get_cmap = lambda name, n=None: plt.colormaps[name].resampled(n) if n else plt.colormaps[name]
try:
    from mpl_toolkits.mplot3d import Axes3D as _Axes3D
except ImportError:
    pass

FIG_DIR = os.path.join(BASE_DIR, '图片')
OUT_DIR = os.path.join(BASE_DIR, '结果')
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

# ★ 数据目录：兼容工程内“数据/”与赛题目录中的 real_attachments/。
_project_root = os.path.normpath(os.path.join(BASE_DIR, '..', '..'))
_data_candidates = [
    os.path.join(_project_root, '数据'),
    os.path.join(_project_root, 'real_attachments'),
    r'C:\Users\qq190\Desktop\第二十三届中国研究生数学建模竞赛 - 中文题目\中文题目\F题\real_attachments',
]
DATA_DIR = next((p for p in _data_candidates if os.path.isdir(p)), _data_candidates[0])
_DATA_INDEX = None

def data_path(name):
    """按文件名定位附件；既支持平铺数据目录，也支持 A/B/C 分目录。"""
    direct = os.path.join(DATA_DIR, name)
    if os.path.exists(direct):
        return direct
    global _DATA_INDEX
    if _DATA_INDEX is None:
        _DATA_INDEX = {}
        for root, _, files in os.walk(DATA_DIR):
            for fn in files:
                _DATA_INDEX.setdefault(fn.lower(), os.path.join(root, fn))
    key = os.path.basename(name).lower()
    if key not in _DATA_INDEX:
        raise FileNotFoundError(f'附件中未找到数据文件: {name}（根目录 {DATA_DIR}）')
    return _DATA_INDEX[key]

def despine(ax):
    """只保留左侧和底侧边框，移除上侧和右侧"""
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

def save_fig(fig, name_cn, dpi=300):
    """保存单张图片，自动加.png后缀（默认 300dpi 高清）"""
    path = os.path.join(FIG_DIR, name_cn + '.png')
    fig.savefig(path, dpi=dpi, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print(f'  已保存: {name_cn}.png')

def save_dual(fig, name_cn, dpi=300, bw=False):
    """保存图片（默认 300dpi 高清）。绘图规则只输出全彩，默认不再生成黑白版（bw=False，仅保存彩色）。
    如需临时验证黑白效果可手动传 bw=True，但最终产出图一律不得黑白。"""
    from PIL import Image, ImageOps
    color_path = os.path.join(FIG_DIR, name_cn + '.png')
    fig.savefig(color_path, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f'  已保存: {name_cn}.png')
    if bw:
        bw_path = os.path.join(FIG_DIR, name_cn + '_bw.png')
        try:
            img = Image.open(color_path)
            img_gray = ImageOps.grayscale(img)
            img_gray.save(bw_path)
            img.close()
        except Exception:
            fig.savefig(bw_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f'  已保存: {name_cn}_bw.png')
        print(f'  ★ 双版本完成: {name_cn}')
    plt.close(fig)

def save_csv(df, name_cn):
    df.to_csv(OUT_DIR + '/' + name_cn, index=False, encoding='utf-8-sig')

def shared_path(name):
    """跨问题共享中间文件的统一路径（存到 求解/ 根，供各问题脚本互相读取）。
    例：问题一存 shared_path('预处理数据.csv')，问题二用 pd.read_csv(shared_path('预处理数据.csv')) 读取。"""
    return os.path.join(SOLVE_DIR, name)

def is_ax_empty(ax):
    """检测坐标轴是否无数据（空图）"""
    return (len(ax.lines) == 0 and len(ax.collections) == 0
            and len(ax.patches) == 0 and len(ax.images) == 0)

def is_flat_line(x, y):
    """折线是否为直线（完全无变化→不画）：所有y相同或所有x相同"""
    if x is None or y is None or len(x) < 2 or len(y) < 2:
        return False
    y_arr = np.asarray(y, dtype=float)
    x_arr = np.asarray(x, dtype=float)
    y_flat = np.allclose(y_arr, y_arr[0])
    x_flat = np.allclose(x_arr, x_arr[0])
    return y_flat or x_flat

def validate_subplots(nrows, ncols):
    """
    子图布局校验，不合法返回错误原因字符串，合法返回None。
    规则（硬性）：单图1×1；一行多列1×N(N≥2)；两行2×N且N≥4；其余禁止。
    禁止：三行及以上；两行列数≤3；2×2四宫格。
    """
    if nrows >= 3:
        return f"禁止超过两行（{nrows}行），行列最多两行"
    if nrows == 2 and ncols <= 3:
        return f"两行列数必须>3（当前{ncols}列），换2×{max(4, ncols)}或改单行布局"
    return None

def get_colors(cmap_name, n):
    """安全获取colormap的n个颜色（替代plt.cm.get_cmap，已废弃会报错）。"""
    cmap = plt.colormaps[cmap_name].resampled(n)
    return [cmap(i) for i in range(n)]

def safe_cmap(cmap_name):
    """安全获取colormap对象（不带resampled）。"""
    return plt.colormaps[cmap_name]

def safe_boxplot(ax, data, labels, **kwargs):
    """安全的boxplot（替代ax.boxplot(labels=...)，labels参数已废弃会报错）。
    内部强制 patch_artist=True，使 boxes 为 Patch 对象，AI 可直接用 bp['boxes'][i].set_facecolor(...) 填色；
    否则 boxes 是 Line2D，set_facecolor 会抛 AttributeError。"""
    kwargs.setdefault('patch_artist', True)
    bp = ax.boxplot(data, **kwargs)
    ax.set_xticklabels(labels, fontsize=11)
    return bp

def safe_heatmap(ax, data, row_labels=None, col_labels=None, cmap='RdBu_r',
                 linewidth=0.3, edgecolor='black', fontsize=11, **kwargs):
    """
    强制带黑色细边框的热力图（替代裸 imshow/heatmap 的纯色块）。
    用 pcolormesh 绘制，每个格子边线统一为黑色细框（edgecolor=black, linewidth=linewidth），
    避免"色块连成一片"的观感。NaN 单元格填白色。默认 vmin/vmax 取数据范围。
    用法：
        safe_heatmap(ax, corr.values, row_labels=corr.index, col_labels=corr.columns, cmap='RdBu_r')
    返回 pcolormesh 的 QuadMesh（可用其 .set_clim 调色标范围）。
    """
    data = np.asarray(data, dtype=float)
    rows, cols = data.shape
    cmap_obj = plt.colormaps[cmap]
    cmap_obj.set_bad(color='#FFFFFF')          # NaN 为白，避免黑洞
    mesh = ax.pcolormesh(np.ma.masked_invalid(data), cmap=cmap_obj,
                         edgecolors=edgecolor, linewidth=linewidth, **kwargs)
    if row_labels is not None:
        ax.set_yticks(np.arange(rows) + 0.5)
        ax.set_yticklabels(list(row_labels), fontsize=fontsize)
    if col_labels is not None:
        ax.set_xticks(np.arange(cols) + 0.5)
        ax.set_xticklabels(list(col_labels), rotation=45, ha='right',
                           rotation_mode='anchor', fontsize=fontsize)
    ax.set_xlim(0, cols)
    ax.set_ylim(0, rows)
    ax.set_aspect('auto')
    return mesh

def png_wh(png_path):
    """读取PNG实际宽高（字节，读16-24字节IHDR）"""
    import struct
    with open(png_path, 'rb') as f:
        f.read(16)
        w, h = struct.unpack('>II', f.read(8))
    return w, h

def latex_includegraphics(png_path, caption, label, pair=False, float_pos='ht'):
    r"""
    按高宽比生成匹配档位的 \includegraphics LaTeX 片段（直接粘论文）。
    pair=True：同一张figure里两张并排minipage，给半宽。float_pos默认'ht'，禁止'H'锁死。
    """
    if float_pos.upper() == 'H':
        raise ValueError("float_pos禁止'H'强制定位，默认用'ht'允许浮动")
    w, h = png_wh(png_path)
    aspect = h / w
    if pair:
        width = '0.48\\textwidth'
    elif aspect >= 0.9:
        width = '0.60\\textwidth'
    elif aspect >= 0.6:
        width = '0.65\\textwidth'
    else:
        width = '0.70\\textwidth'
    rel = png_path.replace('\\', '/')
    return (
        f"\\begin{{figure}}[{float_pos}]\n"
        f"  \\centering\n"
        f"  \\includegraphics[width={width}]{{{rel}}}\n"
        f"  \\caption{{{caption}}}\n"
        f"  \\label{{fig:{label}}}\n"
        f"\\end{{figure}}"
    )

# ★ 数据清洗辅助（避免 "--" 等非数值导致 min/max/idxmax 报错）
def numeric_clean(s, default=None):
    """将列安全转为数值：非数值元素→NaN；default 不为 None 时用 default 填充NaN。
    用法：df['评分'] = numeric_clean(df['评分'])   # 先转数值（NaN保留）
          df['评分'] = numeric_clean(df['评分'], default=0.0)   # 转数值并填0"""
    s = pd.to_numeric(s, errors='coerce')
    if default is not None:
        s = s.fillna(default)
    return s

def safe_idxmax(s, default_idx=None):
    """安全的 idxmax：空序列/全NA 返回 default_idx，否则返回最大值索引。"""
    s = pd.to_numeric(s, errors='coerce')
    s = s.dropna()
    if s.empty:
        return default_idx
    return s.idxmax()

def safe_idxmin(s, default_idx=None):
    """安全的 idxmin：空序列/全NA 返回 default_idx，否则返回最小值索引。"""
    s = pd.to_numeric(s, errors='coerce')
    s = s.dropna()
    if s.empty:
        return default_idx
    return s.idxmin()

def has_na(df):
    """检测DataFrame是否含NaN/空值（用于空值防线）。"""
    return bool(df.isna().any().any())
