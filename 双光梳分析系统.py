# -*- coding: utf-8 -*-
"""
双光梳分析系统 —— 单文件版
=====================================================================
频率控制 / 时域采样 / 干涉图 / 射频频谱图
运行：python 双光梳分析系统.py   (需 numpy、matplotlib、tkinter)

1) 双光梳频率控制：中心波长 λc、光谱带宽、重复频率 fr1/fr2、fceo1/fceo2
2) 时域采样    ：记录时长 T、采样率 fs(可自动，自动时取 Δfr 的整数倍以对齐"干涉图切片")
3) 干涉图      ：全记录时域干涉图 + 单个干涉图(1/Δfr 切片)
4) 射频频谱图  ：干涉图 FFT 的 RF 梳，并提取齿列

物理模型(标准双光梳下变频，与 dual-comb-toolkit 的 RF→光学映射一致)：
  第 p 根齿：光学 f1p=f_c+p·fr1, f2p=f_c+p·fr2；像素域拍频(下变频 RF)=p·Δfr+Δfceo；
  干涉图 I(t)=Σ_p I_env(f_p)·cos(2π(pΔfr+Δfceo)t)；RF谱=FFT[I(t)]，齿距Δfr，展宽≈带宽·Δfr/fr。
"""
import os
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import matplotlib
matplotlib.use(os.environ.get("DBACKEND", "TkAgg"))
from matplotlib.figure import Figure
from matplotlib import font_manager
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk


def _pick_cjk_font():
    """挑一个系统里真实存在的中文字体，避免图中的汉字变成方块。"""
    wanted = ["Microsoft YaHei", "Microsoft YaHei UI", "SimHei", "SimSun",
              "KaiTi", "FangSong", "Noto Sans CJK SC", "Source Han Sans SC",
              "WenQuanYi Zen Hei", "PingFang SC", "Heiti SC"]
    avail = {f.name for f in font_manager.fontManager.ttflist}
    for w in wanted:
        if w in avail:
            return w
    return None


CJK_FONT = _pick_cjk_font()
matplotlib.rcParams["font.family"] = "sans-serif"
if CJK_FONT:
    matplotlib.rcParams["font.sans-serif"] = [CJK_FONT, "DejaVu Sans"]
else:
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]

matplotlib.rcParams["axes.unicode_minus"] = False   # 避免负号缺字形

C = 2.99792458e8


def L2(lam_c=1560.0, bw=10.0, fr1=200.0, fr2=200.01,
        fceo1=0.0, fceo2=0.0, T_ms=1.0, fs_mhz=None, auto_fs=True):
    """计算双光梳下变频干涉图与 RF 频谱。"""
    if fr1 <= 0 or fr2 <= 0:
        raise ValueError("重复频率必须为正数")
    dfr = fr2 - fr1
    if abs(dfr) < 1e-9:
        raise ValueError("Δfr=0，两梳重频相同，无拍频。请调整 fr2。")
    if bw <= 0:
        raise ValueError("光谱带宽必须为正数")
    if T_ms <= 0:
        raise ValueError("记录时长必须为正数")
    dfceo = fceo2 - fceo1

    fr1_hz, fr2_hz = fr1 * 1e6, fr2 * 1e6
    dfr_hz, dfceo_hz = dfr * 1e6, dfceo * 1e6

    lam_c_m = lam_c * 1e-9
    f_center = C / lam_c_m
    f_high = C / ((lam_c - bw / 2.0) * 1e-9)
    f_low = C / ((lam_c + bw / 2.0) * 1e-9)
    FWHM_f = f_high - f_low
    sigma_int = FWHM_f / (2.0 * np.sqrt(np.log(2.0)))

    P = int(round((FWHM_f / 2.0) / fr1_hz))
    p = np.arange(-P, P + 1, dtype=float)
    f_opt1 = f_center + p * fr1_hz
    f_opt2 = f_center + p * fr2_hz
    env_i = np.exp(-((f_opt1 - f_center) / sigma_int) ** 2)

    frf_hz = p * dfr_hz + dfceo_hz
    maxRF = float(np.abs(frf_hz).max())

    fs_min = max(5.0 * maxRF, 150.0e6)
    if auto_fs:
        nspi = int(np.ceil(fs_min / abs(dfr_hz)))
        fs = nspi * abs(dfr_hz)
    else:
        if not fs_mhz or fs_mhz <= 0:
            raise ValueError("手动采样率必须为正数(MHz)")
        fs = fs_mhz * 1e6
    T_sec = T_ms * 1e-3
    N = int(round(T_sec * fs))
    if N < 4096:
        N = 4096
    if N > 12_000_000:
        raise ValueError(f"采样点 N={N} 过大，请减小记录时长 T 或采样率 fs。")
    dt = 1.0 / fs
    dfbin = fs / N
    aliasing = fs < 2.0 * maxRF

    S = np.zeros(N, dtype=complex)
    for i, f in enumerate(frf_hz):
        if p[i] == 0:
            continue
        fa = abs(f)
        if fa > fs / 2.0:
            continue
        k = int(round(fa / dfbin))
        if k <= 0 or k >= N:
            continue
        S[k] += env_i[i]
        S[N - k] += env_i[i]
    sig = np.fft.ifft(S).real
    t = np.arange(N) * dt

    RF = np.fft.rfft(sig)
    freq_rf = np.fft.rfftfreq(N, d=dt)
    mag = np.abs(RF)
    mag[0] = 0.0
    mm = mag.max()
    mag_db = 20.0 * np.log10(np.maximum(mag, mm * 1e-9) / mm) if mm > 0 else np.zeros_like(mag)

    nspi = max(1, int(round(fs / abs(dfr_hz))))
    n_int = N // nspi
    start = min(nspi // 2, max(0, N - nspi))          # 让一个 burst 位于切片中心
    one_int = sig[start:start + nspi].copy()
    one_t = (np.arange(nspi) - nspi // 2) * dt * 1e6  # 以 burst 中心为 0 的时间轴(μs)
    f_one = np.fft.rfftfreq(nspi, d=dt)
    one_spec = np.abs(np.fft.rfft(one_int))
    one_spec[0] = 0.0
    if one_spec.max() > 0:
        one_spec /= one_spec.max()
    mag_lin = mag / mm if mm > 0 else mag            # 全记录 RF 归一化线性幅度

    lam1 = C / f_opt1 * 1e9
    lam2 = C / f_opt2 * 1e9

    burst_us = 1.0 / abs(dfr)
    info = [
        f"中心频率 {f_center/1e12:.4f} THz  (λc={lam_c:.1f} nm)",
        f"光谱带宽(强度FWHM) {bw:.1f} nm = {FWHM_f/1e12:.3f} THz",
        f"梳齿数 {2*P+1}  (半宽P={P}，光学齿距 {fr1:.4f} MHz)",
        f"Δfr = {dfr:.4f} MHz = {dfr*1e3:.3f} kHz;  Δfceo = {dfceo:.4f} MHz",
        f"干涉图周期 1/Δfr = {burst_us:.1f} μs;  本记录含 {n_int} 个干涉图",
        f"RF 带宽(下变频) {maxRF/1e6:.2f} MHz;  映射比例 Δfr/fr = {dfr/fr1:.3e}",
        f"采样率 fs = {fs/1e6:.3f} MHz ({'自动' if auto_fs else '手动'})，采样点 N={N}",
        f"每干涉图采样点 = {nspi} (=fs/Δfr，对齐切片)，RF分辨率 {dfbin/1e3:.2f} kHz",
    ]
    if aliasing:
        info.append(f"⚠ 采样率 {fs/1e6:.1f} MHz 低于 Nyquist({2*maxRF/1e6:.1f} MHz)，RF 存在混叠")
    return dict(t_us=t * 1e6, sig=sig, fs=fs, N=N, dt=dt,
                freq_rf_mhz=freq_rf / 1e6, mag_db=mag_db, mag_lin=mag_lin,
                one_int_us=one_t, one_int=one_int,
                f_one_mhz=f_one / 1e6, one_spec=one_spec,
                lam1=lam1, lam2=lam2, env1=env_i, env2=env_i,
                info=info, dfr=dfr, dfceo=dfceo, P=P, maxRF_mhz=maxRF / 1e6,
                n_int=n_int, nspi=nspi, aliasing=aliasing)


def _teeth(freq_mhz, spec, thr=0.4):
    m = spec
    idx = np.where((m[1:-1] > m[:-2]) & (m[1:-1] >= m[2:]))[0] + 1
    idx = idx[m[idx] > thr * m.max()]
    return idx


def draw_all(fig, d):
    """在给定 Figure 上绘制 6 幅图(供 GUI 与无界面校验共用)。"""
    fig.clear()
    ax1 = fig.add_subplot(2, 3, 1)
    ax2 = fig.add_subplot(2, 3, 2)
    ax3 = fig.add_subplot(2, 3, 3)
    ax4 = fig.add_subplot(2, 3, 4)
    ax5 = fig.add_subplot(2, 3, 5)
    ax6 = fig.add_subplot(2, 3, 6)

    ax1.plot(d["lam1"], d["env1"], lw=1.1, color="tab:blue", label="梳1")
    ax1.plot(d["lam2"], d["env2"], lw=1.0, ls="--", color="tab:red", label="梳2")
    ax1.set_xlabel("波长 (nm)"); ax1.set_ylabel("相对强度")
    ax1.set_title("① 双光梳光学梳谱", fontsize=9)
    ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    ax2.plot(d["t_us"], d["sig"], lw=0.5, color="tab:blue")
    ax2.set_xlim(0, d["t_us"][-1])
    ax2.set_xlabel("时间 (μs)"); ax2.set_ylabel("干涉信号 (a.u.)")
    ax2.set_title(f"② 时域干涉图 (T={d['t_us'][-1]:.0f} μs，{d['n_int']}个)", fontsize=9)
    ax2.grid(alpha=0.3)

    ax3.plot(d["one_int_us"], d["one_int"], lw=0.7, color="tab:orange")
    ax3.set_xlabel("时间 (μs, burst中心=0)"); ax3.set_ylabel("干涉信号")
    ax3.set_title(f"③ 单个干涉图 (1/Δfr={1/abs(d['dfr']):.0f} μs)", fontsize=9)
    ax3.grid(alpha=0.3)

    w = 0.12
    m = d["t_us"] < w
    ax4.plot(d["t_us"][m], d["sig"][m], "-", lw=0.6, color="tab:purple", alpha=0.6)
    ax4.plot(d["t_us"][m], d["sig"][m], "o", ms=3.5, color="tab:purple")
    for sp in d["t_us"][m]:
        ax4.axvline(sp, color="gray", lw=0.3, alpha=0.35)
    ax4.set_xlim(0, w)
    ax4.set_xlabel("时间 (μs)"); ax4.set_ylabel("干涉信号")
    ax4.set_title(f"④ 时域采样 (fs={d['fs']/1e6:.1f} MHz，dt={d['dt']*1e9:.2f} ns)", fontsize=9)
    ax4.grid(alpha=0.3)

    hi = max(d["maxRF_mhz"] * 1.15, 1.0)
    ax5.plot(d["freq_rf_mhz"], d["mag_db"], lw=0.5, color="tab:green")
    ax5.set_xlim(0, hi); ax5.set_ylim(-90, 3)
    ax5.set_xlabel("RF 频率 (MHz)"); ax5.set_ylabel("幅度 (dB)")
    ax5.set_title("⑤ 射频频谱图 (FFT)", fontsize=9)
    ax5.grid(alpha=0.3)

    f6 = d["freq_rf_mhz"]; s6 = d["mag_lin"]
    wzoom = min(0.35, max(0.05, hi * 0.05))          # 放大一小段以看清单根齿
    ax6.plot(f6, s6, lw=0.8, color="tab:green")
    idx = _teeth(f6, s6, 0.25)
    if len(idx):
        idx = idx[f6[idx] <= wzoom]
    if len(idx):
        ax6.stem(f6[idx], s6[idx], linefmt="C3-", markerfmt="C3.", basefmt=" ")
    ax6.set_xlim(0, wzoom); ax6.set_ylim(0, 1.08)
    ax6.set_xlabel("RF 频率 (MHz)"); ax6.set_ylabel("归一化幅度")
    ax6.set_title(f"⑥ RF梳齿放大 (0–{wzoom:.2f} MHz，Δfr={d['dfr']*1e3:.2f} kHz)", fontsize=9)
    ax6.grid(alpha=0.3)

    fig.tight_layout(pad=2.0)


class App:
    def __init__(self, root):
        self.root = root
        root.title("双光梳分析系统 · 频率控制 | 时域采样 | 干涉图 | 射频频谱")
        root.geometry("1420x840")
        self.e = {}
        self.auto = tk.BooleanVar(value=True)
        self._panel()
        self._plot()
        self.run()

    def _row(self, parent, label, key, default, unit):
        f = ttk.Frame(parent)
        f.pack(fill="x", pady=2)
        ttk.Label(f, text=label, width=13).pack(side="left")
        ent = ttk.Entry(f, width=11)
        ent.insert(0, str(default))
        ent.pack(side="left")
        ttk.Label(f, text=unit).pack(side="left", padx=(3, 0))
        self.e[key] = ent

    def _panel(self):
        panel = ttk.Frame(self.root, padding=8)
        panel.pack(side="left", fill="y", padx=6, pady=6)

        g1 = ttk.LabelFrame(panel, text="双光梳频率控制", padding=6)
        g1.pack(fill="x")
        self._row(g1, "中心波长 λc", "lam_c", 1560.0, "nm")
        self._row(g1, "光谱带宽", "bw", 10.0, "nm")
        self._row(g1, "重复频率 fr1", "fr1", 200.0, "MHz")
        self._row(g1, "重复频率 fr2", "fr2", 200.01, "MHz")
        self._row(g1, "fceo1", "fceo1", 0.0, "MHz")
        self._row(g1, "fceo2", "fceo2", 0.0, "MHz")

        g2 = ttk.LabelFrame(panel, text="时域采样", padding=6)
        g2.pack(fill="x", pady=(8, 0))
        self._row(g2, "记录时长 T", "T_ms", 1.0, "ms")
        self._row(g2, "采样率 fs", "fs_mhz", 154.0, "MHz")
        ttk.Checkbutton(g2, text="自动采样率(取Δfr整数倍)", variable=self.auto).pack(anchor="w", pady=(3, 0))

        ttk.Button(panel, text="计算 / 刷新", command=self.run).pack(fill="x", pady=(10, 4))

        self.txt = tk.Text(panel, width=42, height=15,
                           font=(CJK_FONT or "Consolas", 9),
                           state="disabled", wrap="word")
        self.txt.pack(fill="both", expand=True, pady=(4, 0))

    def _plot(self):
        right = ttk.Frame(self.root)
        right.pack(side="left", fill="both", expand=True, padx=(0, 6), pady=6)
        ttk.Label(right, text="缩放：鼠标滚轮缩放当前子图 | 双击子图复位 | 工具栏可框选缩放/平移/Home复位").pack(anchor="w")
        self.fig = Figure(figsize=(12.5, 7.2), dpi=96)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right)
        self.canvas.get_tk_widget().pack(fill="both", expand=True)
        NavigationToolbar2Tk(self.canvas, right).update()
        self.canvas.mpl_connect("scroll_event", self._on_scroll)
        self.canvas.mpl_connect("button_press_event", self._on_dblclick)

    def _on_scroll(self, event):
        ax = event.inaxes
        if ax is None or event.xdata is None or event.ydata is None:
            return
        s = 1.0 / 1.2 if event.button == "up" else 1.2
        x0, x1 = ax.get_xlim(); x = event.xdata
        ax.set_xlim(x - (x - x0) * s, x + (x1 - x) * s)
        y0, y1 = ax.get_ylim(); y = event.ydata
        ax.set_ylim(y - (y - y0) * s, y + (y1 - y) * s)
        self.canvas.draw_idle()

    def _on_dblclick(self, event):
        if event.inaxes is not None:
            self.run()

    def _v(self, k):
        return float(self.e[k].get())

    def run(self):
        try:
            d = L2(lam_c=self._v("lam_c"), bw=self._v("bw"),
                   fr1=self._v("fr1"), fr2=self._v("fr2"),
                   fceo1=self._v("fceo1"), fceo2=self._v("fceo2"),
                   T_ms=self._v("T_ms"),
                   fs_mhz=self._v("fs_mhz") if not self.auto.get() else None,
                   auto_fs=self.auto.get())
        except Exception as ex:
            messagebox.showerror("参数错误", str(ex))
            return
        draw_all(self.fig, d)
        self.canvas.draw()
        self.txt.configure(state="normal")
        self.txt.delete("1.0", "end")
        self.txt.insert("1.0", "\n".join("· " + s for s in d["info"]))
        self.txt.configure(state="disabled")
        if not self.auto.get():
            self.e["fs_mhz"].delete(0, "end")
            self.e["fs_mhz"].insert(0, "%.3f" % (d["fs"] / 1e6))


if __name__ == "__main__":
    root = tk.Tk()
    App(root)
    root.mainloop()
