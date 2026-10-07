#!/usr/bin/env python3
"""vid2script.py - 视频反推剧本的素材准备工具（只依赖 ffmpeg/ffprobe + numpy + opencv）。

子命令:
  analyze VIDEO OUT      全量分析: 元数据、镜头切点、镜头关键帧拼图、1fps 时间码拼图、
                         字幕条自动定位+变化检测拼图、每条字幕的音高(说话人性别提示)、timeline.md
  zoom VIDEO START DUR OUT.png [--fps 4] [--crop x,y,w,h]
                         把某一小段放大成拼图，用于核对口型/动作/道具
  pitch VIDEO START END  单独测某一段的音高（带背景噪声扣除）
  asr VIDEO OUTDIR       （可选）语音转写；环境装不上模型时会说明并退出
  render SPEC OUT.md     把精简剧本数据展开成完整分镜格式（开头自动加总览，可带参考图提示词），并自动检查时间轴/语速/参考图/白名单/台词覆盖/改编残留
"""
import argparse, json, os, re, subprocess, sys, math
import numpy as np
import cv2

FONT = cv2.FONT_HERSHEY_SIMPLEX


def run(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def probe(video):
    r = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", video])
    j = json.loads(r.stdout)
    v = next(s for s in j["streams"] if s["codec_type"] == "video")
    num, den = v.get("r_frame_rate", "30/1").split("/")
    fps = float(num) / float(den or 1)
    return {
        "duration": float(j["format"]["duration"]),
        "width": int(v["width"]), "height": int(v["height"]), "fps": round(fps, 3),
        "has_audio": any(s["codec_type"] == "audio" for s in j["streams"]),
    }


def scene_cuts(video, thr=0.3):
    r = run(["ffmpeg", "-hide_banner", "-i", video, "-filter:v", f"select='gt(scene,{thr})',showinfo", "-f", "null", "-"])
    ts = [float(x) for x in re.findall(r"pts_time:([0-9.]+)", r.stderr)]
    # 合并 0.4s 内的抖动切点（闪白/转场会连续触发）
    out = []
    for t in ts:
        if not out or t - out[-1] > 0.4:
            out.append(t)
    return out


def read_frames(video, fps, vf_extra="", scale_w=None):
    """用 ffmpeg 按 fps 解码成 numpy 帧（BGR）。返回 (times, frames)。"""
    info = probe(video)
    w, h = info["width"], info["height"]
    if scale_w:
        h = int(round(h * scale_w / w / 2) * 2); w = scale_w
    vf = f"fps={fps}" + (f",scale={w}:{h}" if scale_w else "") + (("," + vf_extra) if vf_extra else "")
    if vf_extra.startswith("crop"):
        m = re.match(r"crop=(\d+):(\d+)", vf_extra)
        w, h = int(m.group(1)), int(m.group(2))
    p = subprocess.Popen(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-vf", vf,
                          "-f", "rawvideo", "-pix_fmt", "bgr24", "-"], stdout=subprocess.PIPE)
    frames = []
    size = w * h * 3
    while True:
        buf = p.stdout.read(size)
        if len(buf) < size:
            break
        frames.append(np.frombuffer(buf, np.uint8).reshape(h, w, 3))
    p.wait()
    times = [i / fps for i in range(len(frames))]
    return times, frames


def label(img, text, scale=None):
    img = img.copy()
    s = scale or max(0.5, img.shape[1] / 700)
    th = max(1, int(s * 2))
    (tw, tht), _ = cv2.getTextSize(text, FONT, s, th)
    cv2.rectangle(img, (0, 0), (tw + 8, tht + 10), (0, 0, 0), -1)
    cv2.putText(img, text, (4, tht + 4), FONT, s, (0, 255, 255), th)
    return img


def tile(imgs, cols, out_prefix, rows=3):
    """imgs: 同尺寸图片列表；每张拼图 cols x rows，输出多张。"""
    paths = []
    per = cols * rows
    if not imgs:
        return paths
    h, w = imgs[0].shape[:2]
    for k in range(0, len(imgs), per):
        chunk = imgs[k:k + per]
        while len(chunk) < per:
            chunk.append(np.zeros((h, w, 3), np.uint8))
        grid = np.vstack([np.hstack(chunk[r * cols:(r + 1) * cols]) for r in range(rows)])
        p = f"{out_prefix}_{k // per + 1:02d}.png"
        cv2.imwrite(p, grid)
        paths.append(p)
    return paths


# ---------- 字幕条定位 ----------
def text_mask(gray):
    """亮色字+局部更暗描边/阴影 的像素掩码。用局部对比而不是绝对黑边，
    这样字幕压在白衣服、天空等浅色背景上也能检出。"""
    white = gray > 195
    mn = cv2.erode(gray, np.ones((5, 5), np.uint8))
    return (white & ((gray.astype(np.int16) - mn) > 70)).astype(np.uint8)


def find_sub_band(video, info):
    """在画面下部 40% 内，统计“白字黑边”像素密度最高的水平带。返回 (y0, y1, x0, x1) 原始分辨率坐标。"""
    W, H = info["width"], info["height"]
    sw = 960 if W >= 960 else W - W % 2
    sh = int(round(H * sw / W / 2) * 2)
    # 每秒 2 帧，用相邻帧(0.5s)都出现的亮字像素 —— 字幕是静止的，人物/高光会动
    fps = 2.0
    _, frames = read_frames(video, fps, scale_w=sw)
    y_lo = int(sh * 0.55)
    x0, x1 = int(sw * 0.20), int(sw * 0.80)  # 字幕居中；避开左右角落水印
    prof = np.zeros(sh - y_lo)
    prev = None
    for f in frames:
        m = text_mask(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)[y_lo:, x0:x1])
        if prev is not None:
            prof += (m & prev).sum(1)
        prev = m
    prof /= max(1, len(frames))
    if prof.max() < 0.3:
        return None
    sm = np.convolve(prof, np.ones(3) / 3, "same")
    peak = int(np.argmax(sm))
    for frac in (0.25, 0.4, 0.55, 0.7):
        thr = sm[peak] * frac
        a = peak
        while a > 0 and sm[a - 1] > thr:
            a -= 1
        b = peak
        while b < len(sm) - 1 and sm[b + 1] > thr:
            b += 1
        if b - a <= 0.10 * sh:  # 一到两行字幕的高度
            break
    pad = max(4, int((b - a) * 0.45))
    a, b = max(0, a - pad), min(len(sm) - 1, b + pad)
    k = H / sh
    y0, y1 = int((y_lo + a) * k), int((y_lo + b) * k)
    y0 = max(0, y0); y1 = min(H, max(y1, y0 + int(H * 0.04)))
    # 两行字幕时可能分成两段，取最大连通带即可；横向用 8%~92%
    return y0 - y0 % 2, y1 - y1 % 2, int(W * 0.08) // 2 * 2, int(W * 0.92) // 2 * 2


def _iou(a, b):
    k = np.ones((3, 3), np.uint8)
    a = cv2.dilate(a, k); b = cv2.dilate(b, k)
    inter = int((a & b).sum()); uni = int((a | b).sum())
    return inter / uni if uni else 1.0


def subtitle_events(video, band, fps=5):
    """偏向“宁可多报不漏报”：多报的空条/重复条读图时直接忽略，漏报才是真正的损失。
    判据：字在 3 帧（0.4s）内静止 + 大致水平居中 + 像素量超过下限。"""
    y0, y1, x0, x1 = band
    w, h = x1 - x0, y1 - y0
    times, frames = read_frames(video, fps, vf_extra=f"crop={w}:{h}:{x0}:{y0}")
    raw = [text_mask(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)) for f in frames]
    n = len(raw)
    floor = max(60, int(0.0025 * w * h))
    stable, on = [], []
    for i in range(n):
        m = raw[i] & raw[max(0, i - 1)] & raw[min(n - 1, i + 1)]
        stable.append(m)
        c = int(m.sum())
        ok = c > floor
        if ok:
            cols = m.sum(0)
            cx = (cols * np.arange(w)).sum() / max(c, 1)
            ok = abs(cx - w / 2) < 0.18 * w  # 字幕居中
        on.append(ok)
    segs, cur = [], None
    for i in range(n):
        if not on[i]:
            if cur:
                segs.append(cur); cur = None
            continue
        if cur is None:
            cur = [i, i, stable[i]]
        elif _iou(stable[i], cur[2]) < 0.45:
            segs.append(cur); cur = [i, i, stable[i]]
        else:
            cur[1] = i
    if cur:
        segs.append(cur)
    out = []
    for a, b, ref in segs:
        if b - a + 1 < 2:  # 只闪了一帧的丢弃
            continue
        mid = (a + b) // 2
        out.append({"start": round(times[a], 2), "end": round(times[b] + 1 / fps, 2), "frame": frames[mid], "ref": ref,
                    "mid": mid})
    # 合并相邻、内容相同的段（中间被一帧遮挡打断的情况）
    merged = []
    for s in out:
        if merged and s["start"] - merged[-1]["end"] < 0.65 and _iou(merged[-1]["ref"], s["ref"]) > 0.45:
            merged[-1]["end"] = s["end"]; continue
        merged.append(s)
    for s in merged:
        s["ref"] = stable[s["mid"]]
    counts = [int(m.sum()) for m in raw]
    return merged, times, frames, counts, floor, stable


def band_filmstrip(times, frames, out_prefix, step=2, per=50, keep=None):
    """字幕带胶片：每 step 帧(5fps 下 step=2 即 0.4s)截一条，按时间排开。
    keep(i) 返回 False 的帧跳过（省钱模式只保留“有亮字但不在已检出字幕里”的帧）。"""
    strips = []
    for i in range(0, len(frames), step):
        if keep is not None and not keep(i):
            continue
        fr = frames[i]
        sw = 560
        fr = cv2.resize(fr, (sw, max(20, int(fr.shape[0] * sw / fr.shape[1]))))
        lab = np.zeros((fr.shape[0], 70, 3), np.uint8)
        cv2.putText(lab, f"{times[i]:.1f}", (3, int(fr.shape[0] * 0.7)), FONT, 0.5, (0, 255, 255), 1)
        strips.append(np.hstack([lab, fr]))
    paths = []
    for k in range(0, len(strips), per):
        ch = strips[k:k + per]
        half = (len(ch) + 1) // 2
        c1 = np.vstack(ch[:half])
        c2 = np.vstack(ch[half:]) if ch[half:] else np.zeros_like(c1)
        if c2.shape[0] < c1.shape[0]:
            c2 = cv2.copyMakeBorder(c2, 0, c1.shape[0] - c2.shape[0], 0, 0, cv2.BORDER_CONSTANT)
        p = f"{out_prefix}_{k // per + 1:02d}.png"
        cv2.imwrite(p, np.hstack([c1, np.full((c1.shape[0], 6, 3), 128, np.uint8), c2]))
        paths.append(p)
    return paths


# ---------- 音频 ----------
def load_audio(video, sr=16000):
    p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", video, "-ac", "1", "-ar", str(sr),
                        "-f", "s16le", "-"], capture_output=True)
    return np.frombuffer(p.stdout, np.int16).astype(np.float32) / 32768, sr


def noise_profile(x, sr, speech_segs, n_fft=1024):
    """取不在字幕时段内的音频当作背景（音乐/环境）噪声谱。"""
    mask = np.ones(len(x), bool)
    for s in speech_segs:
        mask[int(s[0] * sr):int(s[1] * sr)] = False
    nz = x[mask]
    if len(nz) < sr:
        nz = x[: sr]
    frames = [nz[i:i + n_fft] * np.hanning(n_fft) for i in range(0, len(nz) - n_fft, n_fft // 2)][:2000]
    if not frames:
        return None
    return np.mean(np.abs(np.fft.rfft(np.array(frames), axis=1)) ** 2, axis=0)


def f0_hps(seg, sr, npow=None, n_fft=1024):
    """谱减降噪 + 谐波乘积谱 估计基频中位数。返回 (f0, voiced_ratio)。"""
    if len(seg) < n_fft * 2:
        return 0.0, 0.0
    vals = []
    total = 0
    big = 8192
    for i in range(0, len(seg) - n_fft, n_fft // 2):
        fr = seg[i:i + n_fft] * np.hanning(n_fft)
        if np.sqrt((fr ** 2).mean()) < 0.008:
            continue
        total += 1
        S = np.abs(np.fft.rfft(fr, n_fft)) ** 2
        if npow is not None:
            S = np.maximum(S - 1.5 * npow, 0.05 * S)
        # 插值到更细的频率网格
        Sb = np.interp(np.linspace(0, len(S) - 1, big // 2 + 1), np.arange(len(S)), np.sqrt(S))
        h = Sb.copy()
        for k in (2, 3, 4):
            h[: len(Sb) // k] *= Sb[::k][: len(Sb) // k]
        freqs = np.linspace(0, sr / 2, big // 2 + 1)
        m = (freqs > 70) & (freqs < 450)
        idx = np.argmax(h * m)
        peak = h[idx]
        if peak <= 0:
            continue
        vals.append(freqs[idx])
    if not vals:
        return 0.0, 0.0
    return float(np.median(vals)), len(vals) / max(total, 1)


def gender_hint(f0):
    if f0 <= 0:
        return "?"
    if f0 < 165:
        return "男"
    if f0 > 210:
        return "女"
    return "?"


# ---------- 命令 ----------
def cmd_analyze(a):
    video, out = a.video, a.out
    os.makedirs(out, exist_ok=True)
    info = probe(video)
    dur = info["duration"]
    print(f"[1/6] 元数据: {info}")
    if a.subs_only and os.path.exists(os.path.join(out, "analysis.json")):
        old = json.load(open(os.path.join(out, "analysis.json"), encoding="utf-8"))
        shots = old["shots"]; sec_sheets = old["sheets"]["seconds"]; shot_sheets = old["sheets"]["shots"]
        print("[2-4/6] --subs-only：沿用已有的镜头表和拼图，只重做字幕")
    else:
        shots, sec_sheets, shot_sheets = analyze_visual(video, out, info, a.scene, a.full)
    analyze_subs_and_write(video, out, info, a.band, shots, sec_sheets, shot_sheets, a.full)


def analyze_visual(video, out, info, scene, full=False):
    dur = info["duration"]
    cuts = scene_cuts(video, scene)
    bounds = [0.0] + [c for c in cuts if 0.2 < c < dur - 0.2] + [dur]
    shots = [{"id": i + 1, "start": round(bounds[i], 2), "end": round(bounds[i + 1], 2)} for i in range(len(bounds) - 1)]
    print(f"[2/6] 镜头切点: {len(shots)} 个镜头")

    # 1fps 时间码拼图：只有 --full 才生成（省钱模式下，有疑问的时段用 zoom 单独看）
    sec_sheets = []
    if full:
        times, frames = read_frames(video, 1, scale_w=480)
        imgs = [label(f, f"{int(t // 60):02d}:{t % 60:04.1f}") for t, f in zip(times, frames)]
        sec_sheets = tile(imgs, 4, os.path.join(out, "sec"), rows=3)
        print(f"[3/6] 1fps 拼图: {len(sec_sheets)} 张 (每张 12 秒)")
    else:
        print("[3/6] 省钱模式：不生成每秒拼图（需要时用 zoom 看具体时段，或加 --full）")

    # 镜头关键帧拼图：省钱模式下短于 3 秒的镜头只取中间一帧，长镜头取 25%/75% 两帧看运镜
    tw = 480 if full else 384
    shot_imgs = []
    for s in shots:
        long_shot = full or (s["end"] - s["start"]) >= 3.0
        picks = ((0.25, "a"), (0.75, "b")) if long_shot else ((0.5, ""),)
        for q, tag in picks:
            t = s["start"] + (s["end"] - s["start"]) * q
            p = subprocess.run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-ss", f"{t:.2f}", "-i", video,
                                "-frames:v", "1", "-vf", f"scale={tw}:-2", "-f", "rawvideo", "-pix_fmt", "bgr24", "-"],
                               capture_output=True)
            hh = len(p.stdout) // (tw * 3)
            if hh <= 0:
                continue
            im = np.frombuffer(p.stdout[: tw * hh * 3], np.uint8).reshape(hh, tw, 3)
            shot_imgs.append(label(im, f"S{s['id']:02d}{tag} {s['start']:.1f}-{s['end']:.1f}s"))
    h0 = shot_imgs[0].shape[0]
    shot_imgs = [cv2.resize(i, (tw, h0)) if i.shape[0] != h0 else i for i in shot_imgs]
    cols, rows = (4, 3) if full else (5, 4)
    shot_sheets = tile(shot_imgs, cols, os.path.join(out, "shots"), rows=rows)
    print(f"[4/6] 镜头拼图: {len(shot_sheets)} 张 (长镜头 a=前段 b=后段，短镜头一帧)")
    return shots, sec_sheets, shot_sheets


def analyze_subs_and_write(video, out, info, band_arg, shots, sec_sheets, shot_sheets, full=False):
    dur = info["duration"]
    band = None
    if band_arg:
        band = tuple(int(v) for v in band_arg.split(","))
    else:
        band = find_sub_band(video, info)
    subs = []
    sub_sheets = []
    film_sheets = []
    if band:
        evs, btimes, bframes, bcounts, bfloor, bstable = subtitle_events(video, band)
        keep = None
        if not full:
            # 省钱模式：胶片只保留可能漏句的帧 ——
            # ① 字幕区有亮字、但不在任何已检出字幕时段内；
            # ② 在某条字幕时段内，但字形和这条字幕的代表帧明显不同（相似的两句被合并成一条的情况）
            def keep(i):
                if bcounts[i] <= 0.4 * bfloor:
                    return False
                t = btimes[i]
                hit = [e for e in evs if e["start"] - 0.2 <= t <= e["end"] + 0.2]
                if not hit:
                    return True
                return all(_iou(bstable[i], e["ref"]) < 0.6 for e in hit) and int(bstable[i].sum()) > bfloor
        film_sheets = band_filmstrip(btimes, bframes, os.path.join(out, "film"), keep=keep)
        del bframes, bstable
        for e in evs:
            e.pop("ref", None); e.pop("mid", None)
        x, sr = (load_audio(video) if info["has_audio"] else (None, 16000))
        npow = noise_profile(x, sr, [(e["start"], e["end"]) for e in evs]) if x is not None else None
        strips = []
        for i, e in enumerate(evs):
            f0, vr = (0.0, 0.0)
            if x is not None:
                f0, vr = f0_hps(x[int(e["start"] * sr):int(e["end"] * sr)], sr, npow)
            shot_ids = [s["id"] for s in shots if s["start"] < e["end"] and s["end"] > e["start"]]
            subs.append({"id": i + 1, "start": e["start"], "end": e["end"], "shots": shot_ids,
                         "f0": round(f0), "voice_hint": gender_hint(f0), "text": ""})
            fr = e["frame"]
            sw = 640
            fr = cv2.resize(fr, (sw, max(24, int(fr.shape[0] * sw / fr.shape[1]))))
            lab = np.zeros((fr.shape[0], 250, 3), np.uint8)
            cv2.putText(lab, f"#{i + 1} {e['start']:.1f}-{e['end']:.1f}", (4, int(fr.shape[0] * 0.45)), FONT, 0.5, (0, 255, 255), 1)
            cv2.putText(lab, f"S{','.join(map(str, shot_ids))} f0={round(f0)} {gender_hint(f0)}", (4, int(fr.shape[0] * 0.9)),
                        FONT, 0.45, (180, 255, 180), 1)
            strips.append(np.hstack([lab, fr]))
        if strips:
            hmax = max(s.shape[0] for s in strips)
            strips = [cv2.copyMakeBorder(s, 0, hmax - s.shape[0], 0, 0, cv2.BORDER_CONSTANT) for s in strips]
            per = 40
            for k in range(0, len(strips), per):
                ch = strips[k:k + per]
                half = (len(ch) + 1) // 2
                c1 = np.vstack(ch[:half]); c2 = ch[half:]
                c2 = np.vstack(c2) if c2 else np.zeros_like(c1)
                if c2.shape[0] < c1.shape[0]:
                    c2 = cv2.copyMakeBorder(c2, 0, c1.shape[0] - c2.shape[0], 0, 0, cv2.BORDER_CONSTANT)
                p = os.path.join(out, f"subs_{k // per + 1:02d}.png")
                cv2.imwrite(p, np.hstack([c1, np.full((c1.shape[0], 6, 3), 128, np.uint8), c2]))
                sub_sheets.append(p)
        print(f"[5/6] 字幕区 y={band[0]}-{band[1]} x={band[2]}-{band[3]}（看 film 拼图确认字没被切掉，不对就 --band 重跑），检测到 {len(subs)} 段字幕变化（含少量重复/空条），"
              f"字幕拼图 {len(sub_sheets)} 张，字幕带胶片 {len(film_sheets)} 张")
    else:
        print("[5/6] 没找到烧录字幕条（可能无字幕）。可用 --band y0,y1,x0,x1 手动指定；台词改用 asr 子命令或请用户提供。")

    # 音频能量概览 + 静音段
    sil = run(["ffmpeg", "-hide_banner", "-i", video, "-af", "silencedetect=n=-38dB:d=0.35", "-f", "null", "-"]).stderr
    silences = re.findall(r"silence_start: ([0-9.]+)[\s\S]*?silence_end: ([0-9.]+)", sil)

    data = {"video": os.path.abspath(video), "info": info, "subtitle_band": band, "shots": shots,
            "subtitles": subs, "silences": [[float(s), float(e)] for s, e in silences],
            "sheets": {"shots": shot_sheets, "seconds": sec_sheets, "subtitles": sub_sheets, "filmstrip": film_sheets}}
    with open(os.path.join(out, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)

    # timeline.md：按镜头列出，字幕文字由 Claude 读图后填写
    W, H = info["width"], info["height"]
    r = W / H
    aspect = "16:9" if abs(r - 16 / 9) < 0.08 else "9:16" if abs(r - 9 / 16) < 0.05 else "4:3" if abs(r - 4 / 3) < 0.05 \
        else "1:1" if abs(r - 1) < 0.05 else "2.35:1" if r > 2.2 else f"{W}:{H}"
    data["info"]["aspect"] = aspect
    with open(os.path.join(out, "analysis.json"), "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)
    lines = [f"# 时间线  时长 {dur:.1f}s  {W}x{H}（{aspect}）  {info['fps']}fps  镜头 {len(shots)}  字幕段 {len(subs)}", "",
             "| 镜头 | 时间 | 时长 | 重叠字幕(#编号 时间 音高提示) |", "|---|---|---|---|"]
    for s in shots:
        ov = [f"#{u['id']} {u['start']:.1f}-{u['end']:.1f} {u['voice_hint']}{u['f0'] or ''}" for u in subs if s["id"] in u["shots"]]
        lines.append(f"| S{s['id']:02d} | {s['start']:.2f}-{s['end']:.2f} | {s['end'] - s['start']:.1f}s | {'; '.join(ov)} |")
    with open(os.path.join(out, "timeline.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print(f"[6/6] 输出: {out}/analysis.json, timeline.md")
    print("待看的拼图（全部都要看，不要抽查）:")
    for k, v in data["sheets"].items():
        print(f"  {k}: {len(v)} 张 -> {v[0] if v else '-'} ...")


def cmd_zoom(a):
    info = probe(a.video)
    vf = f"fps={a.fps}"
    if a.crop:
        x, y, w, h = a.crop.split(",")
        vf += f",crop={w}:{h}:{x}:{y}"
    vf += ",scale=400:-2,drawtext=text='%{pts\\:hms}':x=4:y=4:fontsize=16:fontcolor=yellow:box=1:boxcolor=black"
    n = int(math.ceil(a.dur * a.fps))
    cols = 4
    rows = max(1, int(math.ceil(n / cols)))
    r = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(a.start), "-t", str(a.dur), "-i", a.video,
             "-vf", vf + f",tile={cols}x{rows}", "-frames:v", "1", a.out])
    if r.returncode:
        # drawtext 不可用时退化为无时间码
        vf2 = vf.split(",drawtext")[0]
        run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-ss", str(a.start), "-t", str(a.dur), "-i", a.video,
             "-vf", vf2 + f",tile={cols}x{rows}", "-frames:v", "1", a.out])
    print(a.out, f"({n} 帧, 从 {a.start}s 起每 {1 / a.fps:.2f}s 一帧)")


def cmd_pitch(a):
    x, sr = load_audio(a.video)
    speech = [(a.start, a.end)]
    ana = a.analysis
    if ana and os.path.exists(ana):
        speech += [(s["start"], s["end"]) for s in json.load(open(ana, encoding="utf-8"))["subtitles"]]
    else:
        # 没有字幕时间轴：取全片最安静的 30% 帧当背景噪声
        n_fft = 1024
        fr = np.array([x[i:i + n_fft] for i in range(0, len(x) - n_fft, n_fft)])
        e = np.sqrt((fr ** 2).mean(1))
        quiet = fr[e <= np.percentile(e, 30)] * np.hanning(n_fft)
        npow = np.mean(np.abs(np.fft.rfft(quiet, axis=1)) ** 2, axis=0) if len(quiet) else None
        f0, vr = f0_hps(x[int(a.start * sr):int(a.end * sr)], sr, npow)
        print(f"{a.start}-{a.end}s  f0≈{f0:.0f}Hz  有声比例 {vr:.2f}  提示: {gender_hint(f0)} (男<165, 女>210, 中间不确定；有背景音乐时仅供参考)")
        return
    npow = noise_profile(x, sr, speech)
    f0, vr = f0_hps(x[int(a.start * sr):int(a.end * sr)], sr, npow)
    print(f"{a.start}-{a.end}s  f0≈{f0:.0f}Hz  有声比例 {vr:.2f}  提示: {gender_hint(f0)} (男<165, 女>210, 中间不确定；有背景音乐时仅供参考)")


def cmd_asr(a):
    """语音转写（可选）。依次尝试 faster_whisper / whisper / funasr；都没有就说明原因退出。"""
    wav = os.path.join(a.out_dir, "audio16k.wav")
    os.makedirs(a.out_dir, exist_ok=True)
    run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-i", a.video, "-ac", "1", "-ar", "16000", wav])
    segs = None
    try:
        from faster_whisper import WhisperModel
        m = WhisperModel(a.model, device="auto", compute_type="int8")
        it, _ = m.transcribe(wav, language=a.lang, vad_filter=True, word_timestamps=False)
        segs = [{"start": round(s.start, 2), "end": round(s.end, 2), "text": s.text.strip()} for s in it]
        eng = "faster-whisper"
    except Exception as e1:
        try:
            import whisper
            m = whisper.load_model(a.model)
            r = m.transcribe(wav, language=a.lang)
            segs = [{"start": round(s["start"], 2), "end": round(s["end"], 2), "text": s["text"].strip()} for s in r["segments"]]
            eng = "whisper"
        except Exception as e2:
            try:
                from funasr import AutoModel
                m = AutoModel(model="paraformer-zh", vad_model="fsmn-vad", punc_model="ct-punc")
                r = m.generate(input=wav, sentence_timestamp=True)
                segs = [{"start": s["start"] / 1000, "end": s["end"] / 1000, "text": s["text"]} for s in r[0].get("sentence_info", [])]
                eng = "funasr"
            except Exception as e3:
                print("ASR 不可用：本环境装不上或下载不了语音识别模型。"
                      f"\n  faster_whisper: {type(e1).__name__}: {str(e1)[:120]}"
                      f"\n  whisper: {type(e2).__name__}: {str(e2)[:120]}"
                      f"\n  funasr: {type(e3).__name__}: {str(e3)[:120]}"
                      "\n→ 有烧录字幕就读字幕；没有字幕就请用户提供台词文本/SRT，或只反推画面与动作。")
                sys.exit(2)
    p = os.path.join(a.out_dir, "asr.json")
    with open(p, "w", encoding="utf-8") as f:
        json.dump({"engine": eng, "segments": segs}, f, ensure_ascii=False, indent=1)
    for s in segs:
        print(f"{s['start']:7.2f}-{s['end']:7.2f}  {s['text']}")
    print(f"→ {p}（引擎 {eng}；同音字/专有名词仍需对照画面与上下文校正）")


# ---------- 剧本渲染 + 自动检查 ----------
_PUNCT = re.compile(r"[\s，。！？、；：,.!?;:…—\-\"'“”‘’（）()《》【】\[\]·~～]")


def _norm(s):
    return _PUNCT.sub("", s or "")


def _load_spec(path):
    txt = open(path, encoding="utf-8").read()
    if path.endswith(".json"):
        return json.loads(txt)
    try:
        import yaml
    except ImportError:
        sys.exit("没有 PyYAML：pip install pyyaml，或者把 spec 写成 .json")
    return yaml.safe_load(txt)


def _fmt(x):
    return f"{x:.1f}"


def _line(ln):
    """台词两种写法：列表 [说话人, 类型, 台词, 补写依据?]，或字典 {who, type, text, fill, orig}。
    orig = 原片原句（改编模式下 text 是改写后的台词，覆盖检查用 orig 对字幕表）。"""
    if isinstance(ln, dict):
        return ln["who"], ln.get("type", "") or "", ln["text"], ln.get("fill") or "", ln.get("orig") or ""
    if len(ln) >= 3:
        return ln[0], ln[1] or "", ln[2], (ln[3] if len(ln) >= 4 else "") or "", ""
    return ln[0], "", ln[1], "", ""


def cmd_render(a):
    sp = _load_spec(a.spec)
    chars = sp.get("characters", {})
    seg_len = float(sp.get("seg_len", 15))
    aspect = sp.get("aspect", "16:9")
    orient = "竖屏" if aspect == "9:16" else "横屏"
    kind = sp.get("style_kind", "3D国漫动画短剧")
    texture = sp.get("style_texture", "UE5引擎电影级实时渲染质感")
    white = set(sp.get("whitelist", []))
    std_cons = sp.get("std_constraints", ["不生成任何字幕或说明文字。", "参考图中的文字、标签、三视图边框不得出现在画面中。"])
    warns, out = [], []
    t_global = float(sp.get("start", 0))
    all_lines, filled_lines, adapt_lines = [], [], []
    design = (sp.get("design") or "").strip()
    overview = []  # (分镜号, 起, 止, 时长, 镜头数, 参考图, 剧情)
    filled_chars = orig_chars = 0
    for n, sg in enumerate(sp["segments"], 1):
        shots = sg["shots"]
        L = float(sg.get("len", seg_len))
        total = round(sum(float(s["t"]) for s in shots), 2)
        if abs(total - L) > 0.05:
            warns.append(f"分镜{n}: 镜头时长合计 {total}s ≠ 段长 {L}s")
        cast = sg.get("cast", list(chars))
        refs = sg.get("refs", cast + [sg.get("scene_ref", "背景")])
        for c in cast:
            if c in chars and c not in refs:
                warns.append(f"分镜{n}: {c} 出镜但不在本段参考图里")
        seg_text = sg.get("scene", "") + "".join(sg.get("positions", {}).values()) + sg.get("state", "") + "".join(
            str(s.get(k, "")) for s in shots for k in ("cam", "frame", "action", "q"))
        for r in refs:
            if design and r not in design:
                warns.append(f"分镜{n}: 参考图「{r}」在剧本开头的参考图提示词里没有对应条目")
            if r in chars and r not in cast:
                warns.append(f"分镜{n}: 参考图放了 {r}，但本段 cast 里没有（不出镜的人别放图）")
            elif r not in chars and r not in seg_text:
                warns.append(f"分镜{n}: 参考图「{r}」在本段画面描述里没用到（多放图会被模型硬塞进画面）")
        o = [f"# 分镜{n}｜预计时长：{_fmt(L)}秒｜全片时间：{t_global:g}—{t_global + L:g}秒", "",
             f"**本段参考图：{'、'.join(refs)}**", f"**镜头数量：{len(shots)}个**", f"**剧情范围：**{sg['range']}", "",
             f"生成一段{_fmt(L)}秒的高质量{aspect}{orient}{kind}视频，{texture}，超清画质，稳定人物建模、服装、材质和场景资产，"
             f"电影级构图与符合当前剧情、人物关系和情绪的自然光影" + ("" if n == 1 and not sp.get("start") else "，保持人物服装和场景与上一段连续") + "。",
             "", "**全局设定：**", f"场景：{sg['scene']}", ""]
        pos = sg.get("positions", {})
        if pos:
            o.append(f"人物站位：画面中只有{'、'.join(cast)}{'这两个人' if len(cast) == 2 else ''}。" if cast else "人物站位：")
            for c, p in pos.items():
                o.append(f"- {c}：{p}")
            o.append("")
        if cast:
            o.append("人物服装严格保持：")
            ov = sg.get("look", {})
            for c in cast:
                look = ov.get(c, chars.get(c, {}).get("look", ""))
                if not look:
                    warns.append(f"分镜{n}: {c} 没有造型描述")
                o.append(f"- **{c}严格参考{c}参考图：**{look}")
            if sg.get("state"):
                o.append(sg["state"])
            o.append("")
        o.append(f"声音基底：{sg.get('sound', sp.get('sound', '室内环境声'))}，无背景音乐。")
        vo = sg.get("voices", {})
        speakers = list(dict.fromkeys([_line(ln)[0] for s in shots for ln in s.get("lines", [])]))
        for c in list(dict.fromkeys(cast + [x for x in speakers if x in sp.get("voices", {}) or x in vo])):
            v = vo.get(c, sp.get("voices", {}).get(c, chars.get(c, {}).get("voice", "")))
            if v:
                o.append(f"- {c}：{v}")
        o.append("")
        t = 0.0
        for k, s in enumerate(shots, 1):
            d = float(s["t"])
            if d < 1.2:
                warns.append(f"分镜{n} 镜头{k}: 只有 {d}s，可能被模型吞掉")
            o.append(f"### 镜头{k}：[{_fmt(t)}—{_fmt(t + d)}秒][{s['cam']}]")
            o.append("")
            if s.get("frame"):
                o += [f"起幅构图：{s['frame']}", ""]
            vis = len(_norm(s.get("frame", "") + s.get("action", "")))
            if vis > 150:
                warns.append(f"分镜{n} 镜头{k}: 画面描述 {vis} 字，超过 150（指令太多模型会丢），删到 2–3 个关键动作")
            if s.get("action"):
                o += [f"调度与动作：{s['action']}", ""]
                for w in re.findall(r"【(.+?)】", s["action"]):
                    if w not in white:
                        warns.append(f"分镜{n} 镜头{k}: 屏幕文字【{w}】不在白名单")
            if s.get("q"):
                o += [f"**Q版内心戏：**{s['q']}", ""]
            nchar = 0
            for ln in s.get("lines", []):
                who, vk, txt, why, orig = _line(ln)
                filled = bool(why)  # 补写依据：原片读不出来、AI 按剧情补的台词
                tag = (vk if "（" in vk else f"（{vk}）") if vk else ""  # 自带括号的类型（如 手机打字（画外音朗读，嘴不动））直接接在名字后
                o += [f"对白/旁白：{who}{tag}：“{txt}”", ""]  # 剧本里不加标记，免得模型把标记当文字生成
                nchar += len(_norm(txt))
                if filled:
                    filled_lines.append((n, k, t, who, txt, str(why)))
                    filled_chars += len(_norm(txt))
                else:
                    all_lines.append(orig or txt)
                    orig_chars += len(_norm(txt))
                    if orig:
                        adapt_lines.append((n, k, who, orig, txt))
                        if sp.get("adapt") and len(_norm(orig)) > 6 and _norm(orig) == _norm(txt):
                            warns.append(f"分镜{n} 镜头{k}: 改编模式下「{txt}」和原片一字不差，换个说法")
                if "待补" in txt:
                    warns.append(f"分镜{n} 镜头{k}: 还有【待补】台词，按剧情补写并在第4项写依据")
                if any(x in vk for x in ("内心", "手机", "嘴不动")) and who in cast and "嘴" not in (s.get("action", "") + vk):
                    warns.append(f"分镜{n} 镜头{k}: {who} 是画外音/内心独白，但没写嘴不动")
            if nchar and nchar / d > 6.2:
                warns.append(f"分镜{n} 镜头{k}: 语速 {nchar / d:.1f} 字/秒（{nchar}字/{d}s），超过 6")
            t += d
        if sg.get("tail"):
            o += [f"段尾承接：{sg['tail']}", ""]
        o.append("片段约束：")
        for c in sg.get("constraints", []) + std_cons:
            o.append(f"- {c}")
        out.append("\n".join(o))
        overview.append((n, t_global, t_global + L, L, len(shots), refs, sg["range"]))
        t_global += L
    text = "\n\n---\n\n".join(out)
    if design:
        text = "# 参考图提示词（先用这些生成参考图，再逐段生成视频）\n\n" + design + "\n\n---\n\n" + text
    # 总览：放在最前面，自动统计
    all_refs = list(dict.fromkeys(r for row in overview for r in row[5]))
    total_len = t_global - float(sp.get("start", 0))
    lens = sorted(set(f"{row[3]:g}" for row in overview))
    ov = ["# 总览", "",
          f"- 全片时长：{total_len:g}秒（{int(total_len // 60)}分{total_len % 60:g}秒）" if total_len >= 60 else f"- 全片时长：{total_len:g}秒",
          f"- 分镜：{len(overview)}段（每段时长：{'、'.join(l + '秒' for l in lens)}），镜头合计 {sum(r[4] for r in overview)} 个",
          f"- 需要的参考图：{len(all_refs)}张（{'、'.join(all_refs)}）",
          "- 用法：先生成参考图，再每次复制一段分镜、带上该段的参考图去生成视频，一次只贴一段。", "",
          "| 分镜 | 全片时间 | 时长 | 镜头 | 本段参考图 | 剧情 |", "|---|---|---|---|---|---|"]
    ov += [f"| 分镜{n} | {a0:g}—{a1:g}秒 | {L:g}秒 | {k} | {'、'.join(r)} | {rg} |" for n, a0, a1, L, k, r, rg in overview]
    text = "\n".join(ov) + "\n\n---\n\n" + text
    if sp.get("global_rules"):
        text += "\n\n---\n\n# 全片统一生成规则\n\n" + sp["global_rules"].strip()
    open(a.out, "w", encoding="utf-8").write(text + "\n")
    # 改编检查：原片的人名、地名、标志性道具不能残留在剧本里
    rename = sp.get("rename") or {}
    for old, new in rename.items():
        if old and old in text:
            warns.append(f"原片的「{old}」还出现在剧本里，应改成「{new}」")
    if rename or adapt_lines:
        rows = ["# 改编对照表（给你核对用，不要贴进视频模型）", ""]
        if rename:
            rows += ["## 替换", "", "| 原片 | 改成 |", "|---|---|"] + [f"| {o} | {nw} |" for o, nw in rename.items()] + [""]
        if adapt_lines:
            rows += ["## 台词改写", "", "| 位置 | 说话人 | 原片台词 | 改写后 |", "|---|---|---|---|"]
            rows += [f"| 分镜{n} 镜头{k} | {w} | {o} | {nw} |" for n, k, w, o, nw in adapt_lines]
        rep_a = os.path.splitext(a.out)[0] + "_改编对照.md"
        open(rep_a, "w", encoding="utf-8").write("\n".join(rows) + "\n")
        print(f"改编对照表：{rep_a}")
    # 台词覆盖检查：字幕表里每一句都要在剧本里、只出现一次、顺序一致
    subs = sp.get("subtitles") or []
    if subs:
        joined = [_norm(x) for x in all_lines]
        flat = "|".join(joined)
        pos_prev = -1
        for s in subs:
            ns = _norm(s)
            if not ns:
                continue
            cnt = flat.count(ns)
            if cnt == 0:
                warns.append(f"台词缺失：「{s}」不在剧本里")
            elif cnt > 1 and len(ns) > 2 and sum(1 for x in subs if _norm(x) == ns) < cnt:
                warns.append(f"台词重复：「{s}」出现 {cnt} 次")
            p = flat.find(ns, max(pos_prev, 0))
            if p == -1 and cnt:
                warns.append(f"台词顺序：「{s}」位置和字幕顺序不一致")
            elif p != -1:
                pos_prev = p
    # AI 补写台词：单独列清单（不写进剧本正文），并控制在“一丢丢”
    if filled_lines:
        ratio = filled_chars / max(1, filled_chars + orig_chars)
        if subs and ratio > 0.25:
            warns.append(f"AI 补写台词占全部台词 {ratio:.0%}，超过 25%：先用 film 拼图/zoom 再读一遍原片，能读出来的别补")
        rep = os.path.splitext(a.out)[0] + "_补写台词.md"
        rows = ["# AI 补写的台词（原片读不出来的地方）", "",
                "剧本正文里没有标记，方便直接贴进模型。想换说法就改这几句，字数尽量不变（口型时长已按它算好）。", "",
                "| 位置 | 说话人 | 补写台词 | 依据 |", "|---|---|---|---|"]
        rows += [f"| 分镜{n} 镜头{k}（{tt:.1f}秒起） | {who} | {txt} | {why} |" for n, k, tt, who, txt, why in filled_lines]
        open(rep, "w", encoding="utf-8").write("\n".join(rows) + "\n")
        print(f"AI 补写台词 {len(filled_lines)} 句（占 {ratio:.0%}），清单：{rep}")
    print(f"已生成 {a.out}：{len(sp['segments'])} 段，合计 {t_global - float(sp.get('start', 0)):g} 秒，{len(text)} 字")
    if warns:
        print(f"⚠ {len(warns)} 条问题：")
        for w in warns:
            print("  -", w)
    else:
        print("✓ 自动检查全部通过（时间轴、语速、画面描述长度、参考图及提示词、白名单、画外音嘴型、台词覆盖与补写、改编残留）")


def main():
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("analyze"); p.add_argument("video"); p.add_argument("out")
    p.add_argument("--scene", type=float, default=0.3); p.add_argument("--band", help="字幕条 y0,y1,x0,x1（原始像素）")
    p.add_argument("--subs-only", action="store_true", help="只重做字幕部分（配合 --band 快速重跑）")
    p.add_argument("--full", action="store_true", help="完整模式：生成每秒拼图、每镜头两帧、完整字幕带胶片（更准但更费 token）")
    p.set_defaults(f=cmd_analyze)
    p = sp.add_parser("zoom"); p.add_argument("video"); p.add_argument("start", type=float); p.add_argument("dur", type=float)
    p.add_argument("out"); p.add_argument("--fps", type=float, default=4); p.add_argument("--crop", help="x,y,w,h")
    p.set_defaults(f=cmd_zoom)
    p = sp.add_parser("pitch"); p.add_argument("video"); p.add_argument("start", type=float); p.add_argument("end", type=float)
    p.add_argument("--analysis", help="analyze 生成的 analysis.json，用它的字幕时段排除人声后估噪声（更准）")
    p.set_defaults(f=cmd_pitch)
    p = sp.add_parser("asr"); p.add_argument("video"); p.add_argument("out_dir")
    p.add_argument("--model", default="small"); p.add_argument("--lang", default="zh")
    p.set_defaults(f=cmd_asr)
    p = sp.add_parser("render"); p.add_argument("spec", help="剧本数据 spec.yaml / spec.json"); p.add_argument("out", help="输出 script.md")
    p.set_defaults(f=cmd_render)
    a = ap.parse_args()
    a.f(a)


if __name__ == "__main__":
    main()
