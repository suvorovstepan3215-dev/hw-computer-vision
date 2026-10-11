# -*- coding: utf-8 -*-
"""ДЗ 1 «Фотолаборатория» — шаблон.

Запуск:  python photolab.py <команда> [аргументы]      (список команд — README.md)

Одиннадцать функций с пометкой TODO — ваша работа; всё остальное (разбор
аргументов, чтение и запись, гистограммы, отчётные числа) дано готовым и
менять его не нужно. Самопроверка: ``pytest tests -q``. Имена и сигнатуры
функций не меняйте — по ним работают тесты и проверка.
"""
import argparse
import io
import json
import os
import sys

import cv2
import numpy as np

__version__ = "1.0"

# --------------------------------------------------------------- ввод-вывод ---


def imread(path):
    """Читает BGR uint8 и падает с внятной ошибкой (cv2.imread молча даёт None; кириллица в пути)."""
    if not os.path.exists(path):
        raise FileNotFoundError("файл не найден: %s" % path)
    img = cv2.imdecode(np.fromfile(path, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("не удалось декодировать изображение: %s" % path)
    return img


def imwrite(path, img):
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    ok, buf = cv2.imencode(os.path.splitext(path)[1] or ".png", img)
    if not ok:
        raise ValueError("не удалось закодировать %s" % path)
    buf.tofile(path)
    return path


def as_uint8(x):
    """float → uint8 только через обрезку (правило курса: uint8 переполняется молча)."""
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


# ------------------------------------------------------------ 1. цветокоррекция ---


def gamma_correct(img, gamma):
    """Гамма-коррекция: out = 255 · (in / 255) ** gamma, через таблицу на 256 значений.

    gamma < 1 осветляет тени, gamma > 1 затемняет. Таблицу считаем сами (NumPy),
    применяем индексацией — cv2.LUT допустим, но не обязателен.
    """
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 2–3 строки: таблица lut формы (256,) uint8 по формуле, затем lut[img]
    #raise NotImplementedError("TODO 1")
    # ───────────────────────────────────────────────────────────
    lut = as_uint8(255.0*(np.arange(256, dtype=np.float32)/255.0)**gamma)
    return lut[img]


def white_balance(img, method="grey-world", percentile=99.0):
    """Баланс белого без готовых функций: возвращает (кадр, коэффициенты усиления B, G, R).

    grey-world: средний цвет сцены серый → усиление канала = среднее всех средних / среднее канала.
    white-patch: самое яркое (percentile-й процентиль канала) — белое → усиление = 255 / процентиль.
    """
    x = img.astype(np.float32)
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 6–8 строк: средние или процентили по каналам → gains (3,) → x * gains → as_uint8
    #raise NotImplementedError("TODO 2")
    # ───────────────────────────────────────────────────────────
    flat = x.reshape(-1, 3)
    if method == "grey-world":
        means = flat.mean(axis=0) 
        gains = means.mean()/np.maximum(means, 1)
    elif method == "white-patch":
        ref=np.percentile(flat, percentile, axis=0)
        gains = 255.0/np.maximum(ref, 1)
    else:
        raise ValueError("неизвестный метод, доступны только grey-world или же white-patch: %s" % method)

    out = as_uint8(x * gains)
    return out, gains


def equalize_lab(img):
    """Глобальная эквализация яркости: канал L пространства Lab через cv2.equalizeHist (готовая функция)."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    lab[..., 0] = cv2.equalizeHist(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def clahe_lab(img, clip=2.0, tiles=8):
    """CLAHE по каналу L пространства Lab — цвет не трогаем, только яркость."""
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 3–4 строки: BGR → Lab, cv2.createCLAHE(clip, (tiles, tiles)).apply на L, обратно в BGR
    #raise NotImplementedError("TODO 3")
    # ───────────────────────────────────────────────────────────
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles))
    lab[..., 0] = clahe.apply(lab[..., 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def channel_hist(img):
    """Гистограммы каналов без cv2.calcHist: массив (256, C) — сколько пикселей каждого значения."""
    if img.ndim == 2:
        img = img[..., None]
    return np.stack([np.bincount(img[..., c].ravel(), minlength=256) for c in range(img.shape[2])], axis=1)


def tone_stats(img):
    """Числа для отчёта: среднее и процентили яркости (серого) — до и после коррекции."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    p = np.percentile(g, [1, 50, 99])
    return {"mean": round(float(g.mean()), 1), "p1": int(p[0]), "median": int(p[1]), "p99": int(p[2]),
            "clipped_black_pct": round(float(100 * (g == 0).mean()), 2), "clipped_white_pct": round(float(100 * (g == 255).mean()), 2)}


def local_contrast_report(img, out_eq, out_clahe):
    """Почему CLAHE, а не equalizeHist — измерением, как на слайде 24 L3.

    Тени — пиксели, чья исходная яркость ниже 20-го процентиля; света — выше 80-го.
    Возвращает σ в тенях и среднее в светах для обоих методов.
    """
    L0 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0]
    lo, hi = np.percentile(L0, [20, 80])
    shadows, lights = L0 <= lo, L0 >= hi
    rep = {}
    for name, im in (("исходный", img), ("equalizeHist", out_eq), ("CLAHE", out_clahe)):
        L = cv2.cvtColor(im, cv2.COLOR_BGR2LAB)[..., 0]
        rep[name] = {"shadow_std": round(float(L[shadows].std()), 1), "shadow_mean": round(float(L[shadows].mean()), 1),
                     "light_mean": round(float(L[lights].mean()), 1), "light_std": round(float(L[lights].std()), 1)}
    return rep


def save_hist_figure(pairs, path, title=""):
    """Рисунок «до / после»: по строке на каждую пару (подпись, кадр), кривые B, G, R."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(len(pairs), 2, figsize=(9, 2.6 * len(pairs)), squeeze=False)
    for row, (name, im) in enumerate(pairs):
        ax = axes[row][0]; ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB)); ax.set_title(name, fontsize=9); ax.axis("off")
        h = channel_hist(im); ax = axes[row][1]
        for c, col in enumerate(("#0070C0", "#548235", "#C00000")):
            ax.plot(h[:, c], color=col, lw=1)
        ax.set_xlim(0, 255); ax.set_yticks([]); ax.set_title("гистограмма B, G, R", fontsize=9)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


# ---------------------------------------------------------------- 2. хромакей ---


def hsv_box_from(bgr_pixels, margin=10, p=(1, 99)):
    """Границы HSV по пикселям эталона: H и S — процентили ± margin, V — открыт (≥ 40): яркость меняет свет."""
    hsv = cv2.cvtColor(bgr_pixels.reshape(-1, 1, 3), cv2.COLOR_BGR2HSV).reshape(-1, 3)
    lo = np.clip(np.floor(np.percentile(hsv, p[0], axis=0) - margin), 0, 255)
    hi = np.clip(np.ceil(np.percentile(hsv, p[1], axis=0) + margin), 0, 255)
    return (int(lo[0]), int(lo[1]), 40), (min(int(hi[0]), 179), int(hi[1]), 255)


def color_mask(bgr, lo, hi, open_k=5, close_k=9):
    """HSV-маска по границам (lo, hi) + открытие/закрытие; lo[0] > hi[0] — красный через 179 (занятие 3)."""
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 8–10 строк: cvtColor → inRange (две маски и bitwise_or для красного) → MORPH_OPEN → MORPH_CLOSE
    #raise NotImplementedError("TODO 4")
    # ───────────────────────────────────────────────────────────
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)#тоже еще раз взял из лабы
    lo, hi = np.array(lo, np.uint8), np.array(hi, np.uint8)
    if lo[0] <= hi[0]:
        m = cv2.inRange(hsv, lo, hi)
    else:
        m = cv2.bitwise_or(cv2.inRange(hsv, lo, np.array([179, hi[1], hi[2]], np.uint8)),
                           cv2.inRange(hsv, np.array([0, lo[1], lo[2]], np.uint8), hi))
    if open_k:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k)))
    if close_k:
        m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k)))
    return m


def largest_component(mask, min_area=200):
    """Маска крупнейшей компоненты (uint8 0/255) и её статистика; None — если компонент нет или она мала."""
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 6–8 строк: connectedComponentsWithStats; пропустить фон (строка 0); argmax по площади; min_area
    #raise NotImplementedError("TODO 5")
    # ───────────────────────────────────────────────────────────
    n, labels, stats, cent = cv2.connectedComponentsWithStats(mask)#попытка номер 3
    if n < 2:
        return None
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    if stats[i, cv2.CC_STAT_AREA] < min_area:
        return None
    largest = np.where(labels == i, 255, 0).astype(np.uint8)
    x, y, w, h, a = (int(v) for v in stats[i])
    return largest, {"components": 1, "centroid": [float(cent[i][0]), float(cent[i][1])],
                     "bbox": [x, y, w, h], "area": a}


def chroma_key(img, bg, lo, hi, open_k=5, close_k=9, feather=2.0):
    """Объект по цвету → на новый фон. Возвращает (композит, маска объекта, статистика)."""
    h, w = img.shape[:2]
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_AREA if bg.shape[0] > h else cv2.INTER_LINEAR)
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 6–9 строк: color_mask → largest_component (None → ошибка); альфа = маска/255 (GaussianBlur σ = feather для мягкого края); композит img·α + bg·(1 − α)
    #raise NotImplementedError("TODO 6")
    # ───────────────────────────────────────────────────────────
    raw = color_mask(img, lo, hi, 0, 0)
    full = color_mask(img, lo, hi, open_k, close_k)
    raw_n, clean_n = [cv2.connectedComponents(m)[0] - 1 for m in (raw, full)]
    res = largest_component(full, 200)
    if res is None:
        raise ValueError("object not search:почините свой min_area или ослабьте границы. Сейчас все полетело.")
    mask, st = res
    base = mask.astype(np.float32) / 255.0
    alpha = cv2.GaussianBlur(base, (0, 0), feather)[..., None] if feather > 0 else base[..., None] #тц.
    comp = as_uint8(img.astype(np.float32) * alpha + bg.astype(np.float32) * (1 - alpha))
    holes = len(cv2.findContours(mask, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)[0]) - 1
    stats = {**st, "components": int(clean_n), "raw_components": int(raw_n),
             "holes": holes, "mask_pct": round(100 * float((mask > 0).mean()), 2)}
    return comp, mask, stats


# ---------------------------------------------------------------- 3. геометрия ---


def rotate_keep_all(img, deg, border=(255, 255, 255)):
    """Поворот на deg градусов вокруг центра с холстом по повёрнутым углам — ни один пиксель не отрезан."""
    h, w = img.shape[:2]
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 6–8 строк: getRotationMatrix2D; nw, nh = w·|cos| + h·|sin|, w·|sin| + h·|cos|; сдвиг M[:, 2] на разницу центров; warpAffine (nw, nh)
    #raise NotImplementedError("TODO 7")
    # ───────────────────────────────────────────────────────────
    M=cv2.getRotationMatrix2D((w/2.0, h/2.0), deg, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(round(w*cos+h*sin)), int(round(w*sin+h*cos))
    M[0, 2] += nw/2.0-w/2.0
    M[1, 2] += nh/2.0 - h/2.0
    return cv2.warpAffine(img, M, (nw, nh), borderValue=border)


def resize_aa(img, scale, method="area"):
    """Масштабирование с правильным антиалиасингом.

    Уменьшение: ``area`` — cv2.resize(INTER_AREA); ``blur`` — GaussianBlur σ ≈ 0.4 / scale, затем INTER_LINEAR;
    ``naive`` — INTER_LINEAR без размытия (так делать нельзя: алиасинг, это контроль для измерения).
    Увеличение — всегда INTER_CUBIC: там алиасинга нет, а кубическая даёт +2 дБ на фото (L4).
    """
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 6–9 строк: scale ≥ 1 → INTER_CUBIC; иначе по method: INTER_AREA / GaussianBlur(σ = 0.4 / scale) + INTER_LINEAR / INTER_LINEAR
    #raise NotImplementedError("TODO 8")
    # ───────────────────────────────────────────────────────────
    if scale >= 1:
      return cv2.resize(img, size, interpolation=cv2.INTER_CUBIC)#база
    if method == "area":
      return cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    if method == "blur":
      sigma = 0.4/scale
      blurred = cv2.GaussianBlur(img, (0, 0), sigma)
      return cv2.resize(blurred, size, interpolation=cv2.INTER_LINEAR)   
    if method == "naive":# действительно не native...
        return cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
    raise ValueError("неизвестный метод, доступны лишь area, blur, naive. Ваш прошлый метод: %s" % method)    


def lanczos_reference(img, scale):
    """Независимый эталон уменьшения — Lanczos из Pillow (как в демо 2.2 лекции 1 и 2.4 лекции 4)."""
    from PIL import Image
    h, w = img.shape[:2]
    size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return np.asarray(Image.fromarray(img[..., ::-1]).resize(size, Image.LANCZOS))[..., ::-1].copy()


def psnr(a, b):
    a = a.astype(np.float32) / 255.0; b = b.astype(np.float32) / 255.0
    e = float(np.mean((a - b) ** 2))
    return float("inf") if e == 0 else float(10 * np.log10(1.0 / e))


def order_corners(pts):
    """Четыре точки в любом порядке → float32 (4, 2) в порядке tl, tr, br, bl (занятие 4)."""
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 3–4 строки: суммы x + y (tl — минимум, br — максимум), разности y − x (tr — минимум, bl — максимум)
    #raise NotImplementedError("TODO 9")
    # ───────────────────────────────────────────────────────────
    SSS=np.asarray(pts, np.float32).reshape(4, 2)
    s = SSS.sum(axis=1)
    d = SSS[:, 1]-SSS[:, 0]
    return np.float32([SSS[np.argmin(s)], SSS[np.argmin(d)], SSS[np.argmax(s)],SSS[np.argmax(d)]])


def rectify(img, corners, size=None, aspect=None):
    """Четыре угла → фронтальный вид и H (занятие 4): размер из size / aspect (W / H) / длин сторон в кадре."""
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 8–11 строк: order_corners; W = max(верх, низ); H = W / aspect или max(лево, право); dst; getPerspectiveTransform; warpPerspective (W, H)
    #raise NotImplementedError("TODO 10")
    # ───────────────────────────────────────────────────────────
    c = order_corners(corners)#точки
    d = lambda i, j: float(np.linalg.norm(c[i] - c[j]))
    top, bottom = d(0, 1), d(3, 2) #прямоугольник для наших точек.
    left, right = d(0, 3), d(1, 2)

    if size is not None:
        W, Hh = int(size[0]), int(size[1])
    else:
        W = int(round(max(top, bottom))) 
        Hh = int(round(W/aspect)) if aspect is not None else int(round(max(left, right)))
    dst = np.float32([[0, 0], [W-1, 0], [W-1, Hh-1], [0, Hh-1]])#3
    H = cv2.getPerspectiveTransform(c,dst)#4
    front = cv2.warpPerspective(img, H, (W, Hh), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)#5 #ладно. эту функцию все же самому писать не захотелось
    return front, H


def pick_corners(img, title="4 klika po uglam, Enter - gotovo, r - zanovo"):
    """Окно OpenCV: четыре клика по углам → float32 (4, 2) в координатах кадра (только локально)."""
    pts, scale = [], min(1.0, 1200.0 / img.shape[1])

    def on_click(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x / scale, y / scale))

    cv2.namedWindow(title); cv2.setMouseCallback(title, on_click)
    while True:
        view = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        for p in pts:
            cv2.circle(view, (int(p[0] * scale), int(p[1] * scale)), 6, (0, 200, 255), -1)
        cv2.imshow(title, view)
        key = cv2.waitKey(30) & 0xFF
        if key == ord("r"):
            pts.clear()
        if key in (13, 10, ord("q")) or cv2.getWindowProperty(title, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()
    if len(pts) != 4:
        raise ValueError("нужно ровно 4 клика, получено %d" % len(pts))
    return np.float32(pts)


# ----------------------------------------------------------- 4. контактный лист ---


def contact_sheet(images, captions, cols=None, cell=(320, 240), pad=8, caption_h=26):
    """Мозаика N изображений с подписями: каждое вписано в ячейку cell с сохранением пропорций."""
    n = len(images)
    cols = cols or int(np.ceil(np.sqrt(n)))
    rows = int(np.ceil(n / cols))
    cw, ch = cell
    sheet = np.full((rows * (ch + caption_h + pad) + pad, cols * (cw + pad) + pad, 3), 255, np.uint8)
    # ───────────────────────── ВАШ КОД ─────────────────────────
    # 8–12 строк: для каждого кадра — масштаб min(cw/w, ch/h), resize (INTER_AREA при уменьшении), положить по центру ячейки срезом, cv2.putText подпись под ячейкой
    #raise NotImplementedError("TODO 11")
    # ───────────────────────────────────────────────────────────
    for i, (im, cap) in enumerate(zip(images, captions)):
        r, c = divmod(i, cols)
        x0 = pad+c*(cw + pad)
        y0 = pad+r*(ch+caption_h+ pad)
        ih, iw = im.shape[:2]
        s = min(cw/iw, ch/ih)
        nw, nh = max(1, int(round(iw*s))), max(1, int(round(ih*s)))#эх.
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        thumb = cv2.resize(im, (nw, nh), interpolation=interp)
        if thumb.ndim == 2:#на случай серых картинок.
            thumb = cv2.cvtColor(thumb, cv2.COLOR_GRAY2BGR)
        ox = x0 + (cw-nw)//2
        oy = y0 + (ch-nh)//2
        sheet[oy:oy + nh, ox:ox + nw] = thumb
        cv2.putText(sheet, cap, (x0 + 2, y0 + ch + caption_h - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return sheet


# --------------------------------------------------------------------- CLI ---


def _ints(s):
    return tuple(int(v) for v in s.split(","))


def _log(**kw):
    print(json.dumps(kw, ensure_ascii=False))


def cmd_gamma(a):
    img = imread(a.image); out = gamma_correct(img, a.gamma)
    imwrite(a.out, out)
    if a.hist:
        save_hist_figure([("до", img), ("после, γ = %g" % a.gamma, out)], a.hist)
    _log(command="gamma", gamma=a.gamma, before=tone_stats(img), after=tone_stats(out), out=a.out)


def cmd_wb(a):
    img = imread(a.image); out, gains = white_balance(img, a.method, a.percentile)
    imwrite(a.out, out)
    if a.hist:
        save_hist_figure([("до", img), ("после, %s" % a.method, out)], a.hist)
    means = lambda im: [round(float(v), 1) for v in im.reshape(-1, 3).mean(0)]   # noqa: E731
    _log(command="wb", method=a.method, gains_bgr=[round(float(g), 3) for g in gains], means_before_bgr=means(img), means_after_bgr=means(out), out=a.out)


def cmd_clahe(a):
    img = imread(a.image); out = clahe_lab(img, a.clip, a.tiles); eq = equalize_lab(img)
    imwrite(a.out, out)
    if a.compare:
        imwrite(a.compare, np.hstack([img, eq, out]))
    if a.hist:
        save_hist_figure([("до", img), ("equalizeHist", eq), ("CLAHE clip %g, tiles %d" % (a.clip, a.tiles), out)], a.hist)
    _log(command="clahe", clip=a.clip, tiles=a.tiles, local_contrast=local_contrast_report(img, eq, out), out=a.out)


def cmd_chromakey(a):
    img = imread(a.image); bg = imread(a.bg)
    if a.sample:
        x, y, w, h = a.sample
        lo, hi = hsv_box_from(img[y:y + h, x:x + w].reshape(-1, 3))
    else:
        lo, hi = a.lo, a.hi
    comp, mask, stats = chroma_key(img, bg, lo, hi, a.open, a.close, a.feather)
    imwrite(a.out, comp)
    if a.mask:
        imwrite(a.mask, mask)
    _log(command="chromakey", lo=list(lo), hi=list(hi), **stats, out=a.out)


def cmd_rotate(a):
    img = imread(a.image); out = rotate_keep_all(img, a.deg)
    imwrite(a.out, out)
    h, w = img.shape[:2]; nh, nw = out.shape[:2]
    _log(command="rotate", deg=a.deg, size_in=[w, h], size_out=[nw, nh], canvas_empty_pct=round(100 * (1 - w * h / (nw * nh)), 1), out=a.out)


def cmd_resize(a):
    img = imread(a.image); out = resize_aa(img, a.scale, a.method)
    imwrite(a.out, out)
    rep = {"command": "resize", "scale": a.scale, "method": a.method, "size_out": [out.shape[1], out.shape[0]], "out": a.out}
    if a.report and a.scale < 1:
        ref = lanczos_reference(img, a.scale)
        rep["psnr_vs_lanczos_db"] = {m: round(psnr(ref, resize_aa(img, a.scale, m)), 1) for m in ("naive", "blur", "area")}
    _log(**rep)


def cmd_rectify(a):
    img = imread(a.image)
    corners = pick_corners(img) if a.clicks else np.float32(a.corners).reshape(4, 2)
    size = tuple(int(v) for v in a.size.lower().split("x")) if a.size else None
    front, H = rectify(img, corners, size, a.aspect)
    imwrite(a.out, front)
    c = order_corners(corners)
    side = lambda p, q: round(float(np.linalg.norm(p - q)), 1)   # noqa: E731
    _log(command="rectify", corners_ordered=c.tolist(), sides_px={"top": side(c[0], c[1]), "right": side(c[1], c[2]), "bottom": side(c[3], c[2]), "left": side(c[0], c[3])},
         size_out=[front.shape[1], front.shape[0]], aspect_out=round(front.shape[1] / front.shape[0], 3), out=a.out)


def cmd_sheet(a):
    images = [imread(p) for p in a.images]
    caps = ["%s %dx%d" % (os.path.basename(p), im.shape[1], im.shape[0]) for p, im in zip(a.images, images)]
    sheet = contact_sheet(images, caps, a.cols, (a.cell_w, a.cell_h))
    imwrite(a.out, sheet)
    _log(command="sheet", n=len(images), cols=a.cols or int(np.ceil(np.sqrt(len(images)))), size_out=[sheet.shape[1], sheet.shape[0]], out=a.out)


def build_parser():
    p = argparse.ArgumentParser(prog="photolab", description="ДЗ 1 «Фотолаборатория»: цвет, хромакей, геометрия, контактный лист")
    sub = p.add_subparsers(dest="command", required=True)

    s = sub.add_parser("gamma", help="гамма-коррекция"); s.add_argument("image"); s.add_argument("--gamma", type=float, required=True)
    s.add_argument("-o", "--out", required=True); s.add_argument("--hist", help="PNG с гистограммами до/после"); s.set_defaults(fn=cmd_gamma)

    s = sub.add_parser("wb", help="баланс белого"); s.add_argument("image"); s.add_argument("--method", choices=("grey-world", "white-patch"), default="grey-world")
    s.add_argument("--percentile", type=float, default=99.0); s.add_argument("-o", "--out", required=True); s.add_argument("--hist"); s.set_defaults(fn=cmd_wb)

    s = sub.add_parser("clahe", help="CLAHE по яркости"); s.add_argument("image"); s.add_argument("--clip", type=float, default=2.0); s.add_argument("--tiles", type=int, default=8)
    s.add_argument("-o", "--out", required=True); s.add_argument("--compare", help="PNG: исходный | equalizeHist | CLAHE"); s.add_argument("--hist"); s.set_defaults(fn=cmd_clahe)

    s = sub.add_parser("chromakey", help="объект по цвету на новый фон"); s.add_argument("image"); s.add_argument("--bg", required=True)
    s.add_argument("--lo", type=_ints, help="H,S,V нижняя граница"); s.add_argument("--hi", type=_ints, help="H,S,V верхняя граница")
    s.add_argument("--sample", type=_ints, help="x,y,w,h — прямоугольник на объекте: границы снимаются с него")
    s.add_argument("--open", type=int, default=5); s.add_argument("--close", type=int, default=9); s.add_argument("--feather", type=float, default=2.0)
    s.add_argument("-o", "--out", required=True); s.add_argument("--mask", help="PNG маски объекта"); s.set_defaults(fn=cmd_chromakey)

    s = sub.add_parser("rotate", help="поворот с сохранением всего кадра"); s.add_argument("image"); s.add_argument("--deg", type=float, required=True)
    s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_rotate)

    s = sub.add_parser("resize", help="масштабирование с антиалиасингом"); s.add_argument("image"); s.add_argument("--scale", type=float, required=True)
    s.add_argument("--method", choices=("area", "blur", "naive"), default="area"); s.add_argument("--report", action="store_true", help="PSNR трёх методов к эталону Lanczos")
    s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_resize)

    s = sub.add_parser("rectify", help="выпрямление по 4 углам"); s.add_argument("image")
    s.add_argument("--corners", type=lambda v: [float(t) for t in v.split(",")], help="x1,y1,x2,y2,x3,y3,x4,y4 в любом порядке")
    s.add_argument("--clicks", action="store_true", help="задать углы кликами в окне OpenCV")
    s.add_argument("--aspect", type=float, help="W / H объекта: A4 — 0.707"); s.add_argument("--size", help="WxH выхода, например 640x905")
    s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_rectify)

    s = sub.add_parser("sheet", help="контактный лист"); s.add_argument("images", nargs="+"); s.add_argument("--cols", type=int)
    s.add_argument("--cell-w", type=int, default=320); s.add_argument("--cell-h", type=int, default=240); s.add_argument("-o", "--out", required=True); s.set_defaults(fn=cmd_sheet)
    return p


def main(argv=None):
    a = build_parser().parse_args(argv)
    if a.command == "rectify" and not a.clicks and not a.corners:
        raise SystemExit("rectify: нужны --corners x1,y1,…,x4,y4 или --clicks")
    if a.command == "chromakey" and not a.sample and not (a.lo and a.hi):
        raise SystemExit("chromakey: нужны --lo и --hi или --sample x,y,w,h")
    a.fn(a)
    return 0


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8") if hasattr(sys.stdout, "reconfigure") else None
    sys.exit(main())
