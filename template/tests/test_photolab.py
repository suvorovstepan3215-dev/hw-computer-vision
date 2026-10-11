# -*- coding: utf-8 -*-
"""Самопроверка ДЗ 1 «Фотолаборатория»: ``pytest tests -q`` из корня вашего репозитория.

Данные синтетические — ни ваших фото, ни камеры тестам не нужны. Пока функция
не написана, её тест падает с ``NotImplementedError``: это и есть список дел.
Те же тесты запускает CI и проверяющий; проходят все — половина работы сделана,
вторая половина — ваши фото, числа и отчёт.
"""
import itertools
import os
import subprocess
import sys

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import photolab as pl  # noqa: E402


# ------------------------------------------------------------- сцены ---
@pytest.fixture(scope="module")
def photo():
    """«Фотография» 240 × 320: градиент, цветные пятна, текстура — тёплый оттенок как у лампы."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:240, :320].astype(np.float32)
    base = 60 + 120 * xx / 320 + 20 * np.sin(yy / 9.0)
    img = np.stack([base * 0.7, base * 0.9, base * 1.15], -1)          # BGR: синего меньше, красного больше
    cv2.circle(img, (90, 120), 40, (40, 200, 60), -1)
    cv2.rectangle(img, (200, 60), (290, 170), (210, 80, 40), -1)
    img += rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.uint8)


def disc_scene():
    """Пурпурный диск на серо-зелёном фоне с крапинками и бликом — как в тестах занятия 3."""
    rng = np.random.default_rng(1)
    img = np.full((240, 320, 3), (120, 140, 110), np.uint8)
    cv2.rectangle(img, (200, 20), (300, 120), (60, 160, 200), -1)
    yy, xx = np.mgrid[:240, :320]
    gt = (xx - 100) ** 2 + (yy - 130) ** 2 <= 45 ** 2
    img[gt] = (200, 40, 235)
    for _ in range(40):
        x, y = int(rng.integers(0, 318)), int(rng.integers(0, 238))
        if not gt[y:y + 2, x:x + 2].any():
            img[y:y + 2, x:x + 2] = (200, 40, 235)
    img[126:134, 96:104] = (250, 250, 250)
    return np.clip(img + rng.normal(0, 3, img.shape), 0, 255).astype(np.uint8), gt


PURPLE_LO, PURPLE_HI = (140, 120, 40), (170, 255, 255)
QUAD = np.float32([[120, 90], [430, 60], [470, 400], [90, 380]])


def document_scene():
    w, h = 200, 283
    page = np.full((h, w, 3), 255, np.uint8)
    for i, t in enumerate(("Computer", "Vision", "IU 2026")):
        cv2.putText(page, t, (12, 50 + 60 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 2, cv2.LINE_AA)
    for k in range(0, w, 40):
        cv2.line(page, (k, 200), (k, h - 1), (0, 0, 0), 2)
    cv2.rectangle(page, (2, 2), (w - 3, h - 3), (0, 0, 0), 2)
    Hp = cv2.getPerspectiveTransform(np.float32([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]]), QUAD)
    bg = np.full((480, 640, 3), (120, 130, 140), np.uint8)
    m = cv2.warpPerspective(np.full((h, w), 255, np.uint8), Hp, (640, 480)) > 127
    bg[m] = cv2.warpPerspective(page, Hp, (640, 480))[m]
    return bg, page


# ---------------------------------------------------------- 1. цвет ---
def test_gamma_lut_matches_formula(photo):
    out = pl.gamma_correct(photo, 0.5)
    assert out.shape == photo.shape and out.dtype == np.uint8
    expect = np.clip(np.round((photo / 255.0) ** 0.5 * 255.0), 0, 255).astype(np.uint8)
    assert np.abs(out.astype(int) - expect).max() <= 1
    assert pl.gamma_correct(photo, 1.0).tolist() == photo.tolist(), "γ = 1 — тождество"
    assert pl.gamma_correct(photo, 0.5).mean() > photo.mean() > pl.gamma_correct(photo, 2.0).mean(), "γ < 1 светлее, γ > 1 темнее"


def test_white_balance_grey_world_equalizes_means(photo):
    out, gains = pl.white_balance(photo, "grey-world")
    assert out.shape == photo.shape and out.dtype == np.uint8 and len(gains) == 3
    means = out.reshape(-1, 3).mean(0)
    assert means.max() - means.min() < 3, "после grey-world средние каналов равны (с точностью до обрезки)"
    assert gains[0] > 1 > gains[2], "синего было мало → усилить, красного много → ослабить"


def test_white_balance_white_patch(photo):
    out, gains = pl.white_balance(photo, "white-patch", percentile=99)
    ref = np.percentile(out.reshape(-1, 3), 99, axis=0)
    assert np.all(ref >= 250), "99-й процентиль каждого канала стал белым"
    with pytest.raises(ValueError):
        pl.white_balance(photo, "magic")


def test_clahe_lab_keeps_colour_and_adds_local_contrast(photo):
    out = pl.clahe_lab(photo, 2.0, 8)
    assert out.shape == photo.shape and out.dtype == np.uint8
    L0 = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    L1 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 0].astype(float)
    dark = L0 <= np.percentile(L0, 20)
    assert L1[dark].std() > L0[dark].std(), "в тенях локальный контраст вырос"
    ab0 = cv2.cvtColor(photo, cv2.COLOR_BGR2LAB)[..., 1:].astype(int); ab1 = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)[..., 1:].astype(int)
    assert np.abs(ab0 - ab1).mean() < 3, "цветовые каналы a, b не тронуты"


# ------------------------------------------------------ 2. хромакей ---
def test_color_mask_and_largest_component():
    img, gt = disc_scene()
    m = pl.color_mask(img, PURPLE_LO, PURPLE_HI)
    assert m.dtype == np.uint8 and set(np.unique(m).tolist()) <= {0, 255}
    lc = pl.largest_component(m)
    assert lc is not None
    mask, stats = lc
    tp = ((mask > 0) & gt).sum()
    assert tp / gt.sum() > 0.95 and tp / max((mask > 0).sum(), 1) > 0.97, "диск найден почти целиком и без крапинок"
    assert stats["components"] == 1 and abs(stats["centroid"][0] - 100) < 2
    assert pl.largest_component(np.zeros((50, 50), np.uint8)) is None
    red = np.full((60, 80, 3), 40, np.uint8); cv2.circle(red, (40, 30), 20, (30, 30, 220), -1)
    assert (pl.color_mask(red, (170, 100, 40), (10, 255, 255)) > 0).sum() > 1000, "красный через 0/179"


def test_chroma_key_composites_over_background():
    img, gt = disc_scene()
    bg = np.full((120, 160, 3), (0, 0, 255), np.uint8)                 # маленький красный фон — растянется
    comp, mask, stats = pl.chroma_key(img, bg, PURPLE_LO, PURPLE_HI, feather=0)
    assert comp.shape == img.shape and mask.shape == img.shape[:2]
    inside, outside = mask > 0, mask == 0
    assert np.abs(comp[inside].astype(int) - img[inside]).max() <= 1, "внутри маски — объект"
    assert np.abs(comp[outside].astype(int) - (0, 0, 255)).max() <= 1, "снаружи — новый фон"
    assert 0 < stats["mask_pct"] < 20 and stats["holes"] == 0


# ------------------------------------------------------ 3. геометрия ---
def test_rotate_keep_all_size_and_no_loss(photo):
    out = pl.rotate_keep_all(photo, 30)
    h, w = photo.shape[:2]; c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    assert abs(out.shape[1] - (w * c + h * s)) <= 1 and abs(out.shape[0] - (w * s + h * c)) <= 1
    ones = pl.rotate_keep_all(np.full((h, w, 3), 255, np.uint8), 30, border=(0, 0, 0))
    kept = (cv2.cvtColor(ones, cv2.COLOR_BGR2GRAY) > 127).sum()
    assert abs(kept - w * h) / (w * h) < 0.01, "ни один пиксель не отрезан"
    assert pl.rotate_keep_all(photo, 0).shape == photo.shape


def test_resize_aa_downscale_is_antialiased(photo):
    noise = np.random.default_rng(2).integers(0, 256, (256, 256, 3)).astype(np.uint8)   # только высокие частоты
    small_area = pl.resize_aa(noise, 0.25, "area"); small_naive = pl.resize_aa(noise, 0.25, "naive")
    assert small_area.shape == (64, 64, 3)
    assert abs(small_area.astype(float).mean() - noise.mean()) < 3, "усреднение окна сохраняет среднее"
    assert small_naive.std() > 1.5 * small_area.std(), "усреднение 16 px давит шум вчетверо, 4 px — только вдвое: без антиалиасинга шум «просачивается»"
    small_blur = pl.resize_aa(noise, 0.25, "blur")
    assert small_blur.std() < small_naive.std(), "размытие до уменьшения — тоже антиалиасинг"
    up = pl.resize_aa(photo, 2.0)
    assert up.shape == (480, 640, 3)
    assert pl.psnr(cv2.resize(photo, (640, 480), interpolation=cv2.INTER_CUBIC), up) > 50, "увеличение — INTER_CUBIC"


def test_order_corners_and_rectify():
    for perm in itertools.permutations(range(4)):
        assert np.array_equal(pl.order_corners(QUAD[list(perm)]), QUAD)
    scene, page = document_scene()
    front, H = pl.rectify(scene, QUAD[[2, 0, 3, 1]], size=(200, 283))
    assert front.shape[:2] == (283, 200)
    ink = lambda im: cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) < 128          # noqa: E731
    assert 100 * (ink(front) == ink(page)).mean() > 97
    f_a4, _ = pl.rectify(scene, QUAD, aspect=210 / 297)
    assert abs(f_a4.shape[1] / f_a4.shape[0] - 210 / 297) < 0.01


# ------------------------------------------------- 4. контактный лист ---
def test_contact_sheet_layout(photo):
    imgs = [photo, cv2.resize(photo, (160, 240)), cv2.cvtColor(photo, cv2.COLOR_BGR2GRAY), photo[:100, :300], photo]
    sheet = pl.contact_sheet(imgs, ["a", "b", "c", "d", "e"], cols=3, cell=(160, 120))
    assert sheet.ndim == 3 and sheet.dtype == np.uint8
    assert sheet.shape[1] == 3 * (160 + 8) + 8 and sheet.shape[0] == 2 * (120 + 26 + 8) + 8
    assert (sheet < 250).any(), "подписи и картинки нарисованы"
    cell = sheet[8:128, 8:168]
    assert (cell < 250).mean() > 0.5, "первая ячейка заполнена картинкой"


# ------------------------------------------------------------ CLI ---
def test_cli_rotate_runs(tmp_path, photo):
    src = tmp_path / "in.png"; dst = tmp_path / "out.png"
    cv2.imwrite(str(src), photo)
    r = subprocess.run([sys.executable, os.path.join(ROOT, "photolab.py"), "rotate", str(src), "--deg", "15", "-o", str(dst)],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    assert r.returncode == 0, r.stderr[-400:]
    assert dst.exists() and '"command": "rotate"' in r.stdout
