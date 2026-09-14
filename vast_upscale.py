#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
vast_upscale.py — автоматический AI-апскейл видео (лица, тела, реальная съёмка)
на арендованном GPU в облаке vast.ai.

Что делает скрипт (по шагам):
  1. Проверяет зависимости на этой машине (Ubuntu 24.04) и предлагает установить недостающие.
  2. Сопрягает машину с vast.ai (API-ключ, SSH-ключ, баланс) и с Docker Hub (логин, токен,
     репозиторий для предсобранного образа) — с подробными инструкциями.
  3. Принимает видеофайл(ы) или директорию, анализирует их (ffprobe) и конвертирует
     в формат, безопасный для апскейлера (H.264, постоянный fps, yuv420p, чётные стороны).
  4. Ищет на vast.ai самую выгодную конфигурацию GPU: для каждого оффера оценивает
     полное время run (запуск контейнера + установка + загрузка + обработка + скачивание)
     и его стоимость (аренда × это время + диск + трафик) и по умолчанию берёт самый
     дешёвый run целиком (--optimize cost; есть ratio/balanced/speed); показывает таблицу и меню
     (выбрать номер, подробности, ещё офферы, сменить фильтры/критерий, обновить поиск);
     при необходимости делает interruptible-бид, либо берёт on-demand.
  5. Разворачивает в контейнере SeedVR2 (ByteDance, ICLR 2026 — лучший открытый
     видео-апскейлер для реальной съёмки с людьми) через CLI numz/ComfyUI-SeedVR2_VideoUpscaler,
     умеет ждать пока контейнер реально заработает, и переезжает на другой оффер при сбое.
  6. Загружает видео, апскейлит до 1080p (короткая сторона; можно задать другое),
     скачивает результат рядом с исходником под именем <имя>_1080p.mp4 (со звуком).
  7. На каждом этапе показывает прогресс и оценку оставшегося времени.
  8. По окончании скачивания ВСЕХ результатов останавливает и удаляет контейнер.
     При обрыве скачивания докачивает по частям (rsync --partial) до тех пор,
     пока контейнер жив.

Дополнительно (v1.3):
  • Предсобранный образ: SeedVR2 + зависимости + статический ffmpeg + ВЕСА МОДЕЛИ вшиваются
    в образ, который собирается НА ЭТОЙ МАШИНЕ БЕЗ Docker (и работает в WSL2): скрипт сам
    формирует слои OCI (tar.gz), монтирует слои базового образа pytorch/pytorch в ваш
    репозиторий Docker Hub через Registry API (ничего не скачивая), грузит только новые слои
    (chunked, с докачкой) и публикует манифест — всё до выбора и покупки инстанса
    (--prebuild local, по умолчанию при наличии доступа к Docker Hub; нужен только pip).
    Рабочий инстанс стартует готовым: без apt/pip/скачивания весов. Альтернативы:
    --prebuild strict (снапшот на арендуемом инстансе), inline, never.
  • Проверка реальной полосы каждого поднятого инстанса (Cloudflare/HF CDN, ssh);
    медленнее --min-link (100 Мбит/с) — удаляется, берётся следующий оффер.
  • Блочная обработка: файл режется на блоки под ~8 минут GPU-времени (+контекст спереди,
    как в потоковом режиме SeedVR2); блок — единица очереди, переезда, докачки и --resume;
    блоки склеиваются без перекодирования.
  • Пул машин по DLPerf/$: каждые --pool-scan (15 с) листинг офферов; лучший берётся в пул,
    если в --pool-improve (1.3×) лучше худшей машины пула; худшие выбывают, освободившись
    (work stealing между машинами); в пуле всегда ≥ --pool-min готовых машин.
  • Требования к ресурсам (VRAM/RAM/диск) считаются по файлам и разрешению выхода и
    служат критериями отбора офферов (gpu_ram, cpu_ram, disk_space + проверка модели).
  • Память: batch подбирается по реально свободной VRAM (nvidia-smi) и мегапикселям
    выхода, чанк потокового режима — по лимиту RAM контейнера (cgroup); OOM VRAM →
    щадящая лестница, OOM RAM (SIGKILL) → чанк вдвое меньше.
  • Фоновый режим: --daemon запускает задачу в фоне; --attach подключается к логу и
    живому прогрессу (Ctrl+C — отсоединиться), --status — список, --stop — остановить
    с удалением инстансов.

Быстрый старт:
    python3 vast_upscale.py --check                 # шаги 1–2: зависимости + сопряжение
    python3 vast_upscale.py video.mp4               # всё остальное автоматически
    python3 vast_upscale.py ./videos/ --target 1440 --model 7b --optimize speed
    python3 vast_upscale.py video.mp4 --dry-run     # только план и цены, без заказа

Полезное:
    python3 vast_upscale.py --list-instances        # мои инстансы на vast.ai
    python3 vast_upscale.py --destroy-instance ID   # удалить инстанс вручную
    python3 vast_upscale.py video.mp4 --resume      # продолжить прерванный запуск

Зависимости: Python >= 3.10, ffmpeg/ffprobe, ssh, ssh-keygen, scp, rsync.
Только стандартная библиотека Python (ничего ставить через pip не нужно).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import getpass
import hashlib
import io
import json
import math
import os
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import tarfile
import textwrap
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

VERSION = "1.8.0"

# ----------------------------------------------------------------------------
# Константы: апскейлер, образ, модели
# ----------------------------------------------------------------------------

# SeedVR2 CLI (Apache-2.0). Закреплён на проверенном коммите v2.5.24 (24.12.2025).
SEEDVR2_REPO = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_COMMIT = "4490bd1f482e026674543386bb2a4d176da245b9"

# Docker-образ по умолчанию: официальный PyTorch 2.8.0 + CUDA 12.8 (проверенная связка
# для SeedVR2 v2.5.x; поддерживает Blackwell/RTX 50xx). ~4 ГБ сжатых слоёв.
DEFAULT_IMAGE = "pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime"
IMAGE_SIZE_BYTES = 4_000_000_000  # для оценки времени скачивания образа

# Веса моделей (HF). Размеры и sha256 взяты из HF API и реестра моделей SeedVR2 CLI.
MODEL_FILES: Dict[str, Dict[str, Any]] = {
    "seedvr2_ema_3b_fp16.safetensors": dict(
        repo="numz/SeedVR2_comfyUI", size=6_783_018_808,
        sha256="2fd0e03a3dad24e07086750360727ca437de4ecd456f769856e960ae93e2b304"),
    "seedvr2_ema_3b_fp8_e4m3fn.safetensors": dict(
        repo="numz/SeedVR2_comfyUI", size=3_391_544_696,
        sha256="3bf1e43ebedd570e7e7a0b1b60d6a02e105978f505c8128a241cde99a8240cff"),
    "seedvr2_ema_7b_fp16.safetensors": dict(
        repo="numz/SeedVR2_comfyUI", size=16_479_334_424,
        sha256="7b8241aa957606ab6cfb66edabc96d43234f9819c5392b44d2492d9f0b0bbe4a"),
    "seedvr2_ema_7b_sharp_fp16.safetensors": dict(
        repo="numz/SeedVR2_comfyUI", size=16_479_334_424,
        sha256="20a93e01ff24beaeebc5de4e4e5be924359606c356c9c51509fba245bd2d77dd"),
    "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors": dict(
        repo="AInVFX/SeedVR2_comfyUI", size=8_466_296_338,
        sha256="3d68b5ec0b295ae28092e355c8cad870edd00b817b26587d0cb8f9dd2df19bb2"),
    "seedvr2_ema_7b_sharp_fp8_e4m3fn_mixed_block35_fp16.safetensors": dict(
        repo="AInVFX/SeedVR2_comfyUI", size=8_466_296_338,
        sha256="0d2c5b8be0fda94351149c5115da26aef4f4932a7a2a928c6f184dda9186e0be"),
    "ema_vae_fp16.safetensors": dict(
        repo="numz/SeedVR2_comfyUI", size=501_324_814,
        sha256="20678548f420d98d26f11442d3528f8b8c94e57ee046ef93dbb7633da8612ca1"),
}
VAE_FILE = "ema_vae_fp16.safetensors"

# Модель потребления VRAM SeedVR2 (консервативно, по опубликованным замерам): base — веса DiT+VAE+буферы (ГБ);
# k — ГБ на кадр на мегапиксель ВЫХОДА при тайлинге VAE (без тайлинга пик декодера ~1.5 ГБ/МП/кадр).
# Используется и локально (требования к офферам), и в runner.py внутри контейнера (подбор batch).
VRAM_MODEL: Dict[str, Tuple[float, float]] = {"3b_fp16": (9.0, 0.70), "3b_fp8": (5.5, 0.70), "7b_fp16": (18.5, 0.90), "7b_fp8": (10.5, 0.90)}
VRAM_K_UNTILED = 1.5
RAM_BASE_GB = 6.0            # CUDA-контекст, torch, ОС контейнера
MIN_BATCH = 5                # минимальный разумный батч (4n+1)


def vram_profile_local(dit: str) -> Tuple[float, float]:
    size = "7b" if "7b" in dit else "3b"
    prec = "fp8" if ("fp8" in dit or dit.endswith(".gguf")) else "fp16"
    return VRAM_MODEL[f"{size}_{prec}"]


def output_dims(width: int, height: int, target: int, max_long: int = 0) -> Tuple[float, float]:
    scale = max(1.0, target / max(1, min(width, height)))
    ow, oh = width * scale, height * scale
    if max_long and max(ow, oh) > max_long:
        f = max_long / max(ow, oh)
        ow, oh = ow * f, oh * f
    return ow, oh


@dataclass
class Requirements:
    """Оценка ресурсов под задание: используется как критерий отбора офферов и в таблице."""
    out_mp: float            # максимальные мегапиксели выхода среди файлов
    per_frame_mb: float      # RAM на кадр (вход+выход float32 ×2.5)
    disk_gb: float
    vram_gb: Dict[str, float] = field(default_factory=dict)   # по файлу DiT
    ram_gb: Dict[str, float] = field(default_factory=dict)

    def vram_for(self, dit: str) -> float:
        return self.vram_gb.get(dit) or vram_required(dit, self.out_mp)

    def ram_for(self, dit: str) -> float:
        return self.ram_gb.get(dit) or ram_required(dit, self.per_frame_mb)

    @property
    def vram_min(self) -> float:
        return min(self.vram_gb.values()) if self.vram_gb else 0.0

    @property
    def ram_min(self) -> float:
        return min(self.ram_gb.values()) if self.ram_gb else 0.0


def vram_required(dit: str, out_mp: float, batch: int = MIN_BATCH) -> float:
    base, k = vram_profile_local(dit)
    return math.ceil(base + 1.0 + batch * k * out_mp)


def ram_required(dit: str, per_frame_mb: float, batch: int = MIN_BATCH) -> float:
    weights_gb = MODEL_FILES.get(dit, {}).get("size", 7e9) / 1e9
    chunk = 8 * batch + 1                       # штатный чанк потокового режима
    return math.ceil(RAM_BASE_GB + weights_gb + chunk * per_frame_mb / 1024 * 1.2)

# ----------------------------------------------------------------------------
# Профили GPU: ожидаемая скорость SeedVR2-3B fp16 (сек/кадр при выходе 1080p, батч
# подобран под VRAM). Числа — оценки на основе опубликованных замеров (numz README,
# ai-upscaler-benchmark, статья SeedVR2); после первого файла скрипт калибрует их
# по фактической скорости. min_cuda — минимальная версия CUDA драйвера хоста.
# ----------------------------------------------------------------------------
GPU_PROFILES: Dict[str, Dict[str, float]] = {
    "RTX 3090":            dict(spf=4.0, min_cuda=12.4),
    "RTX 3090 Ti":         dict(spf=3.6, min_cuda=12.4),
    "RTX 4090":            dict(spf=2.2, min_cuda=12.4),
    "RTX 4090D":           dict(spf=2.6, min_cuda=12.4),
    "RTX 5090":            dict(spf=1.5, min_cuda=12.8),
    "RTX 5080":            dict(spf=3.5, min_cuda=12.8),
    "RTX A5000":           dict(spf=4.5, min_cuda=12.4),
    "RTX A5500":           dict(spf=4.0, min_cuda=12.4),
    "RTX A6000":           dict(spf=3.0, min_cuda=12.4),
    "A40":                 dict(spf=3.2, min_cuda=12.4),
    "L4":                  dict(spf=5.0, min_cuda=12.4),
    "L40":                 dict(spf=1.9, min_cuda=12.4),
    "L40S":                dict(spf=1.6, min_cuda=12.4),
    "RTX 5000Ada":         dict(spf=2.2, min_cuda=12.4),
    "RTX 6000Ada":         dict(spf=1.6, min_cuda=12.4),
    "RTX PRO 5000":        dict(spf=1.4, min_cuda=12.8),
    "RTX PRO 6000 WS":     dict(spf=1.0, min_cuda=12.8),
    "RTX PRO 6000 S":      dict(spf=1.0, min_cuda=12.8),
    "RTX PRO 6000 Max-Q":  dict(spf=1.1, min_cuda=12.8),
    "A100 PCIE":           dict(spf=1.3, min_cuda=12.4),
    "A100 SXM4":           dict(spf=1.2, min_cuda=12.4),
    "A800 PCIE":           dict(spf=1.3, min_cuda=12.4),
    "H100 PCIE":           dict(spf=0.7, min_cuda=12.4),
    "H100 SXM":            dict(spf=0.6, min_cuda=12.4),
    "H100 NVL":            dict(spf=0.65, min_cuda=12.4),
    "H200":                dict(spf=0.5, min_cuda=12.4),
    "H200 NVL":            dict(spf=0.5, min_cuda=12.4),
    "B200":                dict(spf=0.4, min_cuda=12.8),
    "B300":                dict(spf=0.35, min_cuda=12.8),
}
DEFAULT_GPU_SET = [g for g in GPU_PROFILES]  # все известные, VRAM отфильтруется по --min-vram

VIDEO_EXTS = {".mp4", ".mkv", ".mov", ".avi", ".webm", ".m4v", ".ts", ".mts", ".m2ts",
              ".wmv", ".flv", ".mpg", ".mpeg", ".3gp", ".vob", ".mxf"}

VAST_URL = "https://console.vast.ai"
VAST_KEY_FILE = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config")) / "vastai" / "vast_api_key"
VAST_KEY_FILE_LEGACY = Path.home() / ".vast_api_key"

REMOTE_ROOT = "/workspace/vu"   # рабочая директория внутри контейнера
REMOTE_ROOT_TEMPLATE: Optional[str] = None   # напр. "/tmp/vu-{iid}" — только для тестов на одном хосте
REMOTE_OPT = "/opt/vu"          # софт + веса (эта директория входит в снапшот предсобранного образа)
REMOTE_OPT_TEMPLATE: Optional[str] = None


def remote_root_for(instance_id) -> str:
    return REMOTE_ROOT_TEMPLATE.format(iid=instance_id) if REMOTE_ROOT_TEMPLATE else REMOTE_ROOT

# ----------------------------------------------------------------------------
# Вывод: цвета, форматирование, прогресс-бары с ETA
# ----------------------------------------------------------------------------
_IS_TTY = sys.stdout.isatty() and os.environ.get("TERM", "dumb") != "dumb" and not os.environ.get("NO_COLOR")


def _c(code: str, s: str) -> str:
    return f"\033[{code}m{s}\033[0m" if _IS_TTY else s


def bold(s): return _c("1", s)
def green(s): return _c("32", s)
def yellow(s): return _c("33", s)
def red(s): return _c("31", s)
def cyan(s): return _c("36", s)
def dim(s): return _c("2", s)


_print_lock = threading.RLock()
_line_open = False  # открыта ли строка прогресса (для корректного переноса)
# Потоки-воркеры (параллельные инстансы) печатают строки с префиксом, а прогресс-бары
# отдают на «доску» оркестратора вместо терминала — см. Orchestrator.
_tls = threading.local()
_progress_file: Optional[Path] = None      # фоновый режим: сюда пишется последняя строка прогресса
_progress_last = [0.0]


def _write_progress(line: str):
    if _progress_file is None:
        return
    now = time.time()
    if now - _progress_last[0] < 1.0 and line:
        return
    _progress_last[0] = now
    try:
        tmp = _progress_file.with_suffix(".tmp")
        tmp.write_text(json.dumps({"line": line, "ts": now}, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_progress_file)
    except OSError:
        pass


def _raw_print(s: str, end: str = "\n"):
    global _line_open
    prefix = getattr(_tls, "prefix", "")
    if prefix and s.strip():
        s = re.sub(r"^(\s*)", lambda m: m.group(1) + prefix, s, count=1)
    with _print_lock:
        if _line_open and end == "\n":
            sys.stdout.write("\n")
            _line_open = False
        sys.stdout.write(s + end)
        sys.stdout.flush()


def info(msg: str): _raw_print(f"  {msg}")
def ok(msg: str): _raw_print(f"  {green('✔')} {msg}")
def warn(msg: str): _raw_print(f"  {yellow('!')} {msg}")
def err(msg: str): _raw_print(f"  {red('✖')} {msg}")
def note(msg: str): _raw_print(f"    {dim(msg)}")


VERBOSE = False


def debug(msg: str):
    if VERBOSE:
        _raw_print(dim(f"    [debug] {msg}"))


def stage(n: int, total: int, title: str):
    _raw_print("")
    _raw_print(bold(cyan(f"━━ Этап {n}/{total}: {title} ")) + dim(_dt.datetime.now().strftime("%H:%M:%S")))


def fmt_time(sec: Optional[float]) -> str:
    if sec is None or sec != sec or sec == float("inf") or sec > 99 * 3600:
        return "??:??"
    sec = max(0, int(round(sec)))
    h, rem = divmod(sec, 3600)
    m, s = divmod(rem, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def fmt_bytes(n: Optional[float]) -> str:
    if n is None:
        return "?"
    n = float(n)
    for unit in ("Б", "КБ", "МБ", "ГБ", "ТБ"):
        if abs(n) < 1024 or unit == "ТБ":
            return f"{n:.0f} {unit}" if unit == "Б" else f"{n:.1f} {unit}"
        n /= 1024
    return f"{n:.1f} ТБ"


def fmt_money(x: Optional[float]) -> str:
    return "?" if x is None else f"${x:.3f}" if x < 1 else f"${x:.2f}"


def ask_yes_no(question: str, default: bool = True, auto: Optional[bool] = None) -> bool:
    """Вопрос да/нет. auto=True/False — ответ без интерактива (режим --yes)."""
    if auto is not None:
        _raw_print(f"  {question} [{'Y/n' if default else 'y/N'}] → {'да' if auto else 'нет'} (авто)")
        return auto
    if not sys.stdin.isatty():
        _raw_print(f"  {question} → {'да' if default else 'нет'} (нет терминала, ответ по умолчанию)")
        return default
    suffix = " [Y/n] " if default else " [y/N] "
    while True:
        try:
            a = input(f"  {question}{suffix}").strip().lower()
        except EOFError:
            return default
        if not a:
            return default
        if a in ("y", "yes", "д", "да"):
            return True
        if a in ("n", "no", "н", "нет"):
            return False


class ProgressBar:
    """Однострочный прогресс-бар с ETA (EMA по скорости + запасная оценка по плану)."""

    def __init__(self, total: Optional[float], desc: str, unit: str = "", est_total_sec: Optional[float] = None,
                 bytes_mode: bool = False, width: int = 28, on_render: Optional[Callable[[str], None]] = None):
        self.total = total
        self.desc = desc
        self.unit = unit
        self.est_total_sec = est_total_sec
        self.bytes_mode = bytes_mode
        self.width = width
        if on_render is None:
            on_render = getattr(_tls, "board_cb", None)   # поток-воркер: прогресс на доску оркестратора
        self.on_render = on_render   # если задан — бар не печатает сам, а отдаёт компактную строку наружу
        self.line = ""
        self.compact = ""
        self.n = 0.0
        self.t0 = time.time()
        self.t_last = self.t0
        self.n_last = 0.0
        self.rate = None  # ед./сек, сглаженная
        self.extra = ""
        self._last_render = 0.0
        self._closed = False
        self.render(force=True)

    def set(self, n: float, extra: Optional[str] = None):
        now = time.time()
        if self.total is not None:
            n = min(n, self.total)
        dt = now - self.t_last
        if dt >= 1.0:
            inst = (n - self.n_last) / dt
            self.rate = inst if self.rate is None else 0.7 * self.rate + 0.3 * inst
            self.t_last, self.n_last = now, n
        self.n = n
        if extra is not None:
            self.extra = extra
        self.render()

    def update(self, dn: float, extra: Optional[str] = None):
        self.set(self.n + dn, extra)

    def eta(self) -> Optional[float]:
        if self._closed:
            return 0.0
        if self.total is None or self.total <= 0:
            if self.est_total_sec:
                return max(0.0, self.est_total_sec - (time.time() - self.t0))
            return None
        remaining = self.total - self.n
        elapsed = time.time() - self.t0
        # по сглаженной скорости, но не ниже 25% средней (иначе при паузе ETA улетает в бесконечность);
        # если данных ещё нет — по плану
        avg = self.n / elapsed if (self.n > 0 and elapsed > 2) else 0.0
        rate = max(self.rate or 0.0, 0.25 * avg)
        if rate > 0:
            eta = remaining / rate
            if self.est_total_sec:
                eta = min(eta, 5 * max(self.est_total_sec, elapsed))
            return eta
        if self.est_total_sec:
            return max(0.0, self.est_total_sec - elapsed)
        return None

    def _fmt_n(self, x: float) -> str:
        if self.bytes_mode:
            return fmt_bytes(x)
        return f"{x:.0f}{self.unit}"

    def render(self, force: bool = False):
        global _line_open
        now = time.time()
        if not force and now - self._last_render < (0.2 if _IS_TTY else 10.0):
            return
        self._last_render = now
        elapsed = now - self.t0
        if self.total and self.total > 0:
            frac = min(1.0, self.n / self.total)
            filled = int(frac * self.width)
            bar = "█" * filled + "░" * (self.width - filled)
            pct = f"{frac * 100:5.1f}%"
            cnt = "" if self.unit == "%" else f"{self._fmt_n(self.n)}/{self._fmt_n(self.total)}"
        else:
            # неизвестный объём — "бегунок"
            pos = int(elapsed * 4) % self.width
            bar = "".join("▓" if abs(i - pos) <= 1 else "░" for i in range(self.width))
            pct = "  ...  "
            cnt = self._fmt_n(self.n) if self.n else ""
        rate = ""
        if self.rate:
            rate = (fmt_bytes(self.rate) + "/с") if self.bytes_mode else f"{self.rate:.2f}{self.unit}/с"
        line = f"  {self.desc} |{bar}| {pct} {cnt} {rate} прошло {fmt_time(elapsed)} осталось ~{fmt_time(self.eta())} {self.extra}"
        self.line = line.strip()
        self.compact = f"{self.desc} {pct.strip()}{(' ' + rate) if rate else ''} ~{fmt_time(self.eta())}"
        if self.on_render is not None:
            self.on_render(self.compact)
            return
        _write_progress(self.line)
        with _print_lock:
            if _IS_TTY:
                cols = shutil.get_terminal_size((120, 20)).columns
                line = line[:cols - 1]
                sys.stdout.write("\r" + line + "\033[K")
                _line_open = True
            else:
                # без терминала: строка только при изменении показателей (без «прошло» — иначе дубли каждые 10 с)
                key = f"{self.desc}|{pct}|{cnt}|{self.extra}"
                if key == getattr(self, "_printed_key", None) and not force:
                    return
                self._printed_key = key
                sys.stdout.write(line + "\n")
            sys.stdout.flush()

    def close(self, final_msg: Optional[str] = None):
        global _line_open
        if self._closed:
            return
        self._closed = True
        if self.total:
            self.n = self.total
        if self.on_render is not None:
            self.on_render("")
        else:
            self.render(force=True)
            with _print_lock:
                if _IS_TTY and _line_open:
                    sys.stdout.write("\n")
                    _line_open = False
        if final_msg:
            ok(final_msg)


class Spinner:
    """Индикатор ожидания без известного объёма (ожидание контейнера, SSH и т.п.).
    Перед текстом — время последнего сообщения (когда статус изменился); одинаковый статус повторно не печатается:
    в терминале строка обновляется на месте, без терминала (лог, --daemon) строка выводится только при изменении."""

    def __init__(self, desc: str, est_total_sec: Optional[float] = None):
        self.desc = desc
        self.est = est_total_sec
        self.t0 = time.time()
        self.status = ""
        self.status_since = self.t0          # когда статус (последнее сообщение) изменился
        self._last = 0.0
        self._printed = ""

    def stamp(self) -> str:
        return _dt.datetime.fromtimestamp(self.status_since).strftime("%H:%M:%S")

    def tick(self, status: str = ""):
        global _line_open
        now = time.time()
        status = status or self.status
        if status != self.status:
            self.status, self.status_since = status, now
        if now - self._last < (0.5 if _IS_TTY else 5.0):
            return
        self._last = now
        el = now - self.t0
        text = f"{self.stamp()} {self.desc}: {self.status}"
        cb = getattr(_tls, "board_cb", None)
        if cb is not None:
            cb(text)
            return
        if not _IS_TTY:
            if text == self._printed:           # тот же статус — не дублировать
                _write_progress(text)
                return
            self._printed = text
            _write_progress(text)
            with _print_lock:
                sys.stdout.write(f"  · {text}\n")
                sys.stdout.flush()
            return
        eta = f" (обычно ~{fmt_time(self.est)}, осталось ~{fmt_time(max(0, self.est - el))})" if self.est else ""
        frames = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
        ch = frames[int(now * 8) % len(frames)]
        line = f"  {ch} {text} — прошло {fmt_time(el)}{eta}"
        _write_progress(line.strip())
        with _print_lock:
            cols = shutil.get_terminal_size((120, 20)).columns
            sys.stdout.write("\r" + line[:cols - 1] + "\033[K")
            _line_open = True
            sys.stdout.flush()

    def close(self, msg: Optional[str] = None):
        global _line_open
        cb = getattr(_tls, "board_cb", None)
        if cb is not None:
            cb("")
        with _print_lock:
            if _IS_TTY and _line_open:
                sys.stdout.write("\n")
                _line_open = False
        if msg:
            ok(msg)


def run(cmd: List[str], check: bool = True, capture: bool = True, timeout: Optional[float] = None,
        input_text: Optional[str] = None, env: Optional[dict] = None) -> subprocess.CompletedProcess:
    debug("$ " + " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, check=check, capture_output=capture, text=True, timeout=timeout,
                          input=input_text, env=env)


class UserAbort(Exception):
    pass


class FatalError(Exception):
    pass

# ----------------------------------------------------------------------------
# Шаг 1. Проверка зависимостей (Ubuntu 24.04) и предложение установки
# ----------------------------------------------------------------------------
APT_PACKAGES = {  # команда -> apt-пакет
    "ffmpeg": "ffmpeg",
    "ffprobe": "ffmpeg",
    "ssh": "openssh-client",
    "ssh-keygen": "openssh-client",
    "scp": "openssh-client",
    "rsync": "rsync",
}


def os_description() -> str:
    try:
        data = Path("/etc/os-release").read_text(encoding="utf-8")
        kv = dict(re.findall(r'^(\w+)="?([^"\n]*)"?$', data, re.M))
        return f"{kv.get('NAME', '?')} {kv.get('VERSION_ID', '')} ({platform.machine()})"
    except OSError:
        return f"{platform.system()} {platform.release()} ({platform.machine()})"


def check_dependencies(auto_yes: Optional[bool], allow_install: bool = True) -> None:
    """Проверяет python/ffmpeg/ssh/rsync; предлагает `sudo apt-get install` для недостающего."""
    stage_ok = True
    info(f"ОС: {os_description()}; Python {platform.python_version()} ({sys.executable})")
    if sys.version_info < (3, 10):
        raise FatalError("Нужен Python 3.10 или новее (в Ubuntu 24.04 по умолчанию 3.12).")
    missing_cmds = [c for c in APT_PACKAGES if shutil.which(c) is None]
    for c in APT_PACKAGES:
        if c in missing_cmds:
            err(f"{c}: не найден")
        else:
            ok(f"{c}: {shutil.which(c)}")
    # версия ffmpeg (нужна >= 5.1 для -fps_mode; в 24.04 стоит 6.1)
    if "ffmpeg" not in missing_cmds:
        try:
            v = run(["ffmpeg", "-version"], check=False).stdout.splitlines()[0]
            m = re.search(r"ffmpeg version (\S+)", v)
            ver = m.group(1) if m else "?"
            note(f"ffmpeg {ver}")
            mm = re.match(r"n?(\d+)\.(\d+)", ver)
            if mm and (int(mm.group(1)), int(mm.group(2))) < (5, 1):
                warn("ffmpeg старше 5.1 — вместо -fps_mode будет использован -vsync (совместимость).")
        except Exception:
            pass
    if shutil.which("vastai"):
        note("vastai CLI найден (не обязателен: скрипт работает с REST API напрямую).")
    note("сборка предсобранного образа не требует Docker: слои и манифест формируются самим скриптом (нужен только pip).")
    if missing_cmds:
        pkgs = sorted({APT_PACKAGES[c] for c in missing_cmds})
        cmd = ["sudo", "apt-get", "install", "-y"] + pkgs
        warn("Не хватает: " + ", ".join(missing_cmds))
        info("Команда установки (Ubuntu 24.04):")
        info("    sudo apt-get update && " + " ".join(cmd))
        if not allow_install:
            raise FatalError("Установите недостающие пакеты и запустите снова (или уберите --no-install).")
        if ask_yes_no("Установить недостающие пакеты сейчас?", default=True, auto=auto_yes):
            try:
                subprocess.run(["sudo", "apt-get", "update"], check=False)
                subprocess.run(cmd, check=True)
            except (subprocess.CalledProcessError, FileNotFoundError) as e:
                raise FatalError(f"Установка не удалась: {e}. Установите вручную и повторите.")
            still = [c for c in missing_cmds if shutil.which(c) is None]
            if still:
                raise FatalError("После установки всё ещё не найдены: " + ", ".join(still))
            ok("Пакеты установлены.")
        else:
            raise FatalError("Без этих инструментов работа невозможна.")
    if stage_ok:
        ok("Все зависимости на месте.")


# ----------------------------------------------------------------------------
# Клиент vast.ai (REST API v0, без сторонних библиотек)
# Эндпоинты и формат тел запросов сверены с исходниками пакета vastai 1.7.0.
# ----------------------------------------------------------------------------
class VastAPIError(Exception):
    def __init__(self, status: int, msg: str, body: Any = None):
        super().__init__(f"HTTP {status}: {msg}")
        self.status = status
        self.msg = msg
        self.body = body


class VastClient:
    def __init__(self, api_key: str, base_url: str = VAST_URL, timeout: float = 60.0):
        self.api_key = api_key
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self._lock = threading.Lock()

    def _request(self, method: str, path: str, params: Optional[dict] = None, body: Any = None,
                 retries: int = 6) -> Any:
        url = self.base_url + ("/api/v0" + path if not path.startswith("/api/") else path)
        if params:
            url += ("&" if "?" in url else "?") + urllib.parse.urlencode(params)
        data = None
        headers = {"Authorization": "Bearer " + self.api_key, "Accept": "application/json"}
        if body is not None or method in ("POST", "PUT"):
            data = json.dumps(body if body is not None else {}).encode()
            headers["Content-Type"] = "application/json"
        delay = 2.0
        last_exc: Optional[Exception] = None
        for attempt in range(retries):
            req = urllib.request.Request(url, data=data, method=method, headers=headers)
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    raw = resp.read().decode("utf-8", "replace")
                    return json.loads(raw) if raw.strip() else {}
            except urllib.error.HTTPError as e:
                raw = e.read().decode("utf-8", "replace")
                try:
                    parsed = json.loads(raw)
                except ValueError:
                    parsed = {"msg": raw[:300]}
                msg = parsed.get("msg") or parsed.get("error") or parsed.get("detail") or str(parsed)[:300]
                # 429 — слишком часто (vast не присылает Retry-After), 5xx — временные
                if e.code == 429 or e.code >= 500:
                    last_exc = VastAPIError(e.code, msg, parsed)
                    debug(f"{method} {path} -> {e.code} {msg}; повтор через {delay:.0f}с")
                    time.sleep(delay)
                    delay = min(delay * 2, 60)
                    continue
                raise VastAPIError(e.code, msg, parsed)
            except (urllib.error.URLError, socket.timeout, ConnectionError, TimeoutError) as e:
                last_exc = e
                debug(f"{method} {path} сетевая ошибка: {e}; повтор через {delay:.0f}с")
                time.sleep(delay)
                delay = min(delay * 2, 60)
        raise FatalError(f"vast.ai API недоступен после {retries} попыток: {last_exc}")

    # --- аккаунт / ключи ---
    def show_user(self) -> dict:
        return self._request("GET", "/users/current")

    def show_ssh_keys(self) -> list:
        r = self._request("GET", "/ssh/")
        return r if isinstance(r, list) else (r.get("ssh_keys") or r.get("keys") or [])

    def create_ssh_key(self, pubkey: str) -> dict:
        return self._request("POST", "/ssh/", body={"ssh_key": pubkey})

    def attach_ssh(self, instance_id: int, pubkey: str) -> dict:
        return self._request("POST", f"/instances/{instance_id}/ssh/", body={"ssh_key": pubkey})

    # --- офферы ---
    def search_offers(self, query: dict, offer_type: str = "on-demand", order=None, limit: int = 100,
                      storage_gb: float = 5.0) -> list:
        q = {"verified": {"eq": True}, "external": {"eq": False}, "rentable": {"eq": True},
             "rented": {"eq": False}}
        q.update(query)
        q["order"] = order or [["dph_total", "asc"]]
        q["type"] = "bid" if offer_type in ("bid", "interruptible") else offer_type
        q["limit"] = int(limit)
        q["allocated_storage"] = float(storage_gb)
        r = self._request("POST", "/bundles/", body=q)
        return r.get("offers", []) if isinstance(r, dict) else r

    # --- инстансы ---
    def create_instance(self, offer_id: int, image: str, disk_gb: float, env: dict, onstart: Optional[str],
                        label: str, price: Optional[float], runtype: str = "ssh_direc ssh_proxy",
                        cancel_unavail: bool = True, image_login: Optional[str] = None) -> dict:
        body = {
            "client_id": "me", "image": image, "env": env, "price": price, "disk": float(disk_gb),
            "label": label, "extra": None, "onstart": onstart, "image_login": image_login,
            "python_utf8": False, "lang_utf8": False, "use_jupyter_lab": False, "jupyter_dir": None,
            "force": False, "cancel_unavail": cancel_unavail, "template_hash_id": None, "user": None,
            "runtype": runtype,
        }
        return self._request("PUT", f"/asks/{offer_id}/", body=body)

    def show_instance(self, instance_id: int) -> Optional[dict]:
        r = self._request("GET", f"/instances/{instance_id}/", params={"owner": "me"})
        return r.get("instances") if isinstance(r, dict) else None

    def show_instances(self) -> list:
        r = self._request("GET", "/instances/", params={"owner": "me"})
        return r.get("instances", []) if isinstance(r, dict) else r

    def stop_instance(self, instance_id: int) -> dict:
        return self._request("PUT", f"/instances/{instance_id}/", body={"state": "stopped"})

    def start_instance(self, instance_id: int) -> dict:
        return self._request("PUT", f"/instances/{instance_id}/", body={"state": "running"})

    def destroy_instance(self, instance_id: int) -> dict:
        return self._request("DELETE", f"/instances/{instance_id}/", body={})

    def change_bid(self, instance_id: int, price: float) -> dict:
        return self._request("PUT", f"/instances/bid_price/{instance_id}/", body={"client_id": "me", "price": price})

    def label_instance(self, instance_id: int, label: str) -> dict:
        return self._request("PUT", f"/instances/{instance_id}/", body={"label": label})

    def take_snapshot(self, instance_id: int, repo_with_tag: str, docker_user: str, docker_pass: str,
                      registry: str = "docker.io", pause: bool = True) -> dict:
        """Снимок контейнера → push в ваш репозиторий (vastai take snapshot). Делает хост, не ваша машина."""
        body = {"id": instance_id, "container_registry": registry, "personal_repo": repo_with_tag,
                "docker_login_user": docker_user, "docker_login_pass": docker_pass, "pause": "true" if pause else "false"}
        return self._request("POST", f"/instances/take_snapshot/{instance_id}/", body=body)


# ----------------------------------------------------------------------------
# Docker Hub: проверка наличия предсобранного образа (публичного или приватного)
# ----------------------------------------------------------------------------
class DockerHub:
    API = "https://hub.docker.com/v2"

    def __init__(self, user: Optional[str], password: Optional[str]):
        self.user, self.password = user, password
        self.token: Optional[str] = None
        self._login_failed = False

    def _req(self, path: str, body: Any = None, auth: bool = True) -> Tuple[int, Any]:
        url = self.API + path
        data = json.dumps(body).encode() if body is not None else None
        headers = {"Accept": "application/json"}
        if data is not None:
            headers["Content-Type"] = "application/json"
        if auth and self.token:
            headers["Authorization"] = "JWT " + self.token
        req = urllib.request.Request(url, data=data, method="POST" if data is not None else "GET", headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=40) as r:
                raw = r.read().decode("utf-8", "replace")
                return r.status, (json.loads(raw) if raw.strip() else {})
        except urllib.error.HTTPError as e:
            raw = e.read().decode("utf-8", "replace")
            try:
                return e.code, json.loads(raw)
            except ValueError:
                return e.code, {"raw": raw[:200]}
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError) as e:
            return 0, {"error": str(e)}

    def login(self) -> bool:
        if self.token:
            return True
        if not (self.user and self.password) or self._login_failed:
            return False
        st, r = self._req("/users/login", {"username": self.user, "password": self.password}, auth=False)
        if st == 200 and r.get("token"):
            self.token = r["token"]
            return True
        self._login_failed = True
        debug(f"Docker Hub login: HTTP {st} {str(r)[:150]}")
        return False

    def tag_info(self, repo: str, tag: str) -> Optional[dict]:
        """Информация о теге или None. repo = 'user/name'."""
        st, r = self._req(f"/repositories/{repo}/tags/{tag}")
        if st in (401, 403, 404) and not self.token and self.login():
            st, r = self._req(f"/repositories/{repo}/tags/{tag}")
        return r if st == 200 else None

    def tags(self, repo: str) -> List[dict]:
        st, r = self._req(f"/repositories/{repo}/tags?page_size=25&ordering=last_updated")
        if st in (401, 403, 404) and not self.token and self.login():
            st, r = self._req(f"/repositories/{repo}/tags?page_size=25&ordering=last_updated")
        return (r.get("results") or []) if st == 200 and isinstance(r, dict) else []

    def repo_private(self, repo: str) -> Optional[bool]:
        st, r = self._req(f"/repositories/{repo}")
        if st in (401, 403, 404) and not self.token and self.login():
            st, r = self._req(f"/repositories/{repo}")
        return bool(r.get("is_private")) if st == 200 and isinstance(r, dict) else None


def _docker_size(txt: str) -> float:
    m = re.match(r"([\d.]+)\s*([kKMG]?)B", txt.strip())
    if not m:
        return 0.0
    return float(m.group(1)) * {"": 1, "k": 1e3, "K": 1e3, "M": 1e6, "G": 1e9}[m.group(2)]


def _iso_to_ts(s: Optional[str]) -> float:
    if not s:
        return 0.0
    try:
        return _dt.datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return 0.0


def model_short(dit: str) -> str:
    m = dit.replace("seedvr2_ema_", "").replace(".safetensors", "")
    m = m.replace("fp8_e4m3fn_mixed_block35_fp16", "fp8mixed").replace("fp8_e4m3fn", "fp8").replace("_", "-")
    return m


def prebuilt_tag(dit: str, base_image: str, py_ver: str = "3.11") -> str:
    """Тег образа: модель, коммит SeedVR2, hash базового образа и версия Python, под которую собраны зависимости."""
    return (f"seedvr2-{model_short(dit)}-{SEEDVR2_COMMIT[:7]}-{hashlib.sha1(base_image.encode()).hexdigest()[:6]}"
            f"-py{py_ver.replace('.', '')}")


IMAGE_CACHE_FILE = VAST_KEY_FILE.parent / "vast_upscale_images.json"
DEFAULT_CACHE_DIR = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")) / "vast_upscale"
LOCAL_CACHE_DIR = DEFAULT_CACHE_DIR        # уточняется у пользователя на этапе 2 (см. ensure_cache_dir), хранится в SETTINGS_FILE
SETTINGS_FILE = VAST_KEY_FILE.parent / "vast_upscale_settings.json"


def load_settings() -> dict:
    try:
        return json.loads(SETTINGS_FILE.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def save_settings(d: dict) -> None:
    try:
        SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
        SETTINGS_FILE.write_text(json.dumps(d, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as e:
        warn(f"Не удалось сохранить настройки в {SETTINGS_FILE}: {e}")


def download_with_resume(url: str, dest: Path, size: int, sha256: str, desc: str) -> None:
    """Скачивание на ЭТУ машину с докачкой (HTTP Range) и проверкой sha256; маркер .sha256ok — не проверять повторно.
    size=0 — размер неизвестен (без проверки размера/хеша, если sha256 пуст)."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    marker = Path(str(dest) + ".sha256ok")
    if dest.exists() and (size == 0 or dest.stat().st_size == size) and marker.exists():
        ok(f"{dest.name}: уже скачан и проверен")
        return
    part = Path(str(dest) + ".part")
    bar = ProgressBar(size or None, desc, bytes_mode=True)
    for attempt in range(1, 9):
        have = part.stat().st_size if part.exists() else 0
        if size and have > size:
            part.unlink()
            have = 0
        done = size and have == size
        if not done:
            req = urllib.request.Request(url, headers={"Range": f"bytes={have}-"} if have else {})
            try:
                with urllib.request.urlopen(req, timeout=60) as r:
                    if have and r.status != 206:
                        part.unlink()                     # сервер не умеет Range — с начала
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        since_sync = 0
                        while True:
                            chunk = r.read(4 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            since_sync += len(chunk)
                            if since_sync >= (256 << 20):
                                drop_page_cache(f, 0, have, sync=True)
                                since_sync = 0
                            bar.set(have)
                        drop_page_cache(f, sync=True)
            except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                bar.render(force=True)
                warn(f"{dest.name}: обрыв ({e}); повтор {attempt}/8 через {5 * attempt} с")
                time.sleep(5 * attempt)
                continue
        if size and have != size:
            warn(f"{dest.name}: размер {have} ≠ {size}, повтор")
            continue
        if sha256:
            bar.set(have, extra="проверка sha256")
            h = sha256_of(part)
            if h != sha256:
                warn(f"{dest.name}: sha256 не совпал — скачиваю заново")
                part.unlink()
                continue
        part.replace(dest)
        marker.write_text(sha256 or "nohash")
        bar.close(f"{dest.name}: скачан{' и проверен' if sha256 else ''} ({fmt_bytes(have)})")
        return
    bar.close()
    raise FatalError(f"Не удалось скачать {url}")


# ----------------------------------------------------------------------------
# Сборка образа БЕЗ Docker: клиент Registry API v2 (Docker Hub) и сборщик слоёв OCI на чистом Python.
# Базовый образ не скачивается — его слои монтируются в ваш репозиторий (cross-repo mount);
# грузятся только новые слои (SeedVR2 + зависимости + ffmpeg, веса модели).
# ----------------------------------------------------------------------------
MT_DOCKER_MANIFEST = "application/vnd.docker.distribution.manifest.v2+json"
MT_DOCKER_LIST = "application/vnd.docker.distribution.manifest.list.v2+json"
MT_OCI_MANIFEST = "application/vnd.oci.image.manifest.v1+json"
MT_OCI_INDEX = "application/vnd.oci.image.index.v1+json"
MT_DOCKER_CONFIG = "application/vnd.docker.container.image.v1+json"
MT_OCI_CONFIG = "application/vnd.oci.image.config.v1+json"
MT_DOCKER_LAYER = "application/vnd.docker.image.rootfs.diff.tar.gzip"
MT_OCI_LAYER = "application/vnd.oci.image.layer.v1.tar+gzip"
MANIFEST_ACCEPT = ", ".join([MT_DOCKER_MANIFEST, MT_DOCKER_LIST, MT_OCI_MANIFEST, MT_OCI_INDEX])
REGISTRY_URL = "https://registry-1.docker.io"
REGISTRY_AUTH_URL = "https://auth.docker.io/token"
REGISTRY_SERVICE = "registry.docker.io"
UPLOAD_CHUNK = 96 << 20   # байт за один PATCH при загрузке готового файла (запасной путь для слоёв базового образа)
STREAM_PIECE = 4 << 20    # кусок потоковой загрузки слоя: единица очереди генератор→сеть, хеша сверки и докачки


class RegistryError(FatalError):
    pass


class PrebuiltBroken(FatalError):
    """Предсобранный образ не прошёл проверку на инстансе (установка на арендованной машине запрещена)."""
    def __init__(self, msg: str, py_have: Optional[str] = None, reasons: str = ""):
        super().__init__(msg)
        self.py_have, self.reasons = py_have, reasons


class RegistryFatal(RegistryError):
    """Ошибка реестра, которую повтор не исправит (4xx кроме 401/404-сессии): не повторять."""


def parse_image_ref(image: str) -> Tuple[str, str]:
    """'pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime' → ('pytorch/pytorch', tag); 'ubuntu' → ('library/ubuntu','latest')."""
    name, _, tag = image.partition("@") if "@" in image else (image, "", "")
    if not tag:
        if ":" in name.rsplit("/", 1)[-1]:
            name, _, tag = name.rpartition(":")
        else:
            tag = "latest"
    name = name.replace("docker.io/", "", 1)
    if "/" not in name:
        name = "library/" + name
    return name, tag


class RegistryClient:
    """Минимальный клиент Docker Registry HTTP API v2 с bearer-авторизацией (Docker Hub)."""

    def __init__(self, user: Optional[str] = None, password: Optional[str] = None, registry: Optional[str] = None,
                 auth_url: Optional[str] = None, service: Optional[str] = None):
        self.user, self.password = user, password
        self.registry = (registry or REGISTRY_URL).rstrip("/")
        self.auth_url, self.service = auth_url or REGISTRY_AUTH_URL, service or REGISTRY_SERVICE
        self._tokens: Dict[str, str] = {}

    # ---- авторизация ----
    def token(self, scopes: List[str]) -> Optional[str]:
        key = "|".join(scopes)
        if key in self._tokens:
            return self._tokens[key]
        q = "&".join(["service=" + urllib.parse.quote(self.service)] + ["scope=" + urllib.parse.quote(sc) for sc in scopes])
        req = urllib.request.Request(self.auth_url + "?" + q)
        if self.user and self.password:
            import base64
            req.add_header("Authorization", "Basic " + base64.b64encode(f"{self.user}:{self.password}".encode()).decode())
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                d = json.loads(r.read().decode())
        except urllib.error.HTTPError as e:
            raise RegistryError(f"auth {e.code}: {e.read().decode('utf-8', 'replace')[:200]}")
        except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError, ValueError) as e:
            raise RegistryError(f"auth: {e}")
        tok = d.get("token") or d.get("access_token")
        if tok:
            self._tokens[key] = tok
        return tok

    def check_push_access(self, repo: str) -> bool:
        try:
            return bool(self.token([f"repository:{repo}:pull,push"]))
        except RegistryError as e:
            debug(f"registry auth: {e}")
            return False

    # ---- запросы ----
    def _req(self, method: str, url: str, scopes: List[str], headers: Optional[dict] = None, data=None,
             timeout: float = 120, raw_resp: bool = False):
        if url.startswith("/"):
            url = self.registry + url
        h = dict(headers or {})
        tok = self.token(scopes)
        if tok:
            h["Authorization"] = "Bearer " + tok
        if hasattr(data, "seek"):
            data.seek(0)                      # тело-файл: с начала при каждой (повторной) отправке
        req = urllib.request.Request(url, data=data, method=method, headers=h)
        try:
            r = urllib.request.urlopen(req, timeout=timeout)
        except urllib.error.HTTPError as e:
            if e.code == 401 and self._tokens.pop("|".join(scopes), None):
                return self._req(method, url, scopes, headers, data, timeout, raw_resp)   # токен протух — обновить
            body = e.read().decode("utf-8", "replace")
            return e.code, {k.lower(): v for k, v in e.headers.items()}, body
        if raw_resp:
            return r
        body = r.read()
        return r.status, {k.lower(): v for k, v in r.headers.items()}, body      # ключи в нижнем регистре: Docker Hub шлёт «location», «range»

    def get_manifest(self, repo: str, ref: str) -> Tuple[str, dict, str]:
        st, h, body = self._req("GET", f"/v2/{repo}/manifests/{ref}", [f"repository:{repo}:pull"],
                                {"Accept": MANIFEST_ACCEPT})
        if st != 200:
            raise RegistryError(f"manifest {repo}:{ref}: HTTP {st} {body[:200] if isinstance(body, str) else ''}")
        mt = h.get("content-type", "").split(";")[0].strip()
        digest = h.get("docker-content-digest") or ("sha256:" + hashlib.sha256(body).hexdigest())
        return mt, json.loads(body), digest

    def manifest_exists(self, repo: str, tag: str) -> Optional[str]:
        try:
            st, h, _ = self._req("HEAD", f"/v2/{repo}/manifests/{tag}", [f"repository:{repo}:pull"], {"Accept": MANIFEST_ACCEPT})
        except (RegistryError, urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
            debug(f"registry HEAD {repo}:{tag}: {e}")
            return None
        return (h.get("docker-content-digest") or "exists") if st == 200 else None

    def resolve_platform(self, repo: str, ref: str, arch: str = "amd64", os_: str = "linux") -> Tuple[str, dict]:
        mt, m, _ = self.get_manifest(repo, ref)
        if mt in (MT_DOCKER_LIST, MT_OCI_INDEX):
            for entry in m.get("manifests", []):
                pl = entry.get("platform") or {}
                if pl.get("architecture") == arch and pl.get("os") == os_ and not pl.get("variant"):
                    return self.get_manifest(repo, entry["digest"])[:2]
            raise RegistryError(f"в {repo}:{ref} нет варианта {os_}/{arch}")
        return mt, m

    def get_blob(self, repo: str, digest: str) -> bytes:
        st, h, body = self._req("GET", f"/v2/{repo}/blobs/{digest}", [f"repository:{repo}:pull"], timeout=300)
        if st != 200:
            raise RegistryError(f"blob {digest[:19]}: HTTP {st}")
        return body

    def download_blob(self, repo: str, digest: str, path: Path, size: int, desc: str) -> None:
        """Потоковое скачивание blob в файл с докачкой (Range) и проверкой sha256
        (используется, если cross-repo mount не сработал)."""
        marker = Path(str(path) + ".sha256ok")
        if path.exists() and path.stat().st_size == size and marker.exists():
            ok(f"{desc}: уже скачан")
            return
        part = Path(str(path) + ".part")
        bar = ProgressBar(size, desc, bytes_mode=True)
        for attempt in range(1, 9):
            have = part.stat().st_size if part.exists() else 0
            if have > size:
                part.unlink()
                have = 0
            try:
                if have < size:
                    hdr = {"Range": f"bytes={have}-"} if have else {}
                    r = self._req("GET", f"/v2/{repo}/blobs/{digest}", [f"repository:{repo}:pull"], hdr, timeout=600, raw_resp=True)
                    if isinstance(r, tuple):
                        raise RegistryError(f"blob {digest[:19]}: HTTP {r[0]}")
                    if have and r.status != 206:
                        part.unlink()
                        have = 0
                    with open(part, "ab" if have else "wb") as f:
                        while True:
                            chunk = r.read(4 << 20)
                            if not chunk:
                                break
                            f.write(chunk)
                            have += len(chunk)
                            bar.set(have)
            except (RegistryError, urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                bar.render(force=True)
                warn(f"{desc}: обрыв ({e}); повтор {attempt}/8 через {5 * attempt} с")
                time.sleep(5 * attempt)
                continue
            if have != size:
                warn(f"{desc}: размер {have} ≠ {size}, повтор")
                continue
            bar.set(have, extra="проверка sha256")
            if "sha256:" + sha256_of(part) != digest:
                warn(f"{desc}: sha256 не совпал — скачиваю заново")
                part.unlink()
                continue
            part.replace(path)
            marker.write_text(digest)
            bar.close(f"{desc}: скачан и проверен ({fmt_bytes(size)})")
            return
        bar.close()
        raise RegistryError(f"blob {digest[:19]}: не удалось скачать")

    def blob_exists(self, repo: str, digest: str) -> bool:
        st, _, _ = self._req("HEAD", f"/v2/{repo}/blobs/{digest}", [f"repository:{repo}:pull"])
        return st == 200

    def mount_blob(self, repo: str, digest: str, from_repo: str) -> bool:
        st, h, _ = self._req("POST", f"/v2/{repo}/blobs/uploads/?mount={digest}&from={from_repo}",
                             [f"repository:{repo}:pull,push", f"repository:{from_repo}:pull"], data=b"")
        return st == 201

    def _abs(self, loc: str) -> str:
        return loc if loc.startswith("http") else self.registry + loc

    def upload_small(self, repo: str, data: bytes, digest: str) -> None:
        if self.blob_exists(repo, digest):
            return
        st, h, body = self._req("POST", f"/v2/{repo}/blobs/uploads/", [f"repository:{repo}:pull,push"], data=b"")
        if st != 202:
            raise RegistryError(f"начать загрузку: HTTP {st} {body[:200]}")
        if not h.get("location"):
            raise RegistryFatal(f"реестр не вернул Location для сессии загрузки (заголовки: {', '.join(sorted(h))})")
        loc = self._abs(h.get("location", ""))
        loc += ("&" if "?" in loc else "?") + "digest=" + digest
        st, h, body = self._req("PUT", loc, [f"repository:{repo}:pull,push"],
                                {"Content-Type": "application/octet-stream", "Content-Length": str(len(data))}, data=data)
        if st not in (201, 204):
            raise RegistryError(f"загрузка blob: HTTP {st} {body[:200]}")

    def upload_status(self, loc: str, scopes: list) -> Optional[int]:
        """Сколько байт уже принято в незавершённой сессии загрузки (GET <Location> → 204 + Range); None, если сессии нет."""
        try:
            st, h, _ = self._req("GET", loc, scopes)
        except (RegistryError, urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError):
            return None
        if st != 204:
            return None
        return _range_end(h.get("range", ""))

    def upload_blob(self, repo: str, path: Path, digest: str, size: int, desc: str, session_dir: Optional[Path] = None) -> None:
        """Кусочная загрузка (PATCH) с докачкой после обрыва; пропуск, если blob уже в реестре.
        Сессия загрузки (Location + принятый объём) сохраняется в session_dir, поэтому после падения/перезапуска
        скрипта (или всей VM) загрузка продолжается с того места, где остановилась, а не с нуля."""
        if self.blob_exists(repo, digest):
            ok(f"{desc}: уже в реестре ({fmt_bytes(size)})")
            return
        scopes = [f"repository:{repo}:pull,push"]
        sess_file = (session_dir / f"upload-{digest[7:19]}.json") if session_dir else None
        loc, offset = "", 0
        if sess_file and sess_file.exists():
            try:
                s = json.loads(sess_file.read_text())
                if s.get("repo") == repo and s.get("location"):
                    got = self.upload_status(s["location"], scopes)
                    if got is not None:
                        loc, offset = s["location"], min(got, s.get("offset", got))
                        info(f"{desc}: продолжаю прерванную загрузку с {fmt_bytes(offset)} из {fmt_bytes(size)}")
            except (OSError, ValueError):
                pass
            if not loc:
                sess_file.unlink(missing_ok=True)
        bar = ProgressBar(size, desc, bytes_mode=True)
        bar.set(offset)

        def save_session():
            if sess_file:
                sess_file.write_text(json.dumps({"repo": repo, "digest": digest, "location": loc, "offset": offset, "t": time.time()}))

        for attempt in range(1, 7):
            try:
                if not loc:
                    st, h, body = self._req("POST", f"/v2/{repo}/blobs/uploads/", scopes, data=b"")
                    if st != 202:
                        raise RegistryError(f"начать загрузку: HTTP {st} {body[:200]}")
                    if not h.get("location"):
                        raise RegistryFatal(f"реестр не вернул Location для сессии загрузки (заголовки: {', '.join(sorted(h))})")
                    loc, offset = self._abs(h.get("location", "")), 0
                    save_session()
                with open(path, "rb") as f:
                    while offset < size:
                        f.seek(offset)
                        chunk = f.read(UPLOAD_CHUNK)
                        end = offset + len(chunk) - 1
                        st, h, body = self._req("PATCH", loc, scopes,
                                                {"Content-Type": "application/octet-stream", "Content-Length": str(len(chunk)),
                                                 "Content-Range": f"{offset}-{end}"}, data=chunk, timeout=900)
                        if st == 404 and "BLOB_UPLOAD" in (body or ""):
                            raise RegistryError("сессия загрузки истекла на сервере")
                        if st in (400, 403, 404, 405, 413, 416, 422):
                            raise RegistryFatal(f"PATCH {loc[:120]}…: HTTP {st} {(body or '')[:300]}")
                        if st not in (202, 204):
                            raise RegistryError(f"PATCH: HTTP {st} {(body or '')[:200]}")
                        loc = self._abs(h.get("location", loc))
                        offset = _range_end(h.get("range", "")) if h.get("range") else end + 1
                        save_session()
                        bar.set(offset)
                final = loc + ("&" if "?" in loc else "?") + "digest=" + digest
                st, h, body = self._req("PUT", final, scopes, {"Content-Length": "0"}, data=b"", timeout=300)
                if st not in (201, 204):
                    raise RegistryError(f"завершение загрузки: HTTP {st} {body[:200]}")
                if sess_file:
                    sess_file.unlink(missing_ok=True)
                bar.close(f"{desc}: загружен ({fmt_bytes(size)})")
                return
            except RegistryFatal:
                raise
            except (RegistryError, urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                bar.render(force=True)
                warn(f"{desc}: обрыв загрузки ({e}); повтор {attempt}/6 через {10 * attempt} с")
                time.sleep(10 * attempt)
                # выясняем, сколько сервер принял в этой сессии; если сессии нет — начнём новую
                got = self.upload_status(loc, scopes) if loc else None
                if got is None:
                    loc, offset = "", 0
                    if sess_file:
                        sess_file.unlink(missing_ok=True)
                else:
                    offset = min(got, offset if offset else got)
                    save_session()
                bar.set(offset)
        bar.close()
        raise RegistryError(f"не удалось загрузить {desc}")

    def upload_layer_stream(self, repo: str, entries, desc: str, session_file: Path, executable: Iterable[str] = (),
                            compresslevel: int = 1) -> Tuple[str, str, int]:
        """Собирает tar.gz-слой и грузит его в реестр ПОТОКОМ одним PATCH-запросом (Transfer-Encoding: chunked —
        так же делает docker push): генерация (tar/gzip/sha256) идёт в отдельном потоке и через очередь кусков
        по 4 МБ уходит прямо в сокет, без пауз между кусками и без временного файла. Сессия (Location, принятый
        объём, sha256 каждого куска) сохраняется в session_file; после обрыва сети/падения скрипта/перезапуска VM
        сервер спрашивается, сколько принято, слой регенерируется детерминированно, принятая часть сверяется по хешам
        и пропускается, а PATCH продолжается с места обрыва. Возвращает (digest blob, diff_id, size)."""
        import queue
        scopes = [f"repository:{repo}:pull,push"]
        total_in = sum(p.stat().st_size for _, p in entries if p.is_file() and not p.is_symlink())
        est_total = total_in + 65536 if compresslevel == 0 else None
        PIECE = STREAM_PIECE
        for round_ in range(1, 8):
            sess = {}
            if session_file.exists():
                try:
                    sess = json.loads(session_file.read_text())
                except (OSError, ValueError):
                    sess = {}
            if sess.get("repo") != repo or not sess.get("location"):
                sess = {}
            got = self.upload_status(sess["location"], scopes) if sess else None
            if sess and got is None:
                sess = {}
            if sess:
                info(f"{desc}: продолжаю прерванную загрузку — на сервере уже {fmt_bytes(got)}")
            else:
                st, h, body = self._req("POST", f"/v2/{repo}/blobs/uploads/", scopes, data=b"")
                if st != 202:
                    raise RegistryError(f"начать загрузку: HTTP {st} {body[:200]}")
                if not h.get("location"):
                    raise RegistryFatal(f"реестр не вернул Location для сессии загрузки (заголовки: {', '.join(sorted(h))})")
                sess = {"repo": repo, "location": self._abs(h.get("location", "")), "offset": 0, "chunks": []}
                build_log(f"upload session {sess['location'][:100]}")
                got = 0
            state = {"loc": sess["location"], "offset": got, "chunks": list(sess.get("chunks") or []), "pos": 0,
                     "sent": 0, "gen": 0, "abort": False, "last_save": 0.0}
            bar = ProgressBar(est_total, desc, bytes_mode=True)

            def status():
                if state["pos"] < state["offset"] and state["gen"] < total_in:
                    return f"сверяю уже загруженное {fmt_bytes(state['pos'])}/{fmt_bytes(state['offset'])}"
                return f"собрано {state['gen'] * 100 // max(total_in, 1)}%"

            def show():
                bar.set(state["offset"] + state["sent"], extra=status())

            def save(force=False):
                if not force and time.time() - state["last_save"] < 5:
                    return
                state["last_save"] = time.time()
                session_file.parent.mkdir(parents=True, exist_ok=True)
                session_file.write_text(json.dumps({"repo": repo, "location": state["loc"], "offset": state["offset"] + state["sent"],
                                                    "chunks": state["chunks"], "t": time.time()}))

            show()
            q: "queue.Queue" = queue.Queue(maxsize=4)
            gen_err: List[BaseException] = []

            def on_piece(piece: bytes):
                """Вызывается генератором на каждый кусок: проверка/учёт хеша; в очередь — только то, чего у сервера нет."""
                if state["abort"]:
                    raise _Abort()
                start = state["pos"]
                end = start + len(piece)
                idx = start // PIECE
                d = hashlib.sha256(piece).hexdigest()
                if idx < len(state["chunks"]):
                    if state["chunks"][idx] != d:
                        raise _RestartUpload("регенерированный поток не совпал с загруженным ранее")
                else:
                    state["chunks"].append(d)
                state["pos"] = end
                if end <= state["offset"]:
                    show()
                    return
                if start < state["offset"]:
                    piece = piece[state["offset"] - start:]
                q.put(piece)

            def producer():
                sink = _ChunkSink(PIECE, on_piece)
                try:
                    diff_id = write_layer_stream(entries, sink, executable, compresslevel,
                                                 lambda n: (state.__setitem__("gen", n), show()))
                    sink.finish()
                    q.put(("done", diff_id, "sha256:" + sink.h.hexdigest(), sink.n))
                except BaseException as e:   # noqa: BLE001 — ошибка уходит в основной поток
                    gen_err.append(e)
                    try:
                        q.put(None, timeout=5)
                    except queue.Full:
                        pass

            t = threading.Thread(target=producer, name="layer-gen", daemon=True)
            t.start()
            result: List[tuple] = []

            def body():
                """Тело PATCH: куски из очереди; http.client шлёт каждый как chunk сразу."""
                while True:
                    item = q.get()
                    if item is None:
                        return
                    if isinstance(item, tuple):
                        result.append(item)
                        return
                    yield item
                    state["sent"] += len(item)
                    save()
                    show()

            try:
                # сессия точно жива? (иначе PATCH на мёртвый URL уйдёт целиком, прежде чем сервер ответит)
                if self.upload_status(state["loc"], scopes) is None:
                    raise _RestartUpload("сессия загрузки потеряна")
                try:
                    st, h, resp = self._req("PATCH", state["loc"], scopes, {"Content-Type": "application/octet-stream"},
                                            data=body(), timeout=900)
                except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                    if gen_err:
                        raise gen_err[0]
                    raise _Retry(str(e))
                if gen_err:
                    raise gen_err[0]
                if st == 404 and "BLOB_UPLOAD" in (resp or ""):
                    raise _RestartUpload("сессия загрузки истекла на сервере")
                if st in (400, 416) and got > 0:
                    raise _RestartUpload(f"сервер не принял продолжение сессии (HTTP {st} {(resp or '')[:120]})")
                if st in (400, 403, 404, 405, 413, 416, 422):
                    raise RegistryFatal(f"PATCH {state['loc'][:120]}…: HTTP {st} {(resp or '')[:300]}")
                if st not in (202, 204):
                    raise _Retry(f"PATCH: HTTP {st} {(resp or '')[:200]}")
                if not result:
                    raise _Retry("генерация слоя завершилась без результата")
                _, diff_id, digest, size = result[0]
                state["loc"] = self._abs(h.get("location", state["loc"]))
                acked = _range_end(h.get("range", "")) if h.get("range") else size
                if acked != size:
                    state["offset"], state["sent"] = acked, 0
                    save(force=True)
                    raise _Retry(f"сервер принял {fmt_bytes(acked)} из {fmt_bytes(size)}")
                final = state["loc"] + ("&" if "?" in state["loc"] else "?") + "digest=" + digest
                bar.set(size, extra="реестр проверяет digest")
                try:
                    st, h, resp = self._req("PUT", final, scopes, {"Content-Length": "0"}, data=b"", timeout=900)
                except (urllib.error.URLError, socket.timeout, TimeoutError, ConnectionError, OSError) as e:
                    st, resp = 0, str(e)
                if st not in (201, 204):
                    # сервер мог завершить blob, а ответ потерялся — проверяем по digest
                    if not self.blob_exists(repo, digest):
                        raise RegistryError(f"завершение загрузки: HTTP {st} {str(resp)[:200]}")
                session_file.unlink(missing_ok=True)
                bar.total = size
                bar.close(f"{desc}: загружен ({fmt_bytes(size)})")
                return digest, diff_id, size
            except _Retry as e:
                bar.render(force=True)
                state["abort"] = True
                self._drain(q, t)
                srv = self.upload_status(state["loc"], scopes)
                if srv is None:
                    warn(f"{desc}: {e}; сессия загрузки потеряна — начинаю заново ({round_}/7)")
                    session_file.unlink(missing_ok=True)
                else:
                    state["offset"], state["sent"] = srv, 0
                    save(force=True)
                    warn(f"{desc}: {e}; на сервере {fmt_bytes(srv)} — продолжу с этого места через {10 * round_} с ({round_}/7)")
                bar.close()
                time.sleep(10 * round_)
            except _RestartUpload as e:
                state["abort"] = True
                self._drain(q, t)
                bar.close()
                warn(f"{desc}: {e} — начинаю загрузку заново ({round_}/7)")
                session_file.unlink(missing_ok=True)
            except BaseException:
                state["abort"] = True
                self._drain(q, t)
                bar.close()
                raise
        raise RegistryError(f"не удалось загрузить {desc}")

    @staticmethod
    def _drain(q, t) -> None:
        """Останавливает поток генерации: опустошает очередь, пока он не завершится."""
        for _ in range(100000):
            if not t.is_alive():
                break
            try:
                q.get(timeout=0.1)
            except Exception:
                pass
        t.join(timeout=5)

    def put_manifest(self, repo: str, tag: str, manifest: dict, media_type: str) -> str:
        data = json.dumps(manifest, indent=2).encode()
        st, h, body = self._req("PUT", f"/v2/{repo}/manifests/{tag}", [f"repository:{repo}:pull,push"],
                                {"Content-Type": media_type, "Content-Length": str(len(data))}, data=data)
        if st not in (201, 204):
            raise RegistryError(f"публикация манифеста: HTTP {st} {body[:300]}")
        return h.get("docker-content-digest") or ("sha256:" + hashlib.sha256(data).hexdigest())


class _HashWriter:
    """Обёртка файла: считает sha256 и объём записанного."""
    def __init__(self, fobj):
        self.f, self.h, self.n = fobj, hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        return self.f.write(b)

    def flush(self):
        self.f.flush()

    def close(self):
        pass


LAYER_MTIME = 1_700_000_000   # фиксированное время файлов в слоях (воспроизводимость)


def dir_entries(root: Path) -> List[Tuple[str, Path]]:
    """Список (путь внутри слоя, файл на диске) для дерева root."""
    return [(p.relative_to(root).as_posix(), p) for p in sorted(root.rglob("*")) if p.is_file() or p.is_symlink()]


def layer_fingerprint(entries: List[Tuple[str, Path]], executable: Iterable[str] = (), compresslevel: int = 1) -> str:
    """Отпечаток содержимого слоя без чтения больших файлов: для файлов с маркером .sha256ok берётся хеш из маркера,
    маленькие (< 4 МБ) хешируются, большие — по размеру и mtime."""
    h = hashlib.sha256(f"v1|{LAYER_MTIME}|{compresslevel}|{','.join(sorted(executable))}|".encode())
    for rel, p in entries:
        if p.is_symlink():
            h.update(f"{rel}|L|{os.readlink(p)}|".encode())
            continue
        st = p.stat()
        marker = Path(str(p) + ".sha256ok")
        if marker.exists():
            h.update(f"{rel}|M|{st.st_size}|{marker.read_text().strip()}|".encode())
        elif st.st_size < 4 << 20:
            h.update(f"{rel}|C|{st.st_size}|{sha256_of(p)}|".encode())
        else:
            h.update(f"{rel}|S|{st.st_size}|{int(st.st_mtime)}|".encode())
    return h.hexdigest()


def is_wsl() -> bool:
    """Скрипт запущен внутри WSL (Windows Subsystem for Linux)."""
    if os.environ.get("WSL_DISTRO_NAME") or os.environ.get("WSL_INTEROP"):
        return True
    try:
        return "microsoft" in Path("/proc/version").read_text().lower()
    except OSError:
        return False


def build_log(msg: str) -> None:
    """Журнал сборки образа (<кэш>/build.log): время, шаг, свободное место, RSS — чтобы после внезапной
    остановки VM/сессии было видно, на чём всё остановилось."""
    try:
        import resource
        rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
        free = shutil.disk_usage(LOCAL_CACHE_DIR).free / 1e9 if LOCAL_CACHE_DIR.exists() else 0
        host = ""
        if is_wsl() and Path("/mnt/c").exists():
            host = f" host_c_free={shutil.disk_usage('/mnt/c').free / 1e9:.1f}G"
        LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with open(LOCAL_CACHE_DIR / "build.log", "a", encoding="utf-8") as f:
            f.write(f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} [free={free:.1f}G{host} rss={rss:.0f}M] {msg}\n")
    except Exception:
        pass


def wsl_drive_mounts() -> Dict[str, str]:
    """Под WSL: {"/mnt/c": "C:", "/mnt/d": "D:", …} — смонтированные диски Windows (drvfs/9p)."""
    out: Dict[str, str] = {}
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and re.fullmatch(r"/mnt/[a-z]", parts[1]) and parts[2] in ("9p", "drvfs", "virtiofs"):
                out[parts[1]] = parts[1][-1].upper() + ":"
    except OSError:
        pass
    if not out:
        for c in "cdefghijklmnopqrstuvwxyz":
            if Path(f"/mnt/{c}").is_dir() and os.path.ismount(f"/mnt/{c}"):
                out[f"/mnt/{c}"] = c.upper() + ":"
    return out


def on_windows_drive(path: Path) -> Optional[str]:
    """Если путь лежит на смонтированном диске Windows (/mnt/x) — его точка монтирования, иначе None."""
    sp = str(path.resolve() if path.exists() else path.absolute())
    for m in wsl_drive_mounts():
        if sp == m or sp.startswith(m + "/"):
            return m
    return None


def inside_wsl_vhdx(path: Path) -> bool:
    """Под WSL2: лежит ли путь в системном виртуальном диске ext4.vhdx (той же ФС, что корень «/»)?
    Диски Windows (/mnt/x), диски через `wsl --mount` (/mnt/wsl/…) и сетевые монтирования — нет."""
    if not is_wsl():
        return False
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        return os.stat(probe).st_dev == os.stat("/").st_dev
    except OSError:
        return False


def describe_location(path: Path) -> Tuple[str, int]:
    """Человеческое описание, где физически лежит путь, и сколько там свободно (байт)."""
    probe = path
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    try:
        free = shutil.disk_usage(probe).free
    except OSError:
        free = 0
    if is_wsl():
        m = on_windows_drive(path)
        if m:
            return f"диск Windows {m[-1].upper()}: (свободно {fmt_bytes(free)})", free
        if not inside_wsl_vhdx(path):
            return f"отдельный диск/монтирование (свободно {fmt_bytes(free)})", free
        host = ""
        if Path("/mnt/c").exists():
            try:
                cfree = shutil.disk_usage("/mnt/c").free
                host = f", растёт на диске C: (свободно на C: {fmt_bytes(cfree)})"
                free = min(free, cfree)
            except OSError:
                pass
        return f"внутри ext4.vhdx WSL2 (свободно в VM {fmt_bytes(shutil.disk_usage(probe).free)}{host})", free
    return f"свободно {fmt_bytes(free)}", free


WSL_COMPACT_HINT = (
    "Под WSL2 место, освобождённое внутри VM, Windows не возвращается само: ext4.vhdx только растёт. Чтобы его сжать, "
    "в PowerShell от администратора: (1) `wsl --shutdown`; (2) путь к диску: "
    "`(Get-ItemProperty HKCU:\\Software\\Microsoft\\Windows\\CurrentVersion\\Lxss\\*).BasePath` (+ `\\ext4.vhdx`); "
    "(3) либо `Optimize-VHD -Path \"<путь>\\ext4.vhdx\" -Mode Full` (нужен модуль Hyper-V: Windows Pro), "
    "либо в `diskpart` по строкам: `select vdisk file=\"<путь>\\ext4.vhdx\"`, `attach vdisk readonly`, `compact vdisk`, `detach vdisk`, `exit`. "
    "Проще всего один раз включить разрежённый диск: `wsl --manage <дистрибутив> --set-sparse true` (WSL 2.0+; имя — из `wsl -l`), "
    "тогда место возвращается автоматически. ")


def check_build_disk(cache_dir: Path, need_bytes: int) -> List[str]:
    """Проблемы с местом под сборку (пустой список — всё в порядке): в файловой системе кэша и — под WSL2, если кэш
    лежит внутри ext4.vhdx — на диске Windows C:, где этот виртуальный диск растёт (когда диск Windows заполняется,
    Hyper-V останавливает VM целиком, и сессия WSL обрывается без предупреждения)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(cache_dir).free
    problems = []
    if free < need_bytes:
        problems.append(f"в {cache_dir} свободно {fmt_bytes(free)}, нужно ≈ {fmt_bytes(need_bytes)}")
    if inside_wsl_vhdx(cache_dir) and Path("/mnt/c").exists():
        try:
            hfree = shutil.disk_usage("/mnt/c").free
        except OSError:
            hfree = None
        if hfree is not None and hfree < need_bytes + 3_000_000_000:
            problems.append(f"на диске Windows C: свободно {fmt_bytes(hfree)}, а ext4.vhdx WSL2 (в нём лежит {cache_dir}) вырастет "
                            f"ещё на ≈ {fmt_bytes(need_bytes)} — при заполнении диска Windows VM WSL2 будет остановлена")
    return problems


def cache_dir_size(path: Path) -> int:
    try:
        return sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    except OSError:
        return 0


def migrate_cache(old: Path, new: Path, mode: str) -> None:
    """mode: move — перенести содержимое old в new (с прогрессом; между дисками — копирование и удаление),
    delete — удалить old, keep — ничего не делать."""
    if mode == "keep" or not old.exists() or old.resolve() == new.resolve():
        return
    if mode == "delete":
        sp = Spinner(f"Удаляю старый кэш {old}")
        sp.tick("…")
        shutil.rmtree(old, ignore_errors=True)
        sp.close(f"Старый кэш удалён: {old}")
        return
    files = [p for p in old.rglob("*") if p.is_file()]
    total = sum(p.stat().st_size for p in files)
    new.mkdir(parents=True, exist_ok=True)
    same_fs = False
    try:
        same_fs = os.stat(old).st_dev == os.stat(new).st_dev
    except OSError:
        pass
    bar = ProgressBar(total, f"Переношу кэш в {new}", bytes_mode=True)
    done = 0
    for p in files:
        rel = p.relative_to(old)
        dst = new / rel
        dst.parent.mkdir(parents=True, exist_ok=True)
        if dst.exists() and dst.stat().st_size == p.stat().st_size:
            p.unlink()
        elif same_fs:
            try:
                p.replace(dst)
            except OSError:
                shutil.copy2(p, dst)
                p.unlink()
        else:
            tmp = Path(str(dst) + ".moving")
            with open(p, "rb") as src, open(tmp, "wb") as out:
                while True:
                    chunk = src.read(8 << 20)
                    if not chunk:
                        break
                    out.write(chunk)
                    done += len(chunk)
                    bar.set(done)
                drop_page_cache(src)
                drop_page_cache(out, sync=True)
            shutil.copystat(p, tmp)
            tmp.replace(dst)
            p.unlink()
            continue
        done += p.stat().st_size if p.exists() else dst.stat().st_size
        bar.set(done)
    shutil.rmtree(old, ignore_errors=True)
    bar.close(f"Кэш перенесён: {old} → {new} ({fmt_bytes(total)})")


def cache_candidates(need_bytes: int) -> List[Tuple[Path, str, int]]:
    """Варианты расположения кэша: по умолчанию + (под WSL) диски Windows с местом."""
    out = [(LOCAL_CACHE_DIR, *describe_location(LOCAL_CACHE_DIR))]
    if is_wsl():
        for m, letter in sorted(wsl_drive_mounts().items()):
            if on_windows_drive(LOCAL_CACHE_DIR) == m:
                continue
            cand = Path(m) / "vast_upscale_cache"
            try:
                free = shutil.disk_usage(m).free
            except OSError:
                continue
            if free >= max(need_bytes, 10_000_000_000) and os.access(m, os.W_OK):
                out.append((cand, f"диск Windows {letter} (свободно {fmt_bytes(free)})", free))
    return out


def choose_cache_dir(need_bytes: int, reason: str = "") -> Path:
    """Интерактивно спрашивает путь к рабочему кэшу; возвращает выбранный (созданный, проверенный на запись)."""
    cands = cache_candidates(need_bytes)
    _raw_print("")
    _raw_print(bold("Рабочая папка скрипта") + " — сюда складываются веса моделей, зависимости и данные сборки образа "
               f"({fmt_bytes(need_bytes) if need_bytes else 'до ~20 ГБ на модель'}), а также служебные файлы запусков "
               "(сконвертированные видео, состояние, логи); всё переиспользуется при повторных запусках.")
    if reason:
        _raw_print("  " + reason)
    for i, (path, desc, free) in enumerate(cands, 1):
        flag = "  ✖ мало места" if need_bytes and free < need_bytes else ""
        _raw_print(f"  {i}) {path} — {desc}{flag}")
    for _ in range(4):
        try:
            ans = input(f"  Номер варианта или свой путь [1]: ").strip()
        except EOFError:
            raise UserAbort()
        if not ans:
            ans = "1"
        if ans.isdigit() and 1 <= int(ans) <= len(cands):
            path = cands[int(ans) - 1][0]
        else:
            path = Path(ans).expanduser()
        try:
            path.mkdir(parents=True, exist_ok=True)
            probe = path / ".write_test"
            probe.write_text("ok")
            probe.unlink()
        except OSError as e:
            warn(f"{path}: нельзя писать ({e}) — выберите другой путь")
            continue
        desc, free = describe_location(path)
        if need_bytes and free < need_bytes:
            warn(f"{path}: {desc} — меньше, чем нужно ({fmt_bytes(need_bytes)})")
            if not ask_yes_no("Всё равно использовать этот путь?", default=False):
                continue
        return path.resolve()
    raise FatalError("Путь к кэшу не выбран.")


def ensure_cache_dir(args, auto_yes: Optional[bool], need_bytes: int = 0) -> Path:
    """Определяет рабочий кэш: --cache-dir (реконфигурация: спрашивает, перенести или удалить старые данные;
    без терминала — --cache-migrate или keep) → сохранённый в настройках → первый запуск: спрашивает у пользователя
    (без терминала / --yes — путь по умолчанию). Результат сохраняется в SETTINGS_FILE и в LOCAL_CACHE_DIR."""
    global LOCAL_CACHE_DIR
    settings = load_settings()
    saved = Path(settings["cache_dir"]).expanduser() if settings.get("cache_dir") else None
    interactive = sys.stdin.isatty() and not auto_yes
    if getattr(args, "cache_dir", None):
        new = Path(args.cache_dir).expanduser().resolve()
        old = saved or (DEFAULT_CACHE_DIR if DEFAULT_CACHE_DIR.exists() else None)
        if old and old.resolve() != new and old.exists() and cache_dir_size(old) > 0:
            mode = getattr(args, "cache_migrate", None)
            if not mode:
                if interactive:
                    _raw_print(f"  В прежнем кэше {old} уже есть {fmt_bytes(cache_dir_size(old))} данных (веса, зависимости, слои).")
                    _raw_print("    п) перенести в новый путь    у) удалить старые данные    о) оставить на месте (не использовать)")
                    try:
                        a = input("  Что сделать со старым кэшем [п]: ").strip().lower()
                    except EOFError:
                        raise UserAbort()
                    mode = {"": "move", "п": "move", "m": "move", "move": "move", "у": "delete", "d": "delete", "delete": "delete",
                            "о": "keep", "k": "keep", "keep": "keep"}.get(a, "move")
                else:
                    mode = "keep"
                    warn(f"Старый кэш {old} оставлен на месте (нет терминала; задайте --cache-migrate move|delete|keep).")
            new.mkdir(parents=True, exist_ok=True)
            migrate_cache(old, new, mode)
        new.mkdir(parents=True, exist_ok=True)
        if saved is None or saved.resolve() != new:
            settings["cache_dir"] = str(new)
            save_settings(settings)
            ok(f"Рабочий кэш: {new} (сохранено в {SETTINGS_FILE}; сменить: --cache-dir <путь>)")
        LOCAL_CACHE_DIR = new
        return new
    if saved:
        LOCAL_CACHE_DIR = saved
        desc, _ = describe_location(saved)
        note(f"рабочий кэш: {saved} — {desc}; сменить: --cache-dir <путь>")
        return saved
    if interactive:
        chosen = choose_cache_dir(need_bytes)
    else:
        chosen = LOCAL_CACHE_DIR
        chosen.mkdir(parents=True, exist_ok=True)
        info(f"Рабочий кэш по умолчанию: {chosen} (нет терминала / --yes; сменить: --cache-dir <путь>)")
    settings["cache_dir"] = str(chosen)
    save_settings(settings)
    ok(f"Рабочий кэш: {chosen} (сохранено в {SETTINGS_FILE}; сменить: --cache-dir <путь>)")
    LOCAL_CACHE_DIR = chosen
    return chosen


class _ChunkSink:
    """Файлоподобный приёмник: считает sha256/объём и отдаёт куски фиксированного размера в callback.
    Память ограничена: буфер ≤ размера куска плюс одна копия куска на время отправки."""
    def __init__(self, chunk: int, cb):
        self.chunk, self.cb, self.buf, self.h, self.n = chunk, cb, bytearray(), hashlib.sha256(), 0

    def write(self, b):
        self.h.update(b)
        self.n += len(b)
        self.buf += b
        while len(self.buf) >= self.chunk:
            mv = memoryview(self.buf)
            piece = bytes(mv[:self.chunk])          # одна копия (без промежуточного bytearray-среза)
            mv.release()
            del self.buf[:self.chunk]               # bytearray удаляет «голову» без переноса остатка
            self.cb(piece)
            del piece
        return len(b)

    def flush(self):
        pass

    def finish(self):
        if self.buf:
            self.cb(bytes(self.buf))
            self.buf = bytearray()


class _ReadProgress:
    """Обёртка чтения большого файла для tar: читает с диска блоками по 4 МБ (важно для 9p/drvfs в WSL2, где
    16-КБ чтения tarfile медленные), отдаёт tarfile куски запрошенного размера, сообщает прогресс каждые 8 МБ
    и сбрасывает прочитанное из page cache каждые 256 МБ (иначе в WSL2 кэш от 16-ГБ файла раздувает память VM)."""
    READ_AHEAD = 4 << 20

    def __init__(self, f, cb):
        self.f, self.cb, self.pos, self._rep, self._drop = f, cb, 0, 0, 0
        self._buf, self._off = b"", 0

    def read(self, n=-1):
        if n is None or n < 0:
            b = self._buf[self._off:] + self.f.read()
            self._buf, self._off = b"", 0
        else:
            if self._off + n > len(self._buf):
                rest = self._buf[self._off:]
                self._buf = rest + self.f.read(max(self.READ_AHEAD, n - len(rest)))
                self._off = 0
            b = self._buf[self._off:self._off + n]
            self._off += len(b)
        self.pos += len(b)
        if self.pos - self._rep >= (8 << 20) or not b:
            self._rep = self.pos
            self.cb(self.pos)
        if self.pos - self._drop >= (256 << 20):
            drop_page_cache(self.f, self._drop, self.pos - self._drop)
            self._drop = self.pos
        return b


def write_layer_stream(entries, sink, executable: Iterable[str] = (), compresslevel: int = 1, progress=None) -> str:
    """Пишет tar.gz-слой в sink (файлоподобный объект) потоком, без временного файла; возвращает diff_id
    (sha256 несжатого tar). Детерминировано: фиксированные mtime/uid/gid, gzip без имени файла и с фиксированным
    временем — повторная генерация даёт те же байты, что позволяет докачивать прерванную загрузку."""
    import gzip
    entries = sorted(entries, key=lambda e: e[0])
    exe = set(executable)
    gz = gzip.GzipFile(filename="", fileobj=sink, mode="wb", compresslevel=compresslevel, mtime=LAYER_MTIME)
    inner = _HashWriter(gz)
    # режим «w|» и стандартный размер копирования оставлены намеренно: от них зависит разбиение потока на блоки
    # gzip, а значит и байты blob — иначе прерванные загрузки прошлых версий не докачались бы
    tar = tarfile.open(fileobj=inner, mode="w|", format=tarfile.GNU_FORMAT)
    seen_dirs: set = set()
    done = 0

    def add_dir(rel: str):
        parts = rel.split("/")
        for i in range(1, len(parts) + 1):
            d = "/".join(parts[:i])
            if d and d not in seen_dirs:
                seen_dirs.add(d)
                ti = tarfile.TarInfo(d + "/")
                ti.type, ti.mode, ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = tarfile.DIRTYPE, 0o755, LAYER_MTIME, 0, 0, "root", "root"
                tar.addfile(ti)

    try:
        for rel, p in entries:
            add_dir(rel.rpartition("/")[0])
            ti = tarfile.TarInfo(rel)
            ti.mtime, ti.uid, ti.gid, ti.uname, ti.gname = LAYER_MTIME, 0, 0, "root", "root"
            if p.is_symlink():
                ti.type, ti.linkname, ti.mode = tarfile.SYMTYPE, os.readlink(p), 0o777
                tar.addfile(ti)
                continue
            ti.size = p.stat().st_size
            ti.mode = 0o755 if (rel in exe or os.access(p, os.X_OK)) else 0o644
            base = done
            with open(p, "rb") as f:
                if ti.size >= (64 << 20):
                    build_log(f"layer stream: {rel} ({ti.size} B)")
                    tar.addfile(ti, _ReadProgress(f, (lambda pos: progress(base + pos)) if progress else (lambda pos: None)))
                    drop_page_cache(f)
                else:
                    tar.addfile(ti, f)
            done += ti.size
            if progress:
                progress(done)
        tar.close()
        gz.close()
    except BaseException:
        # генерация прервана (обрыв/перезапуск загрузки): хвосты tar/gzip при закрытии — в никуда, без «Exception ignored»
        null = _NullSink()
        inner.f = null
        gz.fileobj = null
        for obj in (tar, gz):
            try:
                obj.close()
            except Exception:
                pass
        raise
    return "sha256:" + inner.h.hexdigest()


def _range_end(rng: str) -> int:
    """Сколько байт принято по заголовку Range «0-<N-1>»; «0-0» у пустой сессии (так в спецификации) и «0--1» — ноль."""
    rng = (rng or "").strip()
    if rng in ("", "0-0", "0--1"):
        return 0
    m = re.match(r"0-(\d+)$", rng)
    return int(m.group(1)) + 1 if m else 0


class _RestartUpload(Exception):
    """Сессия загрузки потеряна или регенерированный поток не совпал — начать загрузку слоя заново."""


class _Retry(Exception):
    """Обрыв отправки: спросить сервер, сколько принято, и продолжить с этого места."""


class _Abort(Exception):
    """Генерацию слоя нужно остановить (отправка прервана)."""


SEEDVR2_SRC_URL = "https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/archive/{commit}.tar.gz"
FFMPEG_STATIC_URL = "https://johnvansickle.com/ffmpeg/releases/ffmpeg-release-amd64-static.tar.xz"
BASE_PYTHON = {"pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime": "3.11"}
# версии torch в базовом образе — как constraints при разрешении зависимостей (сами пакеты не ставятся)
BASE_TORCH = {"pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime": {"torch": "2.8.0", "torchvision": "0.23.0", "torchaudio": "2.8.0"}}
# glibc базового образа (Ubuntu 22.04 → 2.35): допустимые теги manylinux при разрешении зависимостей
BASE_GLIBC = {"pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime": 35}
PIP_EXCLUDE = {"torch", "torchvision", "torchaudio", "triton"}


def manylinux_platforms(glibc_minor: int = 35) -> list:
    """Все теги manylinux x86_64, совместимые с glibc 2.<glibc_minor> (pip при --platform не расширяет их сам)."""
    return ["manylinux1_x86_64", "manylinux2010_x86_64", "manylinux2014_x86_64"] + \
           [f"manylinux_2_{m}_x86_64" for m in range(17, glibc_minor + 1)]
# чистые python-пакеты, у которых на PyPI только sdist (нет wheel): с --only-binary=:all: pip не может их взять,
# поэтому wheel собирается локально заранее (py3-none-any подходит для любой платформы)
SDIST_ONLY_PURE = ["antlr4-python3-runtime==4.9.3"]        # зависимость omegaconf 2.3.x


def _canon(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _wheelhouse_has(wheelhouse: Path, name: str) -> bool:
    c = _canon(name)
    return any(_canon(p.name.split("-")[0]) == c for p in wheelhouse.glob("*.whl"))


def _wheel_build_strategies():
    """Способы сборки wheel из sdist: обычный pip wheel; обход поломанного Debian-патча setuptools
    (SETUPTOOLS_USE_DISTUTILS=stdlib, затем --no-build-isolation); другие интерпретаторы в PATH —
    wheel чистого Python (py3-none-any) не зависит от версии интерпретатора, которым собран."""
    yield sys.executable, [], {}
    yield sys.executable, [], {"SETUPTOOLS_USE_DISTUTILS": "stdlib"}
    yield sys.executable, ["--no-build-isolation"], {}
    seen = {os.path.realpath(sys.executable)}
    for name in ("python3.12", "python3.13", "python3.11", "python3.10", "python3"):
        exe = shutil.which(name)
        if exe and os.path.realpath(exe) not in seen:
            seen.add(os.path.realpath(exe))
            yield exe, [], {}


def build_pure_wheels(specs, wheelhouse: Path) -> list:
    """Собирает wheel из sdist локально (для чистого Python получается py3-none-any). Возвращает имена собранных."""
    wheelhouse.mkdir(parents=True, exist_ok=True)
    built = []
    for spec in specs:
        name = re.split(r"[<>=!~\[ ;]", spec, 1)[0]
        if _wheelhouse_has(wheelhouse, name):
            built.append(name)
            continue
        sp = Spinner(f"pip wheel: собираю {name} из sdist")
        last = ""
        for exe, extra, env_add in _wheel_build_strategies():
            sp.tick(f"{spec} ({os.path.basename(exe)}{' ' + ' '.join(extra) if extra else ''}{' ' + ' '.join(env_add) if env_add else ''})")
            env = dict(os.environ, **env_add)
            cp = subprocess.run([exe, "-m", "pip", "wheel", "--quiet", "--no-deps", "--no-input", *extra, "-w", str(wheelhouse), spec],
                                capture_output=True, text=True, env=env)
            if cp.returncode == 0 and _wheelhouse_has(wheelhouse, name):
                break
            last = (cp.stderr or cp.stdout).strip()[-400:]
        sp.close()
        if _wheelhouse_has(wheelhouse, name):
            whl = [p.name for p in wheelhouse.glob("*.whl") if _canon(p.name.split("-")[0]) == _canon(name)]
            ok(f"wheel {name}: {whl[0] if whl else 'собран'}")
            built.append(name)
        else:
            warn(f"wheel {name} не собрался ни одним способом: {last}")
    return built


def parse_pip_conflict(stderr: str) -> list:
    """Из блока «The conflict is caused by:» вытаскивает спецификации зависимостей («X depends on name>=1»)."""
    specs = []
    for m in re.finditer(r"depends on ([A-Za-z0-9][A-Za-z0-9._-]*(?:\[[^\]]*\])?[^\s,;]*)", stderr):
        s = m.group(1).strip()
        if s not in specs:
            specs.append(s)
    return specs


def detect_base_python(cfg: dict) -> Optional[str]:
    """Версия Python базового образа по его конфигу: PYTHON_VERSION=… в истории сборки, python=3.x в командах,
    PYTHON_VERSION в Env; None, если не найдено."""
    texts = [e.get("created_by", "") for e in (cfg.get("history") or [])]
    texts += list((cfg.get("config") or {}).get("Env") or [])
    for t in texts:
        for pat in (r"PYTHON_VERSION=(\d\.\d{1,2})", r"python=(\d\.\d{1,2})", r"python(\d\.\d{1,2})-", r"/python(\d\.\d{1,2})/"):
            m = re.search(pat, t)
            if m:
                return m.group(1)
    return None


def fetch_seedvr2_source(dest: Path) -> None:
    """Исходники SeedVR2 CLI на закреплённом коммите — tarball с GitHub, без git."""
    if (dest / "inference_cli.py").exists() and (dest / ".commit").read_text().strip() == SEEDVR2_COMMIT if (dest / ".commit").exists() else False:
        ok("SeedVR2: исходники уже в кэше")
        return
    tgz = dest.parent / f"seedvr2-{SEEDVR2_COMMIT[:7]}.tar.gz"
    download_with_resume(SEEDVR2_SRC_URL.format(commit=SEEDVR2_COMMIT), tgz, 0, "", "SeedVR2 исходники")
    shutil.rmtree(dest, ignore_errors=True)
    dest.mkdir(parents=True)
    with tarfile.open(tgz, "r:gz") as t:
        members = t.getmembers()
        top = members[0].name.split("/")[0]
        for m in members:
            rel = m.name[len(top) + 1:]
            if not rel or m.isdir() and rel == "":
                continue
            m.name = rel
            t.extract(m, dest)
    (dest / ".commit").write_text(SEEDVR2_COMMIT)
    ok(f"SeedVR2 {SEEDVR2_COMMIT[:7]}: исходники распакованы")


def cross_install_packages(requirements: Path, target: Path, py_version: str, torch_pins=None, glibc_minor: int = 35) -> None:
    """Устанавливает зависимости SeedVR2 в target для Python контейнера (manylinux/x86_64) без запуска
    контейнера: pip умеет ставить чужую платформу при --only-binary=:all:. torch/torchvision уже в базовом образе
    (их версии передаются как constraints, чтобы остальные пакеты разрешались под них). Пакеты, у которых на PyPI
    нет wheel (например antlr4-python3-runtime у omegaconf), собираются локально в wheelhouse и подхватываются
    через --find-links; если pip сообщает о новом таком конфликте, wheel собирается и разрешение повторяется."""
    if (target / ".done").exists():
        ok("Зависимости SeedVR2: уже подготовлены")
        return
    lines = []
    for ln in requirements.read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if not ln or ln.startswith("#"):
            continue
        name = re.split(r"[<>=!~\[ ;]", ln, 1)[0].lower()
        if name in PIP_EXCLUDE:
            continue
        if name == "opencv-python":
            ln = "opencv-python-headless"      # без libGL в контейнере
        lines.append(ln)
    reqs = target.parent / "requirements.cross.txt"
    reqs.write_text("\n".join(lines) + "\n")
    cons = target.parent / "constraints.cross.txt"
    cons.write_text("".join(f"{k}=={v}\n" for k, v in (torch_pins or {}).items()))
    wheelhouse = target.parent / "wheelhouse"
    build_pure_wheels(SDIST_ONLY_PURE, wheelhouse)
    plat = []
    for tag in manylinux_platforms(glibc_minor):
        plat += ["--platform", tag]
    plat += ["--python-version", py_version, "--implementation", "cp", "--abi", "cp" + py_version.replace(".", ""),
             "--abi", "abi3", "--abi", "none", "--only-binary=:all:", "--find-links", str(wheelhouse)]
    report = target.parent / "pip-report.json"
    tried = set()
    for attempt in range(1, 5):
        sp = Spinner("pip: разрешаю зависимости под Python контейнера")
        sp.tick(f"python {py_version}, manylinux x86_64" + (f", попытка {attempt}" if attempt > 1 else ""))
        cp = subprocess.run([sys.executable, "-m", "pip", "install", "--dry-run", "--ignore-installed", "--no-input",
                             "--report", str(report), "--target", str(target), *plat, "-c", str(cons), "-r", str(reqs)],
                            capture_output=True, text=True)
        sp.close()
        if cp.returncode == 0:
            break
        err = cp.stderr
        specs = [s for s in parse_pip_conflict(err) if _canon(re.split(r"[<>=!~\[ ;]", s, 1)[0]) not in tried]
        if "ResolutionImpossible" not in err or not specs:
            raise FatalError(f"pip не смог разрешить зависимости под Python {py_version}:\n{err[-1200:]}")
        info("pip: конфликт из-за пакетов без wheel на PyPI — пробую собрать их из sdist: " + ", ".join(specs))
        for s in specs:
            tried.add(_canon(re.split(r"[<>=!~\[ ;]", s, 1)[0]))
        if not build_pure_wheels(specs, wheelhouse):
            raise FatalError(f"pip не смог разрешить зависимости под Python {py_version} (wheel из sdist не собрались):\n{err[-1200:]}")
    else:
        raise FatalError(f"pip не смог разрешить зависимости под Python {py_version} за 4 попытки")
    rep_ = json.loads(report.read_text())
    pkgs = []
    for it in rep_.get("install", []):
        md = it.get("metadata", {})
        nm = (md.get("name") or "").lower()
        if nm in PIP_EXCLUDE or nm.startswith("nvidia-"):
            continue
        pkgs.append(f"{md['name']}=={md['version']}")
    info(f"pip: {len(pkgs)} пакетов для контейнера: " + ", ".join(p.split("==")[0] for p in pkgs[:12]) + (" …" if len(pkgs) > 12 else ""))
    sp = Spinner("pip: скачиваю и раскладываю пакеты")
    sp.tick("…")
    cp = subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", "--no-compile", "--upgrade",
                         "--target", str(target), *plat, *pkgs], capture_output=True, text=True)
    sp.close()
    if cp.returncode != 0:
        raise FatalError(f"pip не смог установить пакеты в {target}: {cp.stderr[-800:]}")
    (target / ".done").write_text("\n".join(pkgs))
    (target / ".pyver").write_text(py_version)      # для проверки в контейнере: тот ли Python
    ok(f"Зависимости SeedVR2 подготовлены: {fmt_bytes(sum(p.stat().st_size for p in target.rglob('*') if p.is_file()))}")


def fetch_static_ffmpeg(bin_dir: Path) -> None:
    """Статические ffmpeg/ffprobe (johnvansickle) — без apt в контейнере."""
    if (bin_dir / "ffmpeg").exists() and (bin_dir / "ffprobe").exists():
        ok("ffmpeg (static): уже в кэше")
        return
    txz = bin_dir.parent / "ffmpeg-static.tar.xz"
    download_with_resume(FFMPEG_STATIC_URL, txz, 0, "", "ffmpeg static")
    bin_dir.mkdir(parents=True, exist_ok=True)
    with tarfile.open(txz, "r:xz") as t:
        for m in t.getmembers():
            base = m.name.rsplit("/", 1)[-1]
            if base in ("ffmpeg", "ffprobe") and m.isfile():
                with t.extractfile(m) as src, open(bin_dir / base, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                os.chmod(bin_dir / base, 0o755)
    ok("ffmpeg/ffprobe (static) готовы")


ENV_SH = """# окружение предсобранного образа vast_upscale (источник: bootstrap.sh / runner; перед source задайте PY)
export PATH=/opt/vu/bin:$PATH
_vu_py="${PY:-$(command -v python3)}"
_vu_want=$(cat /opt/vu/pylib/.pyver 2>/dev/null)
_vu_have=$("$_vu_py" -c 'import sys; print("%d.%d" % sys.version_info[:2])' 2>/dev/null)
if [ -z "$_vu_want" ] || [ "$_vu_want" = "$_vu_have" ]; then
  case ":${PYTHONPATH:-}:" in *:/opt/vu/pylib:*) ;; *) export PYTHONPATH=/opt/vu/pylib${PYTHONPATH:+:$PYTHONPATH};; esac
else
  echo "vast_upscale: /opt/vu/pylib собран под Python $_vu_want, а в контейнере Python $_vu_have — зависимости будут доустановлены pip" >&2
  export PYTHONPATH=$(echo "${PYTHONPATH:-}" | tr ':' '\\n' | grep -vx /opt/vu/pylib | paste -sd: -)
fi
"""


class Prebuild:
    """Предсобранный образ: base-образ + SeedVR2 + зависимости + веса, сделанный снапшотом
    настроенного контейнера (vast.ai take_snapshot → ваш Docker Hub). Старт инстанса из такого
    образа не тратит оплачиваемое время на установку и скачивание весов (pull образа не тарифицируется)."""

    def __init__(self, client: VastClient, key: Path, args, plan: Plan):
        self.client, self.key, self.args, self.plan = client, key, args, plan
        saved = load_dockerhub_creds()
        self.user = args.docker_user or os.environ.get("DOCKER_USER") or saved.get("user") or ""
        self.password = args.docker_pass or os.environ.get("DOCKER_PASS") or os.environ.get("DOCKER_TOKEN") or saved.get("token") or ""
        self.repo = args.image_repo or saved.get("repo") or (f"{self.user}/vast-upscale" if self.user else "")
        mode = args.prebuild
        if mode == "auto":
            mode = "local" if (self.user and self.password and self.repo) else "never"
        if mode != "never" and not (self.user and self.password and self.repo):
            warn("Предсобранный образ недоступен: нужны --docker-user/--docker-pass (или DOCKER_USER/DOCKER_PASS) и --image-repo.")
            mode = "never"
        self.mode = mode
        self.hub = DockerHub(self.user, self.password) if mode != "never" else None
        self.reg = RegistryClient(self.user, self.password) if mode != "never" else None
        self.py_ver = args.base_python or BASE_PYTHON.get(args.image) or "3.11"    # уточняется по метаданным базы при сборке
        self.cache: Dict[str, str] = {}
        try:
            self.cache = json.loads(IMAGE_CACHE_FILE.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            pass
        self.snapshot_pending: List[Tuple[Session, str, float]] = []   # (сессия, тег, время запроса)
        self._known: Dict[str, Optional[str]] = {}

    @property
    def enabled(self) -> bool:
        return self.mode != "never"

    def login_string(self) -> Optional[str]:
        return f"-u {self.user} -p {self.password} docker.io" if self.user and self.password else None

    def _save_cache(self):
        try:
            IMAGE_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            IMAGE_CACHE_FILE.write_text(json.dumps(self.cache, indent=2), encoding="utf-8")
        except OSError:
            pass

    def existing_image(self, model: dict) -> Optional[str]:
        """Имя готового образа для модели (repo:tag) или None."""
        if not self.enabled:
            return None
        tag = prebuilt_tag(model["dit"], self.args.image, self.py_ver)
        if tag in self._known:
            return self._known[tag]
        ref = self.cache.get(tag)
        if ref:
            r, _, t = ref.partition(":")
            if self.reg and self.reg.manifest_exists(r, t or "latest"):
                self._known[tag] = ref
                return ref
        if self.reg and self.reg.manifest_exists(self.repo, tag):
            self._known[tag] = f"{self.repo}:{tag}"
            self.cache[tag] = self._known[tag]
            self._save_cache()
            return self._known[tag]
        self._known[tag] = None
        return None

    def image_for(self, model: dict) -> Tuple[str, Optional[str], bool]:
        """(образ, image_login, предсобран?) для инстанса с этой моделью."""
        ref = self.existing_image(model)
        if ref:
            return ref, self.login_string(), True
        return self.args.image, None, False

    # ---- снапшот ----
    def request_snapshot(self, session: "Session", model: dict) -> Optional[str]:
        tag = prebuilt_tag(model["dit"], self.args.image, self.py_ver)
        try:
            r = self.client.take_snapshot(session.instance_id, f"{self.repo}:{tag}", self.user, self.password)
        except VastAPIError as e:
            warn(f"Снапшот не запрошен: {e.msg}")
            return None
        if not r.get("success", True):
            warn(f"vast.ai отказал в снапшоте: {r.get('msg') or r}")
            return None
        ok(f"Запрошен снапшот контейнера → docker.io/{self.repo}:{tag} (push делает хост, обычно 5–15 мин)")
        self.snapshot_pending.append((session, tag, time.time()))
        return tag

    def wait_for_tag(self, tag: str, since: float, timeout: float) -> Optional[str]:
        """Ждёт появления тега в Docker Hub; если хост запушил под другим тегом (напр. latest) — берёт его."""
        assert self.hub
        sp = Spinner("Жду публикации образа в Docker Hub", est_total_sec=min(timeout, 900))
        t0 = time.time()
        while time.time() - t0 < timeout:
            info_ = self.hub.tag_info(self.repo, tag) or ({"last_updated": None} if (self.reg and self.reg.manifest_exists(self.repo, tag)) else None)
            if info_ and (info_.get("last_updated") is None or _iso_to_ts(info_.get("last_updated") or info_.get("tag_last_pushed")) >= since - 120):
                sp.close(f"Образ опубликован: {self.repo}:{tag} ({fmt_bytes(info_.get('full_size'))})")
                ref = f"{self.repo}:{tag}"
                self.cache[tag] = ref
                self._save_cache()
                self._known[tag] = ref
                return ref
            for t in self.hub.tags(self.repo):
                if _iso_to_ts(t.get("last_updated") or t.get("tag_last_pushed")) >= since - 120 and t.get("name") != tag:
                    ref = f"{self.repo}:{t['name']}"
                    sp.close(f"Образ опубликован под тегом {t['name']} — запомнил соответствие {tag} → {ref}")
                    self.cache[tag] = ref
                    self._save_cache()
                    self._known[tag] = ref
                    return ref
            sp.tick("проверяю теги репозитория")
            time.sleep(30)
        sp.close()
        warn(f"Образ {self.repo}:{tag} не появился за {fmt_time(timeout)} — следующий запуск снова соберёт окружение.")
        return None

    def finish_pending(self) -> None:
        for session, tag, since in self.snapshot_pending:
            if session.instance_id:
                self.wait_for_tag(tag, since, self.args.snapshot_wait)
        self.snapshot_pending = []

    # ---- локальная сборка на этой машине без Docker: слои OCI + Registry API ----
    def find_layer_in_registry(self, marker: str) -> Optional[Tuple[str, str, int, str]]:
        """Ищет уже загруженный слой в образах этого репозитория (теги других версий скрипта/Python): слой узнаётся
        по строке marker в его записи history (например «weights <dit>»). Возвращает (digest, diff_id, size, тег)."""
        assert self.reg
        names: List[str] = []
        try:
            names = [t.get("name") for t in (self.hub.tags(self.repo) if self.hub else []) if t.get("name")]
        except Exception as e:  # noqa — Hub API необязателен
            debug(f"hub tags: {e}")
        for t in names:
            if not t.startswith("seedvr2-"):
                continue
            try:
                mt, m, _ = self.reg.get_manifest(self.repo, t)
                if mt in (MT_DOCKER_LIST, MT_OCI_INDEX) or not m.get("layers"):
                    continue
                cfg = json.loads(self.reg.get_blob(self.repo, m["config"]["digest"]))
                hist = [h for h in (cfg.get("history") or []) if not h.get("empty_layer")]
                diffs = (cfg.get("rootfs") or {}).get("diff_ids") or []
                layers = m["layers"]
                if len(diffs) != len(layers):
                    continue
                ver = ((cfg.get("config") or {}).get("Labels") or {}).get("vast_upscale.version", "0")
                if tuple(int(x) for x in re.findall(r"\d+", ver)[:3]) < (1, 6, 0):
                    continue          # до 1.6.0 в слое весов не было маркеров проверки — такой слой не годится
                # наши слои — последние в образе; считаем с конца (история базового образа может быть неполной)
                for j in range(1, min(len(hist), len(layers)) + 1):
                    if marker in hist[-j].get("created_by", "") and self.reg.blob_exists(self.repo, layers[-j]["digest"]):
                        return layers[-j]["digest"], diffs[-j], int(layers[-j]["size"]), t
            except (RegistryError, ValueError, KeyError, OSError, urllib.error.URLError, socket.timeout) as e:
                debug(f"registry {self.repo}:{t}: {e}")
        return None

    def layer_without_inputs(self, name: str, marker: str) -> Optional[Tuple[str, str, int]]:
        """Слой, чьи исходные файлы на этой машине отсутствуют (например, веса после смены кэша): берётся из
        кэша digest-ов (<cache>/layers/<name>.json), если blob есть в реестре, иначе ищется в опубликованных образах."""
        assert self.reg
        ldir = LOCAL_CACHE_DIR / "layers"
        side = ldir / f"{name}.json"
        try:
            meta = json.loads(side.read_text()) if side.exists() else {}
        except (OSError, ValueError):
            meta = {}
        if meta.get("digest") and meta.get("diff_id") and self.reg.blob_exists(self.repo, meta["digest"]):
            build_log(f"layer {name}: cached digest {meta['digest'][:19]} present in registry (inputs absent locally)")
            return meta["digest"], meta["diff_id"], int(meta.get("size", 0))
        found = self.find_layer_in_registry(marker)
        if not found:
            return None
        digest, diff_id, size, src = found
        ldir.mkdir(parents=True, exist_ok=True)
        side.write_text(json.dumps({"fingerprint": None, "digest": digest, "diff_id": diff_id, "size": size,
                                    "built": time.time(), "version": VERSION, "reused_from": src}, indent=1))
        build_log(f"layer {name}: reused digest {digest[:19]} from {src} (inputs absent locally)")
        return digest, diff_id, size

    def publish_layer(self, name: str, entries, desc: str, executable=(), compresslevel: int = 1,
                      reuse_from: str = "") -> Tuple[str, str, int]:
        """Слой с кэшем результата: <cache>/layers/<name>.json хранит digest/diff_id/size и отпечаток входа.
        Если отпечаток совпал и blob уже в реестре — ни сборки, ни загрузки; иначе (reuse_from) слой ищется среди
        уже опубликованных образов репозитория; иначе собирается и грузится потоком с докачкой
        (сессия в <cache>/layers/upload-<name>.json)."""
        assert self.reg
        ldir = LOCAL_CACHE_DIR / "layers"
        ldir.mkdir(parents=True, exist_ok=True)
        side = ldir / f"{name}.json"
        fp = layer_fingerprint(entries, executable, compresslevel)
        meta = {}
        if side.exists():
            try:
                meta = json.loads(side.read_text())
            except (OSError, ValueError):
                meta = {}
        if meta.get("fingerprint") in (fp, None) and meta.get("digest") and meta.get("diff_id"):   # None: слой взят из реестра
            if self.reg.blob_exists(self.repo, meta["digest"]):
                ok(f"{desc}: уже в реестре ({fmt_bytes(meta.get('size', 0))}, из кэша {side.name})")
                build_log(f"layer {name}: cached digest {meta['digest'][:19]} present in registry")
                return meta["digest"], meta["diff_id"], meta["size"]
            info(f"{desc}: blob {meta['digest'][:19]}… в реестре не найден — собираю и гружу заново")
        if reuse_from:
            found = self.find_layer_in_registry(reuse_from)
            if found:
                digest, diff_id, size, src = found
                side.write_text(json.dumps({"fingerprint": fp, "digest": digest, "diff_id": diff_id, "size": size,
                                            "built": time.time(), "version": VERSION, "reused_from": src}, indent=1))
                ok(f"{desc}: переиспользую из образа {src} ({fmt_bytes(size)}) — без сборки и загрузки")
                build_log(f"layer {name}: reused digest {digest[:19]} from {src}")
                return digest, diff_id, size
        build_log(f"layer {name}: build+upload start (fingerprint {fp[:12]})")
        digest, diff_id, size = self.reg.upload_layer_stream(self.repo, entries, desc, ldir / f"upload-{name}.json",
                                                             executable, compresslevel)
        side.write_text(json.dumps({"fingerprint": fp, "digest": digest, "diff_id": diff_id, "size": size,
                                    "built": time.time(), "version": VERSION}, indent=1))
        build_log(f"layer {name}: done digest {digest[:19]} size {size}")
        return digest, diff_id, size

    def build_local(self, model: dict) -> Optional[str]:
        """Собирает образ на ЭТОЙ машине без Docker-демона: базовый образ монтируется в ваш репозиторий
        (его слои не скачиваются), поверх — слой софта (SeedVR2, зависимости, ffmpeg) и слой весов;
        слои собираются и грузятся потоком (без временных tar.gz на диске), каждый шаг кэшируется
        (исходники, зависимости, ffmpeg, веса — с докачкой; digest готовых слоёв; сессии загрузки),
        поэтому после обрыва/перезапуска сборка продолжается, а не начинается заново. Возвращает repo:tag."""
        assert self.reg
        dit = model["dit"]
        base_repo, base_tag = parse_image_ref(self.args.image)
        py_ver = self.py_ver
        detected = None
        weights = [(dit, MODEL_FILES[dit]), (VAE_FILE, MODEL_FILES[VAE_FILE])]
        need = sum(w[1]["size"] for w in weights)
        build_log(f"=== build {model_short(dit)} start; cache {LOCAL_CACHE_DIR}")
        # место: веса (если ещё не скачаны) + ~2 ГБ на зависимости/исходники; слои на диск не пишутся
        to_download = sum(m["size"] for n, m in weights
                          if not ((LOCAL_CACHE_DIR / "models" / n).exists() and Path(str(LOCAL_CACHE_DIR / "models" / n) + ".sha256ok").exists()))
        need_disk = to_download + 2_000_000_000
        problems = check_build_disk(LOCAL_CACHE_DIR, need_disk)
        if problems and not getattr(self.args, "ignore_disk_check", False):
            msg = "Мало места для сборки образа: " + "; ".join(problems)
            if sys.stdin.isatty() and not self.args.yes:
                warn(msg)
                if ask_yes_no("Выбрать другой путь для рабочего кэша сейчас?", default=True):
                    old_dir = LOCAL_CACHE_DIR
                    new_dir = choose_cache_dir(need_disk, "Прежний кэш будет перенесён в новый путь (или удалён/оставлен — по вашему выбору).")
                    if new_dir != old_dir:
                        self.args.cache_dir = str(new_dir)
                        self.args.cache_migrate = None
                        ensure_cache_dir(self.args, None, need_disk)
                    problems = check_build_disk(LOCAL_CACHE_DIR, need_disk)
            if problems:
                raise FatalError("Мало места для сборки образа: " + "; ".join(problems) +
                                 ". Освободите место или задайте другой путь: --cache-dir <путь> (старые данные можно перенести). "
                                 + (WSL_COMPACT_HINT if is_wsl() else "") + "Уже скачанное будет использовано.")
        elif problems:
            warn("Мало места для сборки образа: " + "; ".join(problems) + " (продолжаю из-за --ignore-disk-check)")
        info(f"Сборка образа для {model.get('label', dit)} на этой машине (без Docker): базовый {self.args.image} (Python {py_ver}), "
             f"SeedVR2 {SEEDVR2_COMMIT[:7]}, веса {fmt_bytes(need)}; кэш сборки: {LOCAL_CACHE_DIR}")
        if is_wsl():
            note("WSL2: скрипт сам сбрасывает прочитанное из page cache, но VM лучше ограничить в %UserProfile%\\.wslconfig: "
                 "[wsl2] memory=<не больше половины RAM>; [experimental] autoMemoryReclaim=gradual, sparseVhd=true (WSL 2.0+), "
                 "затем `wsl --shutdown`")
        t_all = time.time()
        # 1) базовый образ: манифест + конфиг (слои не скачиваем)
        sp = Spinner("Читаю манифест базового образа")
        sp.tick(f"{base_repo}:{base_tag}")
        base_mt, base_manifest = self.reg.resolve_platform(base_repo, base_tag)
        base_cfg = json.loads(self.reg.get_blob(base_repo, base_manifest["config"]["digest"]))
        sp.close(f"Базовый образ: {len(base_manifest['layers'])} слоёв, {fmt_bytes(sum(l['size'] for l in base_manifest['layers']))}, "
                 f"{base_cfg.get('os')}/{base_cfg.get('architecture')}")
        detected = detect_base_python(base_cfg)
        if detected and not self.args.base_python and detected != py_ver:
            warn(f"Базовый образ, судя по его метаданным, с Python {detected}, а не {py_ver} — зависимости готовлю под {detected} "
                 f"(переопределить: --base-python)")
            py_ver = self.py_ver = detected
        elif detected:
            note(f"Python базового образа: {detected}")
        else:
            note(f"Python базового образа принят как {py_ver} (в метаданных не указан; при несовпадении инстанс доустановит "
                 f"только зависимости pip, а следующий запуск с --base-python <версия> соберёт образ под неё)")
        tag = prebuilt_tag(dit, self.args.image, py_ver)
        ref = f"{self.repo}:{tag}"
        if self.reg.manifest_exists(self.repo, tag):
            ok(f"Образ {ref} уже есть в реестре")
            self.cache[tag] = ref
            self._save_cache()
            self._known[tag] = ref
            return ref
        ctx = LOCAL_CACHE_DIR / "image" / tag
        ctx.mkdir(parents=True, exist_ok=True)
        oci = base_mt == MT_OCI_MANIFEST
        mt_layer = MT_OCI_LAYER if oci else MT_DOCKER_LAYER
        mt_config = MT_OCI_CONFIG if oci else MT_DOCKER_CONFIG
        mt_manifest = MT_OCI_MANIFEST if oci else MT_DOCKER_MANIFEST
        # 2) слои базового образа: cross-repo mount в ваш репозиторий (скачивание+загрузка — только если mount не сработал)
        sp = Spinner("Монтирую слои базового образа в ваш репозиторий")
        for i, layer in enumerate(base_manifest["layers"], 1):
            sp.tick(f"слой {i}/{len(base_manifest['layers'])}")
            if self.reg.blob_exists(self.repo, layer["digest"]):
                continue
            if not self.reg.mount_blob(self.repo, layer["digest"], base_repo):
                sp.close()
                warn(f"Слой {layer['digest'][:19]} не смонтировался — скачиваю и загружаю ({fmt_bytes(layer['size'])})")
                tmp = LOCAL_CACHE_DIR / "layers" / ("base-" + layer["digest"][7:19] + ".blob")
                self.reg.download_blob(base_repo, layer["digest"], tmp, layer["size"], f"Слой базового образа {i}")
                self.reg.upload_blob(self.repo, tmp, layer["digest"], layer["size"], f"Слой базового образа {i}",
                                     session_dir=LOCAL_CACHE_DIR / "layers")
                tmp.unlink(missing_ok=True)
                Path(str(tmp) + ".sha256ok").unlink(missing_ok=True)
                sp = Spinner("Монтирую слои базового образа в ваш репозиторий")
        sp.close(f"Слои базового образа доступны в {self.repo}")
        build_log("base layers mounted")
        # 3) слой софта: исходники, зависимости под Python контейнера, статический ffmpeg
        soft = ctx / "soft"
        opt = soft / "opt" / "vu"
        fetch_seedvr2_source(LOCAL_CACHE_DIR / "seedvr2-src")
        if not (opt / "seedvr2" / "inference_cli.py").exists():
            shutil.copytree(LOCAL_CACHE_DIR / "seedvr2-src", opt / "seedvr2", dirs_exist_ok=True)
        cross_install_packages(opt / "seedvr2" / "requirements.txt", LOCAL_CACHE_DIR / f"pylib-py{py_ver}", py_ver,
                               BASE_TORCH.get(self.args.image), BASE_GLIBC.get(self.args.image, 35))
        if not (opt / "pylib" / ".done").exists():
            shutil.copytree(LOCAL_CACHE_DIR / f"pylib-py{py_ver}", opt / "pylib", dirs_exist_ok=True)
        fetch_static_ffmpeg(LOCAL_CACHE_DIR / "ffmpeg-bin")
        (opt / "bin").mkdir(parents=True, exist_ok=True)
        for b in ("ffmpeg", "ffprobe"):
            if not (opt / "bin" / b).exists():
                shutil.copyfile(LOCAL_CACHE_DIR / "ffmpeg-bin" / b, opt / "bin" / b)
                os.chmod(opt / "bin" / b, 0o755)
        (opt / "env.sh").write_text(ENV_SH, encoding="utf-8")
        (opt / ".prebuilt").write_text(f"{dit} {SEEDVR2_COMMIT}\n", encoding="utf-8")
        soft_entries = dir_entries(soft)
        soft_digest, soft_diff, soft_size = self.publish_layer(
            f"soft-{tag}", soft_entries, "Слой софта", executable=("opt/vu/bin/ffmpeg", "opt/vu/bin/ffprobe"), compresslevel=1)
        # 4) веса: слой уже опубликован (этим или прежним тегом)? — тогда ни скачивания, ни загрузки;
        #    иначе скачиваются в кэш с докачкой и проверкой sha256, в слой идут прямо из кэша (без копий и hardlink)
        w_name = f"weights-{dit.rsplit('.', 1)[0]}"
        ready = self.layer_without_inputs(w_name, f"weights {dit}") if to_download else None
        if ready:
            w_digest, w_diff, w_size = ready
            ok(f"Слой весов уже в реестре ({fmt_bytes(w_size)}) — веса на эту машину не скачиваются")
        else:
            meta_dir = ctx / "weights-meta"
            meta_dir.mkdir(parents=True, exist_ok=True)
            cache_json = {}
            w_entries = []
            for name, meta in weights:
                cached = LOCAL_CACHE_DIR / "models" / name
                download_with_resume(f"https://huggingface.co/{meta['repo']}/resolve/main/{name}", cached, meta["size"],
                                     meta["sha256"], f"Веса {name}")
                build_log(f"weights {name} ready ({meta['size']} B)")
                (meta_dir / (name + ".sha256ok")).write_text(meta["sha256"])
                cache_json[name] = {"size": meta["size"], "mtime": float(LAYER_MTIME), "hash": meta["sha256"]}
                w_entries += [(f"opt/vu/models/{name}", cached), (f"opt/vu/models/{name}.sha256ok", meta_dir / (name + ".sha256ok"))]
            (meta_dir / ".validation_cache.json").write_text(json.dumps(cache_json, indent=2))
            w_entries.append(("opt/vu/models/.validation_cache.json", meta_dir / ".validation_cache.json"))
            # safetensors не сжимаются — gzip уровня 0 (stored): быстрее в разы, размер тот же
            w_digest, w_diff, w_size = self.publish_layer(w_name, w_entries, "Слой весов", compresslevel=0,
                                                          reuse_from=f"weights {dit}")     # есть в образе прежнего тега? — не грузить
        # 5) конфиг образа
        now = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000000000Z")
        cfg = dict(base_cfg)
        cfg.pop("container", None)
        cfg.pop("container_config", None)
        cfg["created"] = now
        cfg.setdefault("rootfs", {"type": "layers", "diff_ids": []})
        cfg["rootfs"]["diff_ids"] = list(cfg["rootfs"].get("diff_ids", [])) + [soft_diff, w_diff]
        hist = list(cfg.get("history", []))
        hist.append({"created": now, "created_by": f"vast_upscale.py {VERSION}: SeedVR2 {SEEDVR2_COMMIT[:7]} + deps + ffmpeg (/opt/vu)"})
        hist.append({"created": now, "created_by": f"vast_upscale.py {VERSION}: weights {dit} (/opt/vu/models)"})
        cfg["history"] = hist
        c = cfg.setdefault("config", {}) or {}
        env = [e for e in (c.get("Env") or []) if not e.startswith("PYTHONPATH=")]
        env = [("PATH=/opt/vu/bin:" + e[5:]) if e.startswith("PATH=") else e for e in env]
        if not any(e.startswith("PATH=") for e in env):
            env.append("PATH=/opt/vu/bin:/opt/conda/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        env.append("PYTHONPATH=/opt/vu/pylib")
        c["Env"] = env
        labels = dict(c.get("Labels") or {})
        labels.update({"vast_upscale.version": VERSION, "vast_upscale.model": dit, "vast_upscale.seedvr2": SEEDVR2_COMMIT})
        c["Labels"] = labels
        cfg["config"] = c
        cfg_bytes = json.dumps(cfg, separators=(",", ":")).encode()
        cfg_digest = "sha256:" + hashlib.sha256(cfg_bytes).hexdigest()
        # 6) публикация конфига и манифеста
        self.reg.upload_small(self.repo, cfg_bytes, cfg_digest)
        manifest = {"schemaVersion": 2, "mediaType": mt_manifest,
                    "config": {"mediaType": mt_config, "size": len(cfg_bytes), "digest": cfg_digest},
                    "layers": [{"mediaType": l.get("mediaType", mt_layer), "size": l["size"], "digest": l["digest"]}
                               for l in base_manifest["layers"]]
                              + [{"mediaType": mt_layer, "size": soft_size, "digest": soft_digest},
                                 {"mediaType": mt_layer, "size": w_size, "digest": w_digest}]}
        mdigest = self.reg.put_manifest(self.repo, tag, manifest, mt_manifest)
        ok(f"Образ опубликован за {fmt_time(time.time() - t_all)}: docker.io/{ref} ({mdigest[:19]}…, "
           f"новые слои {fmt_bytes(soft_size + w_size)})")
        build_log(f"=== build {ref} published {mdigest[:19]} in {time.time() - t_all:.0f}s")
        self.cache[tag] = ref
        self._save_cache()
        self._known[tag] = ref
        return ref

    def rebuild_for_python(self, model: dict, py_ver: str) -> bool:
        """Контейнер оказался с другой версией Python, чем предполагалось: собрать образ под неё (слои весов и уже
        загруженные blob-ы переиспользуются по digest, заново готовятся только зависимости)."""
        if self.mode != "local":
            return False
        self.args.base_python = py_ver
        self.py_ver = py_ver
        self._known.clear()
        try:
            return bool(self.build_local(model))
        except FatalError as e:
            err(str(e))
            return False

    # ---- строгий режим: сначала образ, потом рабочий инстанс ----
    def ensure_image(self, model: dict, cands: List["Candidate"]) -> Optional[str]:
        """Если образа нет — собирает его: local — на этой машине (docker build/push);
        strict — снапшотом на самом дешёвом арендуемом инстансе."""
        if not self.enabled:
            return None
        ref = self.existing_image(model)
        if ref:
            ok(f"Предсобранный образ найден: {ref}")
            return ref
        if self.mode == "local":
            return self.build_local(model)
        tag = prebuilt_tag(model["dit"], self.args.image, self.py_ver)
        info(f"Предсобранного образа {self.repo}:{tag} ещё нет — собираю его (одноразово для этой модели).")
        # сборщик: дешёвый по часу, с быстрым аплинком (push образа), с той же моделью
        builders = [c for c in cands if c.model["dit"] == model["dit"] and float(c.offer.get("inet_up") or 0) >= 300]
        if not builders:
            warn("Нет офферов с аплинком ≥ 300 Мбит/с для сборки образа — беру самый дешёвый подходящий (push будет дольше).")
            builders = [c for c in cands if c.model["dit"] == model["dit"]]
        builders.sort(key=lambda c: c.price)
        if not builders:
            warn("Нет офферов под эту модель для сборки образа — соберу образ на рабочем инстансе (inline).")
            return None
        empty = Plan(jobs=[], target=self.plan.target, model_pref=self.plan.model_pref, total_frames=0, upload_bytes=0,
                     est_download_bytes=0, local_up_bps=self.plan.local_up_bps, local_down_bps=self.plan.local_down_bps,
                     time_value=self.plan.time_value, bid_margin=self.plan.bid_margin, disk_gb=self.plan.disk_gb, optimize="cost")
        b = Session(self.client, self.key, empty, self.args, self.plan.jobs[0].converted.parent if self.plan.jobs else Path("."),
                    jobs=[], tag="builder")
        b.image, b.image_login = self.args.image, None
        t0 = time.time()
        try:
            b.provision(builders[:self.args.max_attempts])
            b.bootstrap()
            since = time.time()
            if not self.request_snapshot(b, model):
                return None
            self.snapshot_pending = []
            ref = self.wait_for_tag(tag, since, self.args.snapshot_wait)
            return ref
        finally:
            if b.instance_id:
                b.destroy()
            note(f"Сборка образа заняла {fmt_time(time.time() - t0)}; аренда сборщика ≈ {fmt_money((b.cand.price if b.cand else 0) * (time.time() - t0) / 3600)}")


# ----------------------------------------------------------------------------
# Шаг 2. Сопряжение с vast.ai: API-ключ, SSH-ключ, баланс
# ----------------------------------------------------------------------------
PAIRING_INSTRUCTIONS = """
  Для работы с vast.ai нужны три вещи:

  1) Аккаунт и баланс.
     • Зарегистрируйтесь: https://cloud.vast.ai/
     • Пополните баланс: https://cloud.vast.ai/billing/  (минимум $5; карта/крипта).
       Без предоплаченного кредита инстанс создать нельзя. Типичный апскейл
       10-минутного 720p-ролика до 1080p стоит $0.3–2 в зависимости от GPU.

  2) API-ключ (даёт скрипту право искать офферы и создавать/удалять инстансы).
     • Откройте https://cloud.vast.ai/manage-keys/  → раздел "API Keys" → "Create new key".
     • Скопируйте ключ (показывается один раз) и вставьте ниже.
       Скрипт сохранит его в ~/.config/vastai/vast_api_key (права 600) — тот же файл,
       который использует официальная утилита `vastai`.  Альтернатива: переменная
       окружения VAST_API_KEY или параметр --api-key.

  3) SSH-ключ (для загрузки видео в контейнер и скачивания результата).
     • Скрипт сам найдёт ~/.ssh/id_ed25519.pub (или id_rsa.pub), при отсутствии —
       сгенерирует новый ключ ed25519 без пароля и загрузит публичную часть в
       аккаунт vast.ai через API (эквивалент https://cloud.vast.ai/manage-keys/ → "SSH Keys").
"""


def read_saved_api_key() -> Optional[str]:
    for p in (VAST_KEY_FILE, VAST_KEY_FILE_LEGACY):
        try:
            k = p.read_text(encoding="utf-8").strip()
            if k:
                return k
        except OSError:
            continue
    return None


def save_api_key(key: str) -> None:
    VAST_KEY_FILE.parent.mkdir(parents=True, exist_ok=True)
    VAST_KEY_FILE.write_text(key + "\n", encoding="utf-8")
    os.chmod(VAST_KEY_FILE, 0o600)


def find_ssh_key(explicit: Optional[str]) -> Tuple[Path, Path]:
    """Возвращает (приватный, публичный). Генерирует ed25519 при отсутствии."""
    ssh_dir = Path.home() / ".ssh"
    if explicit:
        priv = Path(explicit).expanduser()
        pub = Path(str(priv) + ".pub")
        if not priv.exists() or not pub.exists():
            raise FatalError(f"SSH-ключ {priv} или {pub} не найден.")
        return priv, pub
    for name in ("id_ed25519", "id_rsa", "id_ecdsa"):
        priv, pub = ssh_dir / name, ssh_dir / (name + ".pub")
        if priv.exists() and pub.exists():
            return priv, pub
    ssh_dir.mkdir(mode=0o700, exist_ok=True)
    priv = ssh_dir / "id_ed25519"
    info("SSH-ключ не найден — генерирую ~/.ssh/id_ed25519 (без пароля, чтобы работать без интерактива).")
    run(["ssh-keygen", "-t", "ed25519", "-N", "", "-f", str(priv), "-C", f"vast_upscale@{socket.gethostname()}"])
    return priv, Path(str(priv) + ".pub")


def pubkey_fingerprint_part(pub: str) -> str:
    """'ssh-ed25519 AAAA... comment' -> 'AAAA...' (для сравнения с ключами в аккаунте)."""
    parts = pub.strip().split()
    return parts[1] if len(parts) >= 2 else pub.strip()


DOCKERHUB_FILE = VAST_KEY_FILE.parent / "dockerhub.json"

DOCKERHUB_INSTRUCTIONS = """
  Для предсобранного образа (SeedVR2 + веса, собирается на этой машине) нужен ваш репозиторий на Docker Hub:

  1) Аккаунт: https://hub.docker.com/signup — бесплатный подходит (публичный репозиторий,
     ограничений по размеру образа нет).
  2) Токен доступа: https://hub.docker.com/settings/security → «New Access Token»,
     права Read & Write. Можно и пароль, но токен безопаснее (его можно отозвать).
  3) Имя репозитория, куда класть образы (создастся при первом push): по умолчанию <логин>/vast-upscale.

  Данные сохраняются в ~/.config/vastai/dockerhub.json (права 600); альтернатива — переменные
  окружения DOCKER_USER / DOCKER_PASS или параметры --docker-user / --docker-pass / --image-repo.
"""


def load_dockerhub_creds() -> dict:
    try:
        d = json.loads(DOCKERHUB_FILE.read_text(encoding="utf-8"))
        return d if isinstance(d, dict) else {}
    except (OSError, ValueError):
        return {}


def save_dockerhub_creds(user: str, token: str, repo: str) -> None:
    DOCKERHUB_FILE.parent.mkdir(parents=True, exist_ok=True)
    DOCKERHUB_FILE.write_text(json.dumps({"user": user, "token": token, "repo": repo}, indent=2), encoding="utf-8")
    os.chmod(DOCKERHUB_FILE, 0o600)


def ensure_dockerhub(args, auto_yes: Optional[bool], allow_install: bool = True) -> bool:
    """Сопряжение с Docker Hub для предсобранного образа: учётные данные из аргументов/окружения/файла,
    иначе — запрос у пользователя с инструкцией; проверка логина через API Hub; сохранение.
    Возвращает True, если предсобранный образ возможен (учётные данные есть и Docker доступен)."""
    if args.prebuild == "never":
        return False
    saved = load_dockerhub_creds()
    user = args.docker_user or os.environ.get("DOCKER_USER") or saved.get("user") or ""
    token = args.docker_pass or os.environ.get("DOCKER_PASS") or os.environ.get("DOCKER_TOKEN") or saved.get("token") or ""
    repo = args.image_repo or saved.get("repo") or (f"{user}/vast-upscale" if user else "")
    hub = DockerHub(user, token) if user and token else None
    if hub and RegistryClient(user, token).check_push_access(repo):
        ok(f"Docker Hub: доступ есть ({user}), репозиторий docker.io/{repo}")
    else:
        if hub:
            warn("Docker Hub: сохранённые учётные данные не приняты (истёк/отозван токен?)")
        if not sys.stdin.isatty() or auto_yes:
            warn("Учётных данных Docker Hub нет, а спросить некого (нет терминала / --yes). "
                 "Задайте DOCKER_USER/DOCKER_PASS или --docker-user/--docker-pass; образ собран не будет.")
            if args.prebuild in ("local", "strict"):
                raise FatalError("--prebuild требует учётные данные Docker Hub.")
            args.prebuild = "never"
            return False
        _raw_print(DOCKERHUB_INSTRUCTIONS)
        for attempt in range(3):
            try:
                user = input(f"  Логин Docker Hub{(' [' + user + ']') if user else ''} (пусто — без предсобранного образа): ").strip() or user
                if not user:
                    warn("Предсобранный образ отключён (--prebuild never). Включить позже: задайте учётные данные.")
                    args.prebuild = "never"
                    return False
                token = getpass.getpass("  Токен доступа (или пароль) Docker Hub, ввод скрыт: ").strip() or token
                repo = input(f"  Репозиторий для образов [{repo or user + '/vast-upscale'}]: ").strip() or repo or f"{user}/vast-upscale"
            except EOFError:
                raise UserAbort()
            if "/" not in repo:
                repo = f"{user}/{repo}"
            hub = DockerHub(user, token)
            if RegistryClient(user, token).check_push_access(repo):
                ok(f"Docker Hub: доступ подтверждён ({user}), репозиторий docker.io/{repo}")
                break
            warn(f"Docker Hub не принял логин/токен (попытка {attempt + 1}/3). Проверьте токен на https://hub.docker.com/settings/security")
        else:
            raise FatalError("Не удалось войти в Docker Hub.")
        save_dockerhub_creds(user, token, repo)
        ok(f"Учётные данные Docker Hub сохранены в {DOCKERHUB_FILE}")
    args.docker_user, args.docker_pass, args.image_repo = user, token, repo
    priv = hub.repo_private(repo) if hub else None
    if priv is not None:
        note(f"репозиторий {repo}: {'приватный (при заказе будет передан image_login)' if priv else 'публичный'}")
    if args.prebuild in ("auto", "local") and not shutil.which("pip3") and \
            subprocess.run([sys.executable, "-m", "pip", "--version"], capture_output=True).returncode != 0:
        warn("Для сборки образа нужен pip (sudo apt-get install -y python3-pip): им готовятся зависимости под Python контейнера.")
    return True


def ensure_pairing(api_key_arg: Optional[str], ssh_key_arg: Optional[str], auto_yes: Optional[bool]) -> Tuple[VastClient, Path, dict]:
    """Проверяет/запрашивает API-ключ, регистрирует SSH-ключ, показывает баланс."""
    key = api_key_arg or os.environ.get("VAST_API_KEY") or read_saved_api_key()
    if not key:
        _raw_print(PAIRING_INSTRUCTIONS)
        if not sys.stdin.isatty():
            raise FatalError("API-ключ не найден. Задайте VAST_API_KEY или --api-key, либо запустите в терминале.")
        while True:
            try:
                key = getpass.getpass("  Вставьте API-ключ vast.ai (ввод скрыт): ").strip()
            except EOFError:
                raise UserAbort()
            if key:
                break
    client = VastClient(key)
    try:
        user = client.show_user()
    except VastAPIError as e:
        if e.status in (401, 403):
            raise FatalError("API-ключ отклонён vast.ai (401/403). Проверьте ключ на https://cloud.vast.ai/manage-keys/")
        raise
    email = user.get("email") or user.get("username") or "?"
    credit = float(user.get("credit") or 0.0)
    ok(f"vast.ai: аккаунт {email}, баланс {fmt_money(credit)}")
    if not api_key_arg and not os.environ.get("VAST_API_KEY") and read_saved_api_key() != key:
        save_api_key(key)
        ok(f"API-ключ сохранён в {VAST_KEY_FILE}")
    if credit < 1.0:
        warn("Баланс меньше $1 — инстанс, скорее всего, не запустится. Пополните: https://cloud.vast.ai/billing/")
        if not ask_yes_no("Продолжить всё равно?", default=False, auto=auto_yes):
            raise UserAbort()

    priv, pub = find_ssh_key(ssh_key_arg)
    pub_text = pub.read_text(encoding="utf-8").strip()
    mine = pubkey_fingerprint_part(pub_text)
    try:
        keys = client.show_ssh_keys()
    except VastAPIError as e:
        warn(f"Не удалось получить список SSH-ключей аккаунта: {e}")
        keys = []
    registered = any(mine in str(k.get("public_key") or k.get("ssh_key") or k) for k in keys)
    if registered:
        ok(f"SSH-ключ {pub.name} уже зарегистрирован в аккаунте vast.ai")
    else:
        client.create_ssh_key(pub_text)
        ok(f"SSH-ключ {pub.name} загружен в аккаунт vast.ai")
    os.chmod(priv, 0o600)
    return client, priv, user

# ----------------------------------------------------------------------------
# Шаг 3. Видео: поиск, анализ (ffprobe), предконвертация под апскейлер
# ----------------------------------------------------------------------------
@dataclass
class VideoInfo:
    path: Path
    width: int
    height: int
    fps: float
    duration: float
    frames: int
    codec: str
    pix_fmt: str
    has_audio: bool
    vfr: bool
    size_bytes: int
    bit_depth: int = 8


@dataclass
class Job:
    """Один видеофайл на обработку."""
    src: Path
    info: VideoInfo
    converted: Optional[Path] = None      # локальный предконвертированный файл (CFR h264)
    conv_frames: int = 0
    conv_bytes: int = 0
    name: str = ""                        # уникальное короткое имя для удалённых путей
    out_path: Optional[Path] = None       # итоговый файл рядом с исходником
    remote_in: str = ""
    remote_out: str = ""
    downloaded: Optional[Path] = None     # скачанный апскейл (без звука)
    status: str = "pending"               # pending|uploaded|done|failed|downloaded|finished|skipped
    assigned_to: str = "main"             # какой инстанс обрабатывает: main | W1 | W2 …
    in_transfer: bool = False             # переезжает на другой инстанс (не трогать)
    error: str = ""
    est_out_bytes: int = 0
    proc_seconds: float = 0.0
    # блочная обработка: файл режется на блоки фиксированного размера, каждый блок — отдельное задание
    block_of: str = ""                    # имя родительского файла (для блока)
    block_idx: int = -1
    block_total: int = 0
    block_start: int = 0                  # первый кадр блока в исходнике (без контекста)
    block_frames: int = 0                 # кадров в блоке (без контекста)
    block_ctx: int = 0                    # контекстных кадров спереди (их выход отбрасывается)
    blocks: List[str] = field(default_factory=list)   # у родителя: имена блоков (status «split»)
    cache_dir: Optional[Path] = None      # кэш готовых чанков этого видео (<кэш>/chunks/<имя>-<hash требований>)
    assembled: bool = False               # родитель: блоки уже склеены и результат собран (сразу по готовности)

    @property
    def is_block(self) -> bool:
        return bool(self.block_of)

    @property
    def label(self) -> str:
        return f"{self.src.name} · блок {self.block_idx + 1}/{self.block_total}" if self.is_block else self.src.name


def fps_string(fps: float) -> str:
    """fps в виде дроби для ffmpeg: стандартные дробные значения точно (29.97 → 30000/1001)."""
    std = {23.976: "24000/1001", 29.97: "30000/1001", 59.94: "60000/1001", 24: "24", 25: "25", 30: "30",
           50: "50", 60: "60", 48: "48", 120: "120"}
    return next((v for k, v in std.items() if abs(fps - k) < 0.02), f"{fps:.4f}")


def _frac(s: Optional[str]) -> float:
    if not s or s in ("0/0", "N/A"):
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b) if float(b) else 0.0
    return float(s)


def ffprobe(path: Path) -> VideoInfo:
    cp = run(["ffprobe", "-v", "error", "-print_format", "json", "-show_format", "-show_streams", str(path)],
             check=False)
    if cp.returncode != 0:
        raise FatalError(f"ffprobe не смог прочитать {path}: {cp.stderr.strip()[:300]}")
    data = json.loads(cp.stdout)
    vs = [s for s in data.get("streams", []) if s.get("codec_type") == "video" and s.get("disposition", {}).get("attached_pic", 0) == 0]
    if not vs:
        raise FatalError(f"В файле нет видеопотока: {path}")
    v = vs[0]
    fmt = data.get("format", {})
    avg = _frac(v.get("avg_frame_rate"))
    rfr = _frac(v.get("r_frame_rate"))
    fps = avg or rfr or 30.0
    duration = float(v.get("duration") or fmt.get("duration") or 0.0)
    nb = int(v.get("nb_frames") or 0)
    if nb <= 0 and duration > 0:
        nb = int(round(duration * fps))
    vfr = bool(avg and rfr and abs(avg - rfr) / max(rfr, 1e-6) > 0.02)
    bit_depth = 8
    pf = v.get("pix_fmt") or ""
    if re.search(r"p?1[02]le|p?1[02]be|12|10", pf):
        bit_depth = 10
    return VideoInfo(
        path=path, width=int(v.get("width", 0)), height=int(v.get("height", 0)), fps=fps, duration=duration,
        frames=nb, codec=v.get("codec_name", "?"), pix_fmt=pf,
        has_audio=any(s.get("codec_type") == "audio" for s in data.get("streams", [])),
        vfr=vfr, size_bytes=int(fmt.get("size") or path.stat().st_size), bit_depth=bit_depth)


def collect_inputs(inputs: List[str], recursive: bool, suffix: str) -> List[Path]:
    files: List[Path] = []
    for raw in inputs:
        p = Path(raw).expanduser()
        if p.is_dir():
            it = p.rglob("*") if recursive else p.glob("*")
            for f in sorted(it):
                if f.is_file() and f.suffix.lower() in VIDEO_EXTS and ".vast_upscale" not in f.parts:
                    if f.stem.endswith(suffix):
                        continue  # это уже результат нашей работы
                    files.append(f)
        elif p.is_file():
            files.append(p)
        else:
            raise FatalError(f"Не найден файл/директория: {raw}")
    # убрать дубликаты, сохранив порядок
    seen, out = set(), []
    for f in files:
        rp = f.resolve()
        if rp not in seen:
            seen.add(rp)
            out.append(f)
    return out


def ffmpeg_supports_fps_mode() -> bool:
    try:
        v = run(["ffmpeg", "-version"], check=False).stdout.splitlines()[0]
        m = re.search(r"ffmpeg version n?(\d+)\.(\d+)", v)
        return not m or (int(m.group(1)), int(m.group(2))) >= (5, 1)
    except Exception:
        return True


def convert_for_upscaler(job: Job, workdir: Path, pre_downscale: int = 0, crf: int = 10) -> None:
    """H.264 8-бит yuv420p, постоянный fps, чётные стороны, без звука, faststart.
    SeedVR2 CLI читает кадры через OpenCV: переменный fps и экзотические pix_fmt ломают
    счёт кадров и синхронизацию со звуком — поэтому нормализуем заранее."""
    inf = job.info
    out = workdir / f"{job.name}.cfr.mp4"
    job.converted = out
    if out.exists() and out.stat().st_size > 0 and out.stat().st_mtime >= inf.path.stat().st_mtime:
        try:
            pi = ffprobe(out)
            if pi.frames > 0:
                job.conv_frames, job.conv_bytes = pi.frames, pi.size_bytes
                ok(f"{inf.path.name}: конвертированная копия уже есть ({pi.frames} кадров, {fmt_bytes(pi.size_bytes)})")
                return
        except FatalError:
            pass
        warn(f"{inf.path.name}: конвертированная копия повреждена (прерванный запуск?) — конвертирую заново")
        out.unlink(missing_ok=True)
    # целевой fps: округляем к стандартным дробным значениям (29.97 → 30000/1001)
    fps_str = fps_string(inf.fps)
    vf = []
    if pre_downscale and min(inf.width, inf.height) > pre_downscale:
        # для сильно деградированных источников иногда полезно сначала уменьшить
        if inf.width >= inf.height:
            vf.append(f"scale=-2:{pre_downscale}")
        else:
            vf.append(f"scale={pre_downscale}:-2")
    vf.append("scale=trunc(iw/2)*2:trunc(ih/2)*2")   # чётные стороны
    vf.append("format=yuv420p")
    tmp = out.with_suffix(".part.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(inf.path),
           "-map", "0:v:0", "-an", "-sn", "-dn"]
    cmd += ["-fps_mode", "cfr"] if ffmpeg_supports_fps_mode() else ["-vsync", "cfr"]
    cmd += ["-r", fps_str, "-vf", ",".join(vf), "-c:v", "libx264", "-preset", "medium", "-crf", str(crf),
            "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-progress", "pipe:1", str(tmp)]
    total = inf.frames or None
    bar = ProgressBar(total, f"Конвертация {inf.path.name}", unit=" кадр", est_total_sec=(inf.duration * 0.5 if inf.duration else None))
    debug(" ".join(shlex.quote(c) for c in cmd))
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    frame = 0
    assert proc.stdout is not None
    for line in proc.stdout:
        line = line.strip()
        if line.startswith("frame="):
            try:
                frame = int(line.split("=", 1)[1])
            except ValueError:
                pass
            bar.set(frame)
    proc.wait()
    if proc.returncode != 0:
        bar.close()
        errtxt = proc.stderr.read()[-500:] if proc.stderr else ""
        raise FatalError(f"ffmpeg не смог конвертировать {inf.path.name}: {errtxt}")
    tmp.replace(out)
    pi = ffprobe(out)
    job.conv_frames, job.conv_bytes = pi.frames, pi.size_bytes
    bar.close(f"{inf.path.name} → {out.name}: {pi.width}x{pi.height}, {pi.frames} кадров, {fmt_bytes(pi.size_bytes)}")


def nb_frames_fast(path: Path) -> int:
    """Число кадров по контейнеру (без декодирования); 0, если неизвестно."""
    cp = run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_packets", "-show_entries",
              "stream=nb_frames,nb_read_packets", "-of", "json", str(path)], check=False)
    try:
        st = json.loads(cp.stdout)["streams"][0]
        return int(st.get("nb_frames") or st.get("nb_read_packets") or 0)
    except (ValueError, KeyError, IndexError):
        return 0


def plan_block_size(args, spf: float, load_sec: float) -> int:
    """Размер блока в кадрах для оценок (без привязки к файлу): --block-frames N | 0 (без блоков) |
    auto — не длиннее --block-seconds секунд видео (при 30 fps — 240 кадров). Точный размер под fps файла — block_frames_for."""
    v = str(args.block_frames or "auto").lower()
    if v not in ("auto", ""):
        return max(0, int(v))
    return max(1, int(float(args.block_seconds) * 30))


def block_frames_for(job: Job, args) -> int:
    """Размер блока (чанка) для файла: не длиннее --block-seconds секунд (по умолчанию 8) при его fps; --block-frames N —
    явно (но не длиннее того же предела); 0 — без блоков. Зависит только от файла и настроек, поэтому раскладка
    блоков одинакова от запуска к запуску — готовые чанки переиспользуются из кэша."""
    cap = max(1, int(math.floor(float(args.block_seconds) * job.info.fps + 1e-6)))
    v = str(args.block_frames or "auto").lower()
    if v not in ("auto", ""):
        n = int(v)
        return 0 if n <= 0 else min(n, cap)
    return cap


def split_into_blocks(job: Job, workdir: Path, block_frames: int, ctx: int) -> List[Job]:
    """Режет сконвертированный файл на блоки по ~block_frames кадров (+ ctx кадров контекста спереди у каждого,
    кроме первого): кадрово-точный -ss по CFR-копии, повторное кодирование H.264 CRF 10. Возвращает блоки-задания;
    если файл короче 1.5 блока — пустой список (обрабатывается целиком). Идемпотентно: готовые блоки переиспользуются."""
    assert job.converted
    n = job.conv_frames
    if block_frames <= 0 or n <= block_frames:
        return []
    nblocks = max(2, int(math.ceil(n / block_frames)))      # каждый блок не длиннее block_frames (≤ --block-seconds)
    base, rem = divmod(n, nblocks)
    sizes = [base + (1 if i < rem else 0) for i in range(nblocks)]
    fps = job.info.fps
    blocks: List[Job] = []
    start = 0
    bar = ProgressBar(nblocks, f"Нарезка {job.src.name} на блоки", unit=" блок")
    for i, frames in enumerate(sizes):
        c = min(ctx, start)
        name = f"{job.name}.b{i:03d}"
        seg = workdir / f"{name}.cfr.mp4"
        want = c + frames
        if not (seg.exists() and seg.stat().st_size > 0 and seg.stat().st_mtime >= job.converted.stat().st_mtime
                and nb_frames_fast(seg) == want):
            tmp = seg.with_suffix(".part.mp4")
            ss = max(0.0, (start - c - 0.5) / fps)
            cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-accurate_seek", "-ss", f"{ss:.6f}",
                   "-i", str(job.converted), "-frames:v", str(want), "-an", "-sn", "-dn", "-c:v", "libx264", "-preset", "fast",
                   "-crf", "10", "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-fps_mode", "passthrough", str(tmp)]
            cp = run(cmd, check=False)
            if cp.returncode != 0:
                bar.close()
                raise FatalError(f"ffmpeg не смог вырезать блок {i + 1}/{nblocks} из {job.src.name}: {cp.stderr[-300:]}")
            got = nb_frames_fast(tmp)
            if got != want:
                bar.close()
                raise FatalError(f"блок {i + 1}/{nblocks} из {job.src.name}: вырезано {got} кадров вместо {want}")
            tmp.replace(seg)
        from dataclasses import replace as _replace
        info_b = _replace(job.info, path=seg, frames=want, duration=want / fps, size_bytes=seg.stat().st_size)
        b = Job(src=job.src, info=info_b, converted=seg, conv_frames=want, conv_bytes=seg.stat().st_size, name=name,
                block_of=job.name, block_idx=i, block_total=nblocks, block_start=start, block_frames=frames, block_ctx=c)
        b.remote_in = f"{REMOTE_ROOT}/in/{name}.cfr.mp4"
        b.remote_out = f"{REMOTE_ROOT}/out/{name}.mp4"
        blocks.append(b)
        start += frames
        bar.set(i + 1)
    bar.close(f"{job.src.name}: {nblocks} блоков по ~{base} кадров (+{ctx} кадров контекста), "
              f"{fmt_bytes(sum(b.conv_bytes for b in blocks))}")
    job.blocks = [b.name for b in blocks]
    job.status = "split"
    return blocks


def assemble_blocks(parent: Job, blocks: List[Job], workdir: Path) -> Path:
    """Склеивает скачанные блоки без перекодирования (одинаковый кодер/параметры у всех блоков);
    проверяет число кадров."""
    blocks = sorted(blocks, key=lambda b: b.block_idx)
    for b in blocks:
        if not (b.downloaded and b.downloaded.exists()):
            raise FatalError(f"{b.label}: результат не скачан")
    lst = workdir / f"{parent.name}.concat.txt"
    lst.write_text("".join(f"file '{str(b.downloaded).replace(chr(39), chr(39) + chr(92) + chr(39) + chr(39))}'\n" for b in blocks), encoding="utf-8")
    out = workdir / f"{parent.name}.up.mp4"
    tmp = out.with_suffix(".part.mp4")
    want = sum(b.block_frames for b in blocks)
    for extra in ([], ["-fflags", "+genpts"]):
        cp = run(["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", *extra, "-f", "concat", "-safe", "0",
                  "-i", str(lst), "-c", "copy", "-movflags", "+faststart", str(tmp)], check=False)
        if cp.returncode != 0:
            continue
        got = nb_frames_fast(tmp)
        if got == want:
            tmp.replace(out)
            lst.unlink(missing_ok=True)
            return out
        warn(f"{parent.src.name}: после склейки {got} кадров вместо {want}" + (" — пробую с genpts" if not extra else ""))
    raise FatalError(f"{parent.src.name}: склейка блоков не удалась (см. {lst})")


def chunk_requirements(job: Job, args, block_frames: int, block_ctx: int) -> dict:
    """«Выходные требования» чанков: исходник (путь, размер, mtime) и всё, что влияет на результат блока —
    цель, ограничение длинной стороны, модель, seed, доп. аргументы CLI, коммит SeedVR2, раскладка блоков."""
    src = job.src.resolve()
    st = src.stat()
    return {"src": str(src), "size": st.st_size, "mtime": int(st.st_mtime), "target": int(args.target),
            "max_long": int(args.max_long_side or 0), "model": str(args.model), "seed": int(args.seed),
            "extra": str(args.extra_args or ""), "pre_downscale": int(getattr(args, "pre_downscale", 0) or 0),
            "seedvr2": SEEDVR2_COMMIT[:7], "block_frames": int(block_frames), "block_ctx": int(block_ctx),
            "fps": fps_string(job.info.fps)}


def chunk_cache_dir(job: Job, req: dict) -> Path:
    key = hashlib.sha1(json.dumps(req, sort_keys=True).encode("utf-8")).hexdigest()[:12]
    return LOCAL_CACHE_DIR / "chunks" / f"{job.name}-{key}"


def chunk_marker(path: Path) -> Path:
    return path.with_suffix(".json")


def restore_cached_chunks(parent: Job, blocks: List[Job], req: dict) -> int:
    """Готовые чанки из кэша (<кэш>/chunks/<имя>-<hash требований>/<блок>.mp4 + .json): блок с маркером и совпадающим
    размером считается скачанным и не пересчитывается. Возвращает число восстановленных блоков."""
    d = chunk_cache_dir(parent, req)
    parent.cache_dir = d
    kept = 0
    for b in blocks:
        b.cache_dir = d
        p = d / f"{b.name}.mp4"
        m = chunk_marker(p)
        if p.exists() and m.exists():
            try:
                meta = json.loads(m.read_text(encoding="utf-8"))
                if int(meta.get("size", -1)) == p.stat().st_size and int(meta.get("frames", -1)) == b.block_frames:
                    b.status, b.downloaded = "downloaded", p
                    kept += 1
                    continue
            except (OSError, ValueError):
                pass
            m.unlink(missing_ok=True)
    try:
        d.mkdir(parents=True, exist_ok=True)
        (d / "meta.json").write_text(json.dumps(dict(req, video=parent.src.name, blocks=len(blocks)), indent=2, ensure_ascii=False),
                                     encoding="utf-8")
    except OSError:
        pass
    return kept


def chunk_cache_store(job: Job, size: int, sha: str) -> None:
    """Скачанный блок лежит в кэше чанков — записать маркер (переживает перезапуски скрипта)."""
    if not (job.is_block and job.downloaded and job.cache_dir):
        return
    try:
        chunk_marker(job.downloaded).write_text(json.dumps({"size": size, "sha256": sha, "frames": job.block_frames,
                                                            "start": job.block_start, "ctx": job.block_ctx,
                                                            "ts": time.time()}), encoding="utf-8")
    except OSError as e:
        debug(f"chunk marker: {e}")


def chunk_cache_clear(parent: Job) -> None:
    """Полный выходной файл собран — все чанки этого видео удаляются из кэша."""
    if parent.cache_dir and parent.cache_dir.exists():
        shutil.rmtree(parent.cache_dir, ignore_errors=True)
        note(f"{parent.src.name}: чанки удалены из кэша ({parent.cache_dir.name})")


class Assembler:
    """Сборка готового видео сразу, как только скачаны все его чанки (не дожидаясь остальных файлов):
    склейка без перекодирования → звук → результат рядом с исходником → чанки удаляются из кэша."""

    def __init__(self, parents: List[Job], blocks_by_parent: Dict[str, List[Job]], workdir: Path, args):
        self.parents = {p.name: p for p in parents}
        self.blocks = blocks_by_parent
        self.workdir, self.args = workdir, args
        self.lock = threading.Lock()
        self.threads: List[threading.Thread] = []
        self.results: Dict[str, Tuple[bool, str]] = {}

    def notify(self, job: Job) -> None:
        """Вызывается после скачивания блока (из любого потока-сессии)."""
        if not job.is_block:
            return
        p = self.parents.get(job.block_of)
        if not p:
            return
        with self.lock:
            blocks = self.blocks.get(p.name) or []
            if p.assembled or p.status != "split" or not blocks or not all(b.status == "downloaded" and b.downloaded for b in blocks):
                return
            p.assembled = True
        t = threading.Thread(target=self._assemble, args=(p, blocks), daemon=True, name=f"assemble-{p.name}")
        self.threads.append(t)
        t.start()

    def _assemble(self, p: Job, blocks: List[Job]) -> None:
        try:
            p.downloaded = assemble_blocks(p, blocks, self.workdir)
            p.status = "downloaded"
            out = mux_audio(p, f"{self.args.target}p")
            p.status = "finished"
            pi = ffprobe(out)
            ok(f"{out}  ({pi.width}x{pi.height}, {fmt_bytes(pi.size_bytes)}{', со звуком' if p.info.has_audio else ''}) — "
               f"собрано из {len(blocks)} чанков сразу по готовности")
            chunk_cache_clear(p)               # выходные чанки этого видео — из кэша; нарезка входа живёт до конца всей очереди
            if not self.args.keep_converted:
                try:
                    if p.downloaded:
                        p.downloaded.unlink()
                except OSError:
                    pass
            self.results[p.name] = (True, "")
        except (FatalError, OSError) as e:
            p.assembled = False
            p.status, p.error = "failed", str(e)
            err(f"{p.src.name}: сборка не удалась: {e}")
            self.results[p.name] = (False, str(e))

    def wait(self) -> None:
        for t in self.threads:
            t.join()


def mux_audio(job: Job, target_label: str) -> Path:
    """Собирает итог: апскейленное видео + звук из исходника (AAC), рядом с исходником."""
    assert job.downloaded and job.out_path
    tmp = job.out_path.with_name(job.out_path.stem + ".part.mp4")
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-i", str(job.downloaded)]
    if job.info.has_audio:
        cmd += ["-i", str(job.info.path), "-map", "0:v:0", "-map", "1:a:0?", "-c:v", "copy",
                "-c:a", "aac", "-b:a", "192k", "-shortest"]
    else:
        cmd += ["-map", "0:v:0", "-c:v", "copy"]
    cmd += ["-movflags", "+faststart", "-metadata", f"comment=upscaled to {target_label} with SeedVR2 via vast_upscale.py", str(tmp)]
    cp = run(cmd, check=False)
    if cp.returncode != 0:
        raise FatalError(f"ffmpeg mux не удался для {job.src.name}: {cp.stderr[-400:]}")
    tmp.replace(job.out_path)
    return job.out_path


# ----------------------------------------------------------------------------
# Шаг 4. План: выбор модели под VRAM, оценка времени/стоимости, поиск и ранжирование офферов
# ----------------------------------------------------------------------------
def pick_model(pref: str, vram_gb: float) -> Optional[dict]:
    """Возвращает конфигурацию SeedVR2 под доступную VRAM: файл DiT, batch, доп. флаги,
    множитель скорости относительно 3B fp16, время загрузки модели."""
    pref = pref.lower()
    if pref == "auto":
        pref = "7b" if vram_gb >= 40 else "3b"
    if pref == "3b":
        table = [(130, 89), (94, 57), (78, 45), (62, 33), (46, 25), (38, 17), (30, 13), (22, 9), (18, 5), (14, 5)]
        batch = next((b for v, b in table if vram_gb >= v), None)
        if batch is None:
            return None
        return dict(dit="seedvr2_ema_3b_fp16.safetensors", batch=batch, extra=[], spf_factor=1.0, load_sec=45,
                    label="SeedVR2-3B fp16")
    variant = "sharp" if pref == "7b-sharp" else ""
    fp16 = f"seedvr2_ema_7b{'_sharp' if variant else ''}_fp16.safetensors"
    fp8 = f"seedvr2_ema_7b{'_sharp' if variant else ''}_fp8_e4m3fn_mixed_block35_fp16.safetensors"
    if vram_gb >= 38:
        table = [(130, 65), (94, 41), (78, 29), (62, 21), (46, 13), (38, 9)]
        batch = next(b for v, b in table if vram_gb >= v)
        return dict(dit=fp16, batch=batch, extra=[], spf_factor=1.15, load_sec=90,
                    label=f"SeedVR2-7B{' sharp' if variant else ''} fp16")
    if vram_gb >= 30:
        return dict(dit=fp8, batch=9, extra=["--blocks_to_swap", "16"], spf_factor=1.8, load_sec=75,
                    label=f"SeedVR2-7B{' sharp' if variant else ''} fp8 (+blockswap)")
    if vram_gb >= 22:
        return dict(dit=fp8, batch=5, extra=["--blocks_to_swap", "24", "--vae_decode_tiled"], spf_factor=2.4,
                    load_sec=75, label=f"SeedVR2-7B{' sharp' if variant else ''} fp8 (+blockswap 24)")
    return None


@dataclass
class Plan:
    """Сводка по заданию для оценок."""
    jobs: List[Job]
    target: int
    model_pref: str
    total_frames: int
    upload_bytes: int
    est_download_bytes: int
    local_up_bps: float      # байт/сек, оценка нашего исходящего канала
    local_down_bps: float
    time_value: float        # $/час ожидания (для balanced)
    bid_margin: float
    disk_gb: float
    optimize: str = "cost"   # cost | ratio | balanced | speed
    req: Optional[Requirements] = None


def estimate_out_bytes(job: Job, target: int) -> int:
    """Оценка размера результата: libx264 crf 12 ≈ 12–20 Мбит/с при 1080p; масштабируем по площади."""
    inf = job.info
    short = min(inf.width, inf.height) or 720
    scale = target / short
    out_pixels = inf.width * inf.height * scale * scale
    mbps = 15.0 * out_pixels / (1920 * 1080)
    return int(inf.duration * mbps * 1e6 / 8) if inf.duration else int(job.conv_bytes * scale * scale * 0.8)


def estimate_requirements(jobs: List[Job], args, disk_gb: float) -> Requirements:
    out_mp = 0.0
    per_frame = 0.0
    for j in jobs:
        ow, oh = output_dims(j.info.width, j.info.height, args.target, args.max_long_side)
        out_mp = max(out_mp, ow * oh / 1e6)
        per_frame = max(per_frame, (j.info.width * j.info.height + ow * oh) * 3 * 4 * 2.5 / 2 ** 20)
    req = Requirements(out_mp=round(out_mp, 2), per_frame_mb=round(per_frame, 1), disk_gb=disk_gb)
    for dit in ("seedvr2_ema_3b_fp16.safetensors", "seedvr2_ema_7b_fp16.safetensors",
                "seedvr2_ema_7b_fp8_e4m3fn_mixed_block35_fp16.safetensors"):
        req.vram_gb[dit] = vram_required(dit, out_mp)
        req.ram_gb[dit] = ram_required(dit, per_frame)
    return req


def describe_requirements(req: Requirements) -> str:
    v3, v7 = req.vram_for("seedvr2_ema_3b_fp16.safetensors"), req.vram_for("seedvr2_ema_7b_fp16.safetensors")
    r3, r7 = req.ram_for("seedvr2_ema_3b_fp16.safetensors"), req.ram_for("seedvr2_ema_7b_fp16.safetensors")
    return (f"выход до {req.out_mp} МП → VRAM ≥ {v3:.0f} ГБ (3B) / {v7:.0f} ГБ (7B fp16) при batch {MIN_BATCH}, "
            f"RAM ≥ {r3:.0f} / {r7:.0f} ГБ, диск ≥ {req.disk_gb:.0f} ГБ")


def build_plan(jobs: List[Job], args) -> Plan:
    total_frames = sum(j.conv_frames or j.info.frames for j in jobs)
    upload = sum(j.conv_bytes for j in jobs)
    for j in jobs:
        j.est_out_bytes = estimate_out_bytes(j, args.target)
    dl = sum(j.est_out_bytes for j in jobs)
    # диск: образ распакованный ~10 ГБ + pip ~2 ГБ + веса + вход + выход + запас
    weights = 17.0 if args.model in ("7b", "7b-sharp") else (17.0 if args.model == "auto" else 7.3)
    disk = 14 + weights + (upload + dl) / 1e9 * 1.3 + 4
    if any(j.is_block for j in jobs):
        disk += 3            # PNG-кадры блока перед кодированием (~1 ГБ на 300 кадров 1080p)
    disk = float(args.disk) if args.disk else float(max(32, math.ceil(disk)))
    tv = 1.0 if args.time_value is None else float(args.time_value)
    optimize = "balanced" if args.time_value is not None and args.optimize == "cost" else args.optimize
    req = estimate_requirements(jobs, args, disk)
    return Plan(jobs=jobs, target=args.target, model_pref=args.model, total_frames=total_frames,
                upload_bytes=upload, est_download_bytes=dl,
                local_up_bps=args.local_upload_mbps * 1e6 / 8, local_down_bps=args.local_download_mbps * 1e6 / 8,
                time_value=tv, bid_margin=args.bid_margin, disk_gb=disk, optimize=optimize, req=req)


@dataclass
class Candidate:
    offer: dict
    kind: str                # "on-demand" | "bid"
    price: float             # $/ч, которые платим (dph_total или наш бид)
    model: dict
    t_setup: float
    t_upload: float
    t_proc: float
    t_download: float
    cost_total: float
    score: float
    warnings: List[str] = field(default_factory=list)

    @property
    def t_total(self) -> float:
        return self.t_setup + self.t_upload + self.t_proc + self.t_download

    @property
    def gpu(self) -> str:
        return self.offer.get("gpu_name", "?")

    @property
    def vram_gb(self) -> float:
        return round((self.offer.get("gpu_ram") or 0) / 1024.0)


def candidate_score(cost: float, hours: float, plan: Plan) -> float:
    """Критерий ранжирования (меньше — лучше):
    ratio    — стоимость всего прогона × его длительность ($·ч): дёшево И быстро, без ручного «курса» часа;
    balanced — стоимость + time_value $/ч × длительность;
    cost     — только стоимость прогона;  speed — только длительность."""
    if plan.optimize == "ratio":
        return cost * hours
    if plan.optimize == "cost":
        return cost
    if plan.optimize == "speed":
        return hours
    return cost + plan.time_value * hours


def score_label(plan: Plan) -> str:
    return {"ratio": "$·ч", "balanced": "$+t", "cost": "$", "speed": "ч"}.get(plan.optimize, "score")


def fmt_score(c: "Candidate", plan: Plan) -> str:
    if plan.optimize == "speed":
        return fmt_time(c.score * 3600)
    if plan.optimize == "cost":
        return fmt_money(c.score)
    return f"{c.score:.3f}"


def estimate_candidate(offer: dict, kind: str, plan: Plan, calib: Optional[float] = None) -> Optional[Candidate]:
    gpu = offer.get("gpu_name", "")
    prof = GPU_PROFILES.get(gpu)
    if prof is None:
        return None
    vram = round((offer.get("gpu_ram") or 0) / 1024.0)
    model = pick_model(plan.model_pref, vram)
    if model is None:
        return None
    ram_gb = (offer.get("cpu_ram") or 0) / 1024.0
    if plan.req is not None:
        fits = vram >= plan.req.vram_for(model["dit"]) and ram_gb >= plan.req.ram_for(model["dit"])
        if not fits and plan.model_pref == "auto" and "7b" in model["dit"]:
            model = pick_model("3b", vram)          # 7B не помещается под это разрешение — берём 3B
            fits = model is not None and vram >= plan.req.vram_for(model["dit"]) and ram_gb >= plan.req.ram_for(model["dit"])
        if not fits:
            return None
    if (offer.get("cuda_max_good") or 0) < prof["min_cuda"]:
        return None
    inet_down = max(float(offer.get("inet_down") or 50.0), 5.0) * 1e6 / 8   # Мбит/с -> байт/с
    inet_up = max(float(offer.get("inet_up") or 50.0), 5.0) * 1e6 / 8
    # 1) запуск контейнера: pull образа + старт + apt/pip + веса (HF CDN ~ до 120 МБ/с)
    weights_bytes = MODEL_FILES[model["dit"]]["size"] + MODEL_FILES[VAE_FILE]["size"]
    t_pull = IMAGE_SIZE_BYTES / inet_down          # пока образ скачивается (loading), GPU-время не тарифицируется
    t_setup = 90 + t_pull + 150 + weights_bytes / min(inet_down, 120e6) * 1.15
    # 2) загрузка входа
    t_upload = plan.upload_bytes / min(plan.local_up_bps, inet_down) + 5 * len(plan.jobs)
    # 3) обработка
    spf = (calib if calib else prof["spf"]) * model["spf_factor"]
    t_proc = sum((j.conv_frames or j.info.frames) * spf + model["load_sec"] for j in plan.jobs)
    # 4) скачивание результата
    t_download = plan.est_download_bytes / min(plan.local_down_bps, inet_up) + 5 * len(plan.jobs)
    price = float(offer.get("dph_total") or 0.0)
    warnings: List[str] = []
    if kind == "bid":
        mb = float(offer.get("min_bid") or offer.get("dph_base") or price)
        price = round(mb * (1.0 + plan.bid_margin) + 0.001, 4)
        # риск прерывания: считаем, что ждём в среднем +25% времени
        t_proc *= 1.25
        warnings.append("interruptible: возможны паузы, если нас перебьют по цене")
    t_total = t_setup + t_upload + t_proc + t_download
    hours = t_total / 3600
    paid_hours = max(0.0, t_total - t_pull) / 3600   # аренда GPU идёт с момента running
    storage_cost = float(offer.get("storage_cost") or 0.0) * plan.disk_gb * hours / 730.0
    # трафик: inet_*_cost в $/ГБ (по справке CLI); значения обычно ничтожны
    bw_cost = (plan.upload_bytes / 1e9) * float(offer.get("inet_down_cost") or 0) + \
              (plan.est_download_bytes / 1e9) * float(offer.get("inet_up_cost") or 0)
    # пересчёт dph_total под наш диск: dph_total включает allocated_storage из запроса
    cost = price * paid_hours + storage_cost + bw_cost
    score = candidate_score(cost, hours, plan)
    if float(offer.get("reliability2") or offer.get("reliability") or 1.0) < 0.97:
        warnings.append(f"надёжность хоста {float(offer.get('reliability2') or offer.get('reliability') or 0):.3f}")
    if (offer.get("cpu_ram") or 0) / 1024.0 < 24:
        warnings.append(f"мало RAM ({(offer.get('cpu_ram') or 0) / 1024.0:.0f} ГБ) — будет потоковый режим с малыми чанками")
    if not offer.get("direct_port_count"):
        warnings.append("нет прямых портов — SSH через прокси (медленнее передача)")
    return Candidate(offer=offer, kind=kind, price=price, model=model, t_setup=t_setup, t_upload=t_upload,
                     t_proc=t_proc, t_download=t_download, cost_total=cost, score=score, warnings=warnings)


def build_offer_query(args, plan: Plan, quiet: bool = False) -> dict:
    gpus = [g.strip() for g in args.gpu.split(",")] if args.gpu else DEFAULT_GPU_SET
    gpus = [g.replace("_", " ") for g in gpus]
    for g in gpus:
        if g not in GPU_PROFILES and not quiet:
            warn(f"GPU '{g}' нет в таблице профилей — оценка скорости для него невозможна, пропускаю.")
    gpus = [g for g in gpus if g in GPU_PROFILES]
    query = {
        "gpu_name": {"in": gpus},
        "num_gpus": {"eq": 1},
        "gpu_ram": {"gte": int(max(args.min_vram, plan.req.vram_min if plan.req else 0) * 1000)},   # МБ (×1000, как в CLI)
        "reliability": {"gte": float(args.min_reliability)},
        "inet_down": {"gte": float(args.min_inet)},
        "inet_up": {"gte": float(args.min_inet) / 2},
        "disk_space": {"gte": float(plan.disk_gb)},
        "cuda_max_good": {"gte": float(args.min_cuda)},
        "cpu_ram": {"gte": int(max(16.0, plan.req.ram_min if plan.req else 0) * 1000)},
    }
    if args.country:
        query["geolocation"] = {"in": [c.strip().upper() for c in args.country.split(",")]}
    if args.max_price:
        query["dph_total"] = {"lte": float(args.max_price)}
    return query


def rental_kinds(args) -> List[str]:
    return {"auto": ["on-demand", "bid"], "on-demand": ["on-demand"], "bid": ["bid"]}[args.rental]


def offer_price(offer: dict, kind: str, plan: Plan) -> float:
    """Цена, которую платим: on-demand — dph_total; bid — min_bid с надбавкой."""
    if kind == "bid":
        mb = float(offer.get("min_bid") or offer.get("dph_base") or offer.get("dph_total") or 0)
        return round(mb * (1.0 + plan.bid_margin) + 0.001, 4)
    return float(offer.get("dph_total") or 0.0)


def price_per_tflop(offer: dict, price: float) -> Optional[float]:
    """$/ч на TFLOP (по total_flops vast.ai); None, если TFLOPS неизвестны."""
    tf = float(offer.get("total_flops") or 0.0)
    if tf <= 0 or price <= 0:
        return None
    return price / tf


def offer_value(offer: dict, price: float) -> Optional[float]:
    """Производительность на доллар: DLPerf/$ (индекс dlperf vast.ai; для bid — по нашей цене), иначе TFLOPS/$."""
    perf = float(offer.get("dlperf") or 0.0) or float(offer.get("total_flops") or 0.0)
    if perf <= 0 or price <= 0:
        return None
    return perf / price


@dataclass
class Bargain:
    offer: dict
    kind: str
    price: float
    ppt: float          # $/ч на TFLOP
    median_ppt: float   # медиана по остальным офферам
    ratio: float        # во сколько раз дешевле медианы
    value: float = 0.0  # DLPerf/$ (или TFLOPS/$)


def list_market(client: VastClient, args, plan: Plan, exclude_machines: set, dit: Optional[str] = None) -> List[Bargain]:
    """Все подходящие офферы рынка, лучшие по DLPerf/$ первыми (для управления пулом). dit — модель предсобранного
    образа: офферы, которым по VRAM положена другая модель, отбрасываются (образ им не подойдёт)."""
    query = build_offer_query(args, plan, quiet=True)
    empty = Plan(jobs=[], target=plan.target, model_pref=plan.model_pref, total_frames=0, upload_bytes=0, est_download_bytes=0,
                 local_up_bps=plan.local_up_bps, local_down_bps=plan.local_down_bps, time_value=plan.time_value,
                 bid_margin=plan.bid_margin, disk_gb=plan.disk_gb, optimize=plan.optimize, req=plan.req)
    out: List[Bargain] = []
    for kind in rental_kinds(args):
        try:
            offers = client.search_offers(query, offer_type=kind, order=[["dph_total", "asc"]], limit=300, storage_gb=plan.disk_gb)
        except VastAPIError as e:
            debug(f"скан офферов ({kind}) не удался: {e}")
            continue
        for o in offers:
            if o.get("gpu_name") not in GPU_PROFILES or o.get("machine_id") in exclude_machines:
                continue
            c = estimate_candidate(o, kind, empty)              # не проходит по VRAM/RAM/CUDA под наш план?
            if c is None or (dit and c.model["dit"] != dit):
                continue
            price = offer_price(o, kind, plan)
            v = offer_value(o, price)
            if v is None:
                continue
            ppt = price_per_tflop(o, price) or 0.0
            out.append(Bargain(offer=o, kind=kind, price=price, ppt=ppt, median_ppt=0.0, ratio=0.0, value=v))
    out.sort(key=lambda b: -b.value)
    return out


def search_candidates(client: VastClient, plan: Plan, args, calib: Optional[float] = None) -> List[Candidate]:
    query = build_offer_query(args, plan)
    kinds = rental_kinds(args)
    cands: List[Candidate] = []
    for kind in kinds:
        try:
            offers = client.search_offers(query, offer_type=kind, order=[["dph_total", "asc"]], limit=200,
                                          storage_gb=plan.disk_gb)
        except VastAPIError as e:
            warn(f"Поиск офферов ({kind}) не удался: {e}")
            continue
        debug(f"{kind}: {len(offers)} офферов")
        for o in offers:
            c = estimate_candidate(o, kind, plan, calib)
            if c is None:
                continue
            if args.max_price and c.price > float(args.max_price):
                continue
            cands.append(c)
    cands.sort(key=lambda c: c.score)
    # дедупликация по machine_id: один хост — один лучший вариант
    seen, uniq = set(), []
    for c in cands:
        key = (c.offer.get("machine_id"), c.kind)
        if key in seen:
            continue
        seen.add(key)
        uniq.append(c)
    return uniq


def print_candidates(cands: List[Candidate], plan: Plan, n: int = 8, start: int = 0) -> None:
    sl = score_label(plan)
    show_score = plan.optimize != "cost"   # в режиме cost критерий = «$ за run», колонку не дублируем
    hdr = (f"  {'#':>2} {'GPU':<20} {'VRAM':>5} {'тип':<9} {'$/ч':>7} {'сеть↓/↑ Мбит':>13} {'надёжн.':>7} "
           f"{'модель':<24} {'~время run':>10} {'~$ за run':>9}" + (f" {sl:>7}" if show_score else ""))
    _raw_print(bold(hdr))
    for i, c in enumerate(cands[start:start + n], start + 1):
        o = c.offer
        rel = float(o.get("reliability2") or o.get("reliability") or 0)
        line = (f"  {i:>2} {c.gpu:<20} {c.vram_gb:>4.0f}G {c.kind:<9} {c.price:>7.3f} "
                f"{float(o.get('inet_down') or 0):>6.0f}/{float(o.get('inet_up') or 0):<6.0f} {rel:>7.3f} "
                f"{c.model['label']:<24} {fmt_time(c.t_total):>10} {fmt_money(c.cost_total):>9}"
                + (f" {fmt_score(c, plan):>7}" if show_score else ""))
        _raw_print(line if i > 1 else green(line))
        if c.warnings and i <= 3:
            note("  " + "; ".join(c.warnings))
    if start + n < len(cands):
        note(f"… ещё {len(cands) - start - n} офферов (m — показать)")
    crit = {"cost": "минимальная стоимость всего run (сортировка по «$ за run»)",
            "ratio": "стоимость run × длительность run ($·ч) — дёшево и быстро одновременно",
            "balanced": f"стоимость run + {fmt_money(plan.time_value)}/ч × длительность",
            "speed": "только длительность run"}[plan.optimize]
    if plan.req:
        note(f"Отбор: {describe_requirements(plan.req)}; офферы, где выбранная модель не помещается, скрыты.")
    note(f"Критерий: {crit}. «run» = запуск контейнера + установка + загрузка {fmt_bytes(plan.upload_bytes)} + "
         f"обработка {plan.total_frames} кадров + скачивание ~{fmt_bytes(plan.est_download_bytes)}; "
         f"«$ за run» = аренда × время + диск + трафик. Скорость GPU — оценка, уточняется после первого файла.")


def _num(x, fmt=".0f", default="?") -> str:
    try:
        return format(float(x), fmt)
    except (TypeError, ValueError):
        return default


def print_candidate_details(c: Candidate, plan: Plan, idx: int = 0) -> None:
    o = c.offer
    _raw_print(bold(f"  №{idx} (vast offer id {o.get('id')}): {c.gpu} {c.vram_gb:.0f} ГБ, {c.kind}, {fmt_money(c.price)}/ч"))
    note(f"хост {o.get('host_id')}, машина {o.get('machine_id')}, {o.get('geolocation')}; надёжность "
         f"{_num(o.get('reliability2') or o.get('reliability'), '.3f')}; CUDA {o.get('cuda_max_good')}, драйвер {o.get('driver_version')}")
    note(f"CPU {o.get('cpu_cores_effective') or o.get('cpu_cores')} ядер, RAM {_num((o.get('cpu_ram') or 0) / 1024)} ГБ; "
         f"диск {_num(o.get('disk_space'))} ГБ ({_num(o.get('disk_bw'))} МБ/с); сеть ↓{_num(o.get('inet_down'))}/↑{_num(o.get('inet_up'))} Мбит/с; "
         f"прямых портов {o.get('direct_port_count')}")
    note(f"цены: on-demand {fmt_money(o.get('dph_total'))}/ч, min_bid {fmt_money(o.get('min_bid'))}/ч, "
         f"диск {o.get('storage_cost')} $/ГБ/мес, трафик ↓{o.get('inet_down_cost')} ↑{o.get('inet_up_cost')} $/ГБ")
    note(f"модель {c.model['label']}, batch {c.model['batch']}; время: запуск+установка {fmt_time(c.t_setup)}, "
         f"загрузка {fmt_time(c.t_upload)}, обработка {fmt_time(c.t_proc)}, скачивание {fmt_time(c.t_download)} → "
         f"всего {fmt_time(c.t_total)}; стоимость run {fmt_money(c.cost_total)}; критерий {fmt_score(c, plan)} {score_label(plan)}")
    for w in c.warnings:
        note("⚠ " + w)


def ask_text(prompt: str, default: str = "") -> str:
    try:
        a = input(f"  {prompt}" + (f" [{default}]" if default != "" else "") + ": ").strip()
    except EOFError:
        raise UserAbort()
    return a if a else default


def ask_choice(prompt: str, current: str, choices: List[str]) -> str:
    while True:
        a = ask_text(f"{prompt} ({'/'.join(choices)})", current).lower()
        if a in choices:
            return a
        warn(f"Допустимо: {', '.join(choices)}")


def ask_float(prompt: str, current: float) -> float:
    while True:
        a = ask_text(prompt, str(current))
        try:
            return float(a.replace(",", "."))
        except ValueError:
            warn("Нужно число")


def edit_filters(args) -> None:
    """Интерактивное изменение фильтров/критерия подбора (пустой ввод — оставить как есть)."""
    info("Пустой ввод — оставить текущее значение; для списка GPU «-» = все известные.")
    g = ask_text("GPU через запятую (напр. RTX 4090,RTX 5090,H100 SXM)", args.gpu or "-")
    args.gpu = "" if g.strip() in ("-", "") else g
    args.rental = ask_choice("Тип аренды", args.rental, ["auto", "on-demand", "bid"])
    if args.rental != "on-demand":
        args.bid_margin = ask_float("Надбавка к min_bid при биде (0.25 = +25%)", args.bid_margin)
    args.max_price = ask_float("Потолок $/ч (0 = нет)", args.max_price)
    args.optimize = ask_choice("Критерий", args.optimize, ["cost", "ratio", "balanced", "speed"])
    if args.optimize == "balanced":
        args.time_value = ask_float("Сколько $ стоит час вашего ожидания", 1.0 if args.time_value is None else args.time_value)
    else:
        args.time_value = None
    args.model = ask_choice("Модель SeedVR2", args.model, ["auto", "3b", "7b", "7b-sharp"])
    args.min_vram = ask_float("Минимум VRAM, ГБ", args.min_vram)
    args.min_inet = ask_float("Минимум входящей скорости хоста, Мбит/с", args.min_inet)
    args.min_reliability = ask_float("Минимальная надёжность хоста (0–1)", args.min_reliability)
    c = ask_text("Страны через запятую (напр. DE,NL,FI; «-» = любые)", args.country or "-")
    args.country = "" if c.strip() in ("-", "") else c


def interactive_offer_selection(client: VastClient, plan: Plan, args, cands: List[Candidate],
                                rebuild_plan: Callable[[], Plan]) -> Tuple[Candidate, List[Candidate], Plan]:
    """Меню после таблицы офферов: Enter — #1, N — оффер N, iN — подробности, m — ещё,
    f — фильтры/критерий, r — повторить поиск, q — выход. Возвращает (выбор, список с выбором первым, план)."""
    shown = 8
    while True:
        if not cands:
            warn("Под текущие фильтры офферов нет.")
        else:
            print_candidates(cands, plan, n=shown)
        _raw_print("")
        info("Enter — взять #1 · N — выбрать оффер N · iN — подробности · m — ещё · f — фильтры/критерий · r — обновить поиск · q — выход")
        a = ask_text("Выбор", "1" if cands else "f").lower().replace(" ", "")
        if a in ("q", "quit", "выход", "в"):
            raise UserAbort()
        if a in ("m", "more", "ещё", "е"):
            shown = min(len(cands), shown + 8) if cands else 8
            continue
        if a in ("r", "refresh", "обновить", "о"):
            sp = Spinner("Повторяю поиск офферов")
            sp.tick("запрос к vast.ai")
            cands = search_candidates(client, plan, args)
            sp.close(f"Подходящих офферов: {len(cands)}")
            shown = 8
            continue
        if a in ("f", "filters", "фильтры", "ф"):
            edit_filters(args)
            plan = rebuild_plan()
            sp = Spinner("Ищу офферы с новыми фильтрами")
            sp.tick("запрос к vast.ai")
            cands = search_candidates(client, plan, args)
            sp.close(f"Подходящих офферов: {len(cands)}")
            shown = 8
            continue
        m = re.match(r"^(?:i|и|info)(\d+)$", a)
        if m and cands:
            idx = int(m.group(1))
            if 1 <= idx <= len(cands):
                print_candidate_details(cands[idx - 1], plan, idx)
            else:
                warn(f"Нет оффера с номером {idx}")
            continue
        if a.isdigit() and cands:
            idx = int(a)
            if 1 <= idx <= len(cands):
                chosen = cands[idx - 1]
                ordered = [chosen] + [c for c in cands if c is not chosen]
                return chosen, ordered, plan
            warn(f"Нет оффера с номером {idx} (показано {min(shown, len(cands))} из {len(cands)})")
            continue
        warn("Не понял ввод. Введите номер оффера, m, f, r, iN или q.")

# ----------------------------------------------------------------------------
# Шаг 5. Скрипты, выполняемые ВНУТРИ контейнера (генерируются и загружаются по SSH)
# ----------------------------------------------------------------------------
BOOTSTRAP_SH = r"""#!/usr/bin/env bash
# bootstrap.sh — сгенерирован vast_upscale.py. Ставит SeedVR2 CLI и скачивает веса.
set -u
ROOT=@@ROOT@@
OPT=@@OPT@@            # софт и веса живут здесь: эта директория входит в снапшот образа
STATUS=$ROOT/bootstrap.status
mkdir -p $ROOT/in $ROOT/out $ROOT/logs $OPT/models
echo $$ > $ROOT/bootstrap.pid
: > $STATUS
step(){ echo "STEP $1 $2 $(date +%s)" >> "$STATUS"; echo; echo "=== STEP $1 $2 $(date '+%H:%M:%S') ==="; }
fail(){ echo "FAIL $1 $(date +%s)" >> "$STATUS"; echo "!!! FAIL: $1"; exit 1; }
link_dirs(){
  [ -d $ROOT/models ] && [ ! -L $ROOT/models ] && rmdir $ROOT/models 2>/dev/null
  ln -sfn $OPT/models $ROOT/models
  [ -e $ROOT/seedvr2 ] && [ ! -L $ROOT/seedvr2 ] && rm -rf $ROOT/seedvr2
  ln -sfn $OPT/seedvr2 $ROOT/seedvr2
}
source $ROOT/bootstrap.env
export DEBIAN_FRONTEND=noninteractive PIP_DISABLE_PIP_VERSION_CHECK=1 PYTHONUNBUFFERED=1
# SSH-сессии на vast.ai не всегда наследуют PATH docker-образа — ищем python с torch явно
PY=""
for c in /opt/conda/bin/python3 /venv/main/bin/python3 /usr/local/bin/python3 /usr/bin/python3 python3; do
  if command -v "$c" >/dev/null 2>&1 && "$c" -c "import torch" >/dev/null 2>&1; then PY=$(command -v "$c"); break; fi
done
[ -n "$PY" ] || fail "python с установленным torch не найден в контейнере (образ $VU_IMAGE?)"
export PY
echo "$PY" > $ROOT/python_path
echo "python: $PY ($($PY -c 'import sys; print(sys.version.split()[0])'))"
[ -f $OPT/env.sh ] && . $OPT/env.sh      # предсобранный образ: /opt/vu/bin (ffmpeg), /opt/vu/pylib (зависимости под этот Python)

mark_ok(){ # file sha — маркер проверки + запись в .validation_cache.json (SeedVR2 тогда не пересчитывает хеш при загрузке)
  $PY - "$1" "$2" <<'EOF'
import json, os, sys
f, sha = sys.argv[1], sys.argv[2]
cache = os.path.join(os.path.dirname(f), ".validation_cache.json")
try: d = json.load(open(cache))
except Exception: d = {}
d[os.path.basename(f)] = {"size": os.path.getsize(f), "mtime": os.path.getmtime(f), "hash": sha}
json.dump(d, open(cache, "w"), indent=2)
EOF
  : > "$1.sha256ok"
}
weights_ok(){ # file size sha: файл на месте и проверен; без маркера — только пересчёт sha256 (ничего не скачивается)
  local f="$1" size="$2" sha="$3" have got
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  [ "$have" = "$size" ] || { echo "prebuilt: нет весов $(basename $f) (есть $have из $size байт)"; return 1; }
  [ -f "$f.sha256ok" ] && return 0
  echo "prebuilt: нет маркера проверки $(basename $f).sha256ok — считаю sha256 (без скачивания)"
  got=$(sha256sum "$f" | cut -d' ' -f1)
  [ "$got" = "$sha" ] || { echo "prebuilt: sha256 весов $(basename $f) не совпадает: $got"; return 1; }
  mark_ok "$f" "$sha"
}

# Быстрый путь: предсобранный образ — всё уже на месте; каждая проверка с диагностикой
IMPORTS="import cv2, diffusers, peft, einops, omegaconf, gguf, safetensors, rotary_embedding_torch, psutil, tqdm"
prebuilt_ok=1
weights_bad=0
if [ -f $OPT/.prebuilt ] || [ "${VU_PREBUILT:-0}" = "1" ]; then
  echo "prebuilt image: $(cat $OPT/.prebuilt 2>/dev/null | tr '\n' ' ')"
  weights_ok "$OPT/models/$DIT_FILE" "$DIT_SIZE" "$DIT_SHA" || { prebuilt_ok=0; weights_bad=1; }
  weights_ok "$OPT/models/$VAE_FILE" "$VAE_SIZE" "$VAE_SHA" || { prebuilt_ok=0; weights_bad=1; }
  [ -f $OPT/seedvr2/inference_cli.py ] || { echo "prebuilt: нет $OPT/seedvr2/inference_cli.py"; prebuilt_ok=0; }
  command -v ffmpeg >/dev/null 2>&1 || { echo "prebuilt: ffmpeg не найден в PATH ($PATH)"; prebuilt_ok=0; }
  $PY -c "$IMPORTS" 2>$ROOT/imports.err || { echo "prebuilt: не импортируются зависимости: $(tail -1 $ROOT/imports.err)"; prebuilt_ok=0; }
else
  prebuilt_ok=0
fi
if [ $prebuilt_ok = 1 ]; then
  step 5 verify
  link_dirs
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
  echo "DONE $(date +%s)" >> "$STATUS"
  echo "=== BOOTSTRAP DONE (prebuilt) ==="
  exit 0
fi
if [ "${VU_PREBUILT:-0}" = "1" ] && [ $weights_bad = 1 ]; then
  # инстанс заказан из предсобранного образа, а весов нужной модели в нём нет (или они битые): это не тот образ
  # или не та модель; качать 7–17 ГБ на арендованной машине — нельзя, инстанс будет заменён
  fail "prebuilt-verify"
fi
echo "Доустанавливаю только недостающее (что уже есть — не повторяется)"

step 1 apt
need=0
for c in ffmpeg rsync git curl; do command -v $c >/dev/null 2>&1 || need=1; done
python3 -c "import ctypes; ctypes.CDLL('libGL.so.1')" >/dev/null 2>&1 || need=1
if [ $need = 1 ]; then
  for i in 1 2 3 4; do
    apt-get update -qq >/dev/null 2>&1
    apt-get install -y -qq --no-install-recommends ffmpeg rsync git curl ca-certificates libgl1 libglib2.0-0 libsm6 libxext6 >/dev/null 2>&1 && break
    echo "apt retry $i"; sleep 15
  done
fi
for c in ffmpeg rsync git curl; do command -v $c >/dev/null 2>&1 || fail "apt: $c not installed"; done

step 2 clone
if [ -f $OPT/seedvr2/inference_cli.py ] && [ "$(cat $OPT/seedvr2/.commit 2>/dev/null)" = "$SEEDVR2_COMMIT" ]; then
  echo "SeedVR2 $SEEDVR2_COMMIT уже на месте (из образа)"
else
  if [ -e $OPT/seedvr2 ] && [ ! -d $OPT/seedvr2/.git ]; then
    echo "$OPT/seedvr2 есть, но это не git-клон нужного коммита — убираю"; rm -rf $OPT/seedvr2
  fi
  if [ ! -d $OPT/seedvr2/.git ]; then
    for i in 1 2 3; do git clone -q $SEEDVR2_REPO $OPT/seedvr2 && break; sleep 10; done
  fi
  [ -d $OPT/seedvr2/.git ] || fail "git clone"
  ( cd $OPT/seedvr2 && ( git cat-file -e $SEEDVR2_COMMIT 2>/dev/null || git fetch -q origin $SEEDVR2_COMMIT ) && git checkout -q $SEEDVR2_COMMIT ) || fail "git checkout $SEEDVR2_COMMIT"
  echo "$SEEDVR2_COMMIT" > $OPT/seedvr2/.commit
fi
link_dirs

step 3 pip
$PY -c "import torch, sys; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'gpu', torch.cuda.get_device_name(0)); sys.exit(0 if torch.cuda.is_available() else 1)" || fail "torch/CUDA недоступна (драйвер хоста?)"
if $PY -c "$IMPORTS" >/dev/null 2>&1; then
  echo "зависимости Python уже на месте"
else
  for i in 1 2 3; do $PY -m pip install -q --no-input -r $OPT/seedvr2/requirements.txt && break; echo "pip retry $i"; sleep 10; done
  $PY -c "$IMPORTS" || {
    $PY -m pip install -q --no-input opencv-python-headless || true
    $PY -c "$IMPORTS" || fail "python deps import"
  }
fi

step 4 weights
dl(){ # url file size sha
  local url="$1" f="$2" size="$3" sha="$4" have got rc
  have=$(stat -c %s "$f" 2>/dev/null || echo 0)
  if [ "$have" = "$size" ] && [ -f "$f.sha256ok" ]; then echo "$(basename $f): уже проверен"; return 0; fi
  for try in 1 2 3 4 5 6 7 8; do
    have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    if [ "$have" != "$size" ]; then
      [ "$have" -gt "$size" ] && rm -f "$f"
      # -C - докачивает с места обрыва; обрывы соединения обрабатывает внешний цикл
      curl -L -sS --fail --retry 5 --retry-delay 5 --connect-timeout 30 -C - -o "$f" "$url"; rc=$?
      if [ $rc = 33 ]; then echo "server has no range support, restarting $f"; rm -f "$f"; continue; fi
      have=$(stat -c %s "$f" 2>/dev/null || echo 0)
    fi
    if [ "$have" = "$size" ]; then
      echo "verifying $(basename $f)"
      got=$(sha256sum "$f" | cut -d' ' -f1)
      if [ "$got" = "$sha" ]; then
        mark_ok "$f" "$sha"
        return 0
      fi
      echo "sha256 mismatch for $f: $got"; rm -f "$f"
    else
      echo "size mismatch for $f: $have != $size (curl rc=${rc:-?})"
    fi
    sleep 5
  done
  return 1
}
dl "$VAE_URL" "$OPT/models/$VAE_FILE" "$VAE_SIZE" "$VAE_SHA" || fail "download $VAE_FILE"
dl "$DIT_URL" "$OPT/models/$DIT_FILE" "$DIT_SIZE" "$DIT_SHA" || fail "download $DIT_FILE"

step 5 verify
echo "$DIT_FILE $SEEDVR2_COMMIT" >> $OPT/.prebuilt
cp -f $ROOT/python_path $OPT/python_path 2>/dev/null || true
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
df -h $ROOT | tail -1
free -g | head -2
echo "DONE $(date +%s)" >> "$STATUS"
echo "=== BOOTSTRAP DONE ==="
"""

RUNNER_PY = r'''#!/usr/bin/env python3
# runner.py — сгенерирован vast_upscale.py. Обрабатывает очередь заданий SeedVR2 в контейнере,
# пишет прогресс в status.json, делает автоповтор при нехватке VRAM с более щадящими настройками.
import hashlib, json, os, re, subprocess, sys, time

ROOT = "@@ROOT@@"
JOBS = os.path.join(ROOT, "jobs.json")
STATUS = os.path.join(ROOT, "status.json")
REPO = os.path.join(ROOT, "seedvr2")
PHASES = {"Encoding": (0.00, 0.25), "Upscaling": (0.25, 0.35), "Decoding": (0.60, 0.30), "Post-processing": (0.90, 0.10)}
RE_BATCH = re.compile(r"(Encoding|Upscaling|Decoding|Post-processing) batch (\d+)/(\d+)")
RE_CHUNK = re.compile(r"Chunk (\d+)/(\d+):")
RE_FPS = re.compile(r"Average FPS: ([\d.]+)")
RE_OOM = re.compile(r"out of memory|OutOfMemoryError|cudaErrorMemoryAllocation|CUBLAS_STATUS_ALLOC_FAILED", re.I)


def write_status(st):
    st["ts"] = time.time()
    tmp = STATUS + ".tmp"
    with open(tmp, "w") as f:
        json.dump(st, f)
    os.replace(tmp, STATUS)


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def round_4n1(x, lo=1):
    n = max((int(x) - 1) // 4, 0)
    return max(lo, 4 * n + 1)


def free_ram_bytes():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) * 1024
    except Exception:
        pass
    return 16 * 1024 ** 3


def ram_budget():
    """(доступно байт, лимит контейнера байт|None): MemAvailable хоста, ограниченный cgroup-лимитом контейнера."""
    lim = None
    for pth in ("/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"):
        try:
            v = open(pth).read().strip()
            if v.isdigit() and int(v) < (1 << 60):
                lim = int(v) if lim is None else min(lim, int(v))
        except OSError:
            pass
    used = 0
    for pth in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        try:
            used = int(open(pth).read().strip())
            break
        except (OSError, ValueError):
            pass
    avail = free_ram_bytes()
    if lim:
        avail = min(avail, max(0, lim - used))
    return avail, lim


def gpu_mem_gb():
    """(свободно ГБ, всего ГБ) по nvidia-smi; (0, 0) если недоступно."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.free,memory.total", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=20).stdout.strip().splitlines()[0]
        free, total = [float(x) for x in out.split(",")]
        return free / 1024.0, total / 1024.0
    except Exception:
        return 0.0, 0.0


# Модель потребления VRAM SeedVR2 (консервативно, по опубликованным замерам):
# base — веса DiT+VAE+буферы; k — ГБ на кадр на мегапиксель ВЫХОДА (без тайлинга VAE пик в декодере ~1.5 ГБ/МП/кадр,
# с тайлингом остаётся DiT ~0.7 ГБ/МП/кадр для 3B и ~0.9 для 7B).
VRAM_MODEL = @@VRAM_MODEL@@
VRAM_K_UNTILED = @@VRAM_K_UNTILED@@


def vram_profile(dit):
    size = "7b" if "7b" in dit else "3b"
    prec = "fp8" if "fp8" in dit or ".gguf" in dit else "fp16"
    return VRAM_MODEL["%s_%s" % (size, prec)]


def plan_memory(job, cfg, batch_hint, blocks_swapped=0):
    """Подбирает batch (VRAM) и chunk (RAM контейнера) под реальные лимиты. Возвращает dict."""
    w, h, fr = job["width"], job["height"], job["frames"]
    scale = max(1.0, cfg["target"] / max(1, min(w, h)))
    out_w, out_h = w * scale, h * scale
    if cfg.get("max_long") and max(out_w, out_h) > cfg["max_long"]:
        f = cfg["max_long"] / max(out_w, out_h)
        out_w, out_h = out_w * f, out_h * f
    out_mp = out_w * out_h / 1e6
    base, k_tiled = vram_profile(cfg["dit"])
    if blocks_swapped:
        base = base * max(0.35, 1.0 - blocks_swapped / 36.0)
    vfree, vtotal = gpu_mem_gb()
    tiled = False
    if vfree > 0:
        budget = vfree - base - 1.0
        untiled = round_4n1(int(budget / (VRAM_K_UNTILED * out_mp))) if budget > 0 else 0
        tiled_b = round_4n1(int(budget / (k_tiled * out_mp))) if budget > 0 else 1
        # без тайлинга VAE быстрее, но батч меньше; большой батч важнее для временной согласованности
        if untiled >= 33 or (untiled >= 9 and tiled_b < 2 * untiled):
            batch = untiled
        else:
            tiled = True
            batch = max(1, tiled_b)
        batch = max(1, min(batch, 121))
        if batch_hint and batch_hint < batch:
            batch = batch_hint            # явный --batch-size пользователя / уменьшенный после OOM
    else:
        batch = batch_hint or int(cfg["batch"])
    # RAM: CLI держит кадры чанка как float32 + копии для цветокоррекции/энкодера
    per_frame = (w * h + out_w * out_h) * 3 * 4 * 2.5
    avail, lim = ram_budget()
    max_frames = int(avail * 0.5 / per_frame)
    chunk = round_4n1(min(8 * batch + 1, max(max_frames, 1)), lo=1)
    if chunk < batch:
        batch = round_4n1(chunk, lo=1)
    return {"batch": batch, "chunk": chunk if fr > chunk else 0, "tiled": tiled, "out_mp": round(out_mp, 2),
            "vram_free_gb": round(vfree, 1), "vram_total_gb": round(vtotal, 1), "ram_avail_gb": round(avail / 2 ** 30, 1),
            "ram_limit_gb": round(lim / 2 ** 30, 1) if lim else None, "per_frame_mb": round(per_frame / 2 ** 20, 1),
            "chunk_gb": round(chunk * per_frame / 2 ** 30, 2)}


def build_cmd(job, cfg, attempt, mem_state):
    extra = list(cfg.get("extra", []))
    swapped = int(extra[extra.index("--blocks_to_swap") + 1]) if "--blocks_to_swap" in extra else 0
    user_batch = int(cfg.get("user_batch") or 0)
    hint = user_batch or 0
    # лестница после OOM VRAM: тайлинг VAE → batch/2 → blockswap 16 → batch 5 + blockswap 32
    if attempt >= 2:
        hint = round_4n1(max(5, (mem_state.get("batch", cfg["batch"]) + 1) // 2), 5)
    if attempt >= 3:
        hint = max(5, round_4n1((hint + 1) // 2, 5))
        if "--blocks_to_swap" not in extra:
            extra += ["--blocks_to_swap", "16"]
            swapped = 16
    if attempt >= 4:
        hint = 5
        if "--blocks_to_swap" in extra:
            extra[extra.index("--blocks_to_swap") + 1] = "32"
        else:
            extra += ["--blocks_to_swap", "32"]
        swapped = 32
        if "--swap_io_components" not in extra:
            extra.append("--swap_io_components")
    mem = plan_memory(job, cfg, hint, swapped)
    if attempt >= 1:
        mem["tiled"] = True
    # после OOM по RAM (SIGKILL) — чанк вдвое меньше
    ram_div = mem_state.get("ram_div", 1)
    if ram_div > 1 and mem["chunk"]:
        mem["chunk"] = round_4n1(max(mem["chunk"] // ram_div, 1), lo=1)
    elif ram_div > 1:
        mem["chunk"] = round_4n1(max(job["frames"] // ram_div, 1), lo=1)
    if mem["tiled"]:
        for f in ("--vae_encode_tiled", "--vae_decode_tiled"):
            if f not in extra:
                extra.append(f)
    batch = mem["batch"]
    chunk_args = ["--chunk_size", str(mem["chunk"])] if mem["chunk"] else []
    # блок: выход — PNG-последовательность (контекстные кадры потом отбрасываются, кодирование — своим ffmpeg)
    out_args = [job["remote_out"] + ".png", "--output_format", "png"] if job.get("block") else [job["remote_out"]]
    cmd = [sys.executable, "-u", "inference_cli.py", job["remote_in"], "--output"] + out_args + [
           "--dit_model", cfg["dit"], "--model_dir", os.path.join(ROOT, "models"),
           "--resolution", str(cfg["target"]), "--max_resolution", str(cfg.get("max_long", 0)),
           "--batch_size", str(batch), "--uniform_batch_size", "--temporal_overlap", "3", "--prepend_frames", "4",
           "--color_correction", "lab", "--seed", str(cfg.get("seed", 42)), "--video_backend", "ffmpeg",
           "--attention_mode", "sdpa", "--debug"] + chunk_args + extra + list(cfg.get("user_extra", []))
    return cmd, batch, mem["chunk"], mem


def encode_block(job, out, log_path):
    """PNG-кадры блока (CLI пишет <out>.png/<stem>/<stem>_NNNNNN.png) → mp4 без контекстных кадров."""
    import glob, shutil
    b = job["block"]
    png_root = out + ".png"
    stem = os.path.splitext(os.path.basename(job["remote_in"]))[0]
    png_dir = os.path.join(png_root, stem)
    files = sorted(glob.glob(os.path.join(png_dir, stem + "_*.png")))
    want = int(job["frames"])
    if len(files) < want:
        return False, "блок: CLI записал %d кадров вместо %d (%s)" % (len(files), want, png_dir)
    first = int(files[0].rsplit("_", 1)[1].split(".")[0])
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error", "-nostdin", "-y", "-framerate", str(b["fps"]),
           "-start_number", str(first + int(b["ctx"])), "-i", os.path.join(png_dir, stem + "_%06d.png"),
           "-frames:v", str(int(b["frames"])), "-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "medium", "-crf", "12",
           "-movflags", "+faststart", out + ".part.mp4"]
    with open(log_path, "a") as log:
        log.write("\n===== encode: %s =====\n" % " ".join(cmd))
        cp = subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    if cp.returncode != 0 or not os.path.exists(out + ".part.mp4"):
        return False, "блок: ffmpeg не смог закодировать кадры (rc=%s), см. лог" % cp.returncode
    os.replace(out + ".part.mp4", out)
    shutil.rmtree(png_root, ignore_errors=True)
    return True, ""


def run_job(job, cfg, st, idx, total):
    name = job["name"]
    log_path = os.path.join(ROOT, "logs", name + ".log")
    max_attempts = 5
    mem_state = {"ram_div": 1}
    for attempt in range(max_attempts):
        cmd, batch, chunk, mem = build_cmd(job, cfg, attempt, mem_state)
        mem_state["batch"] = batch
        st.update({"job": name, "idx": idx, "total": total, "attempt": attempt + 1, "phase": "start",
                   "chunk": [1, 1], "batch": [0, 1], "pct": 0.0, "started": time.time(), "frames": job["frames"],
                   "batch_size": batch, "chunk_size": chunk, "cmd": " ".join(cmd), "mem": mem})
        write_status(st)
        oom = False
        tail = []
        chunk_i, chunk_n = 1, 1
        with open(log_path, "a") as log:
            log.write("\n\n===== attempt %d: %s =====\n" % (attempt + 1, " ".join(cmd)))
            log.flush()
            env = dict(os.environ, PYTHONUNBUFFERED="1", PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True")
            proc = subprocess.Popen(cmd, cwd=REPO, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                    env=env, errors="replace")
            last_write = 0.0
            for line in proc.stdout:
                log.write(line)
                tail.append(line.rstrip())
                if len(tail) > 60:
                    tail.pop(0)
                m = RE_CHUNK.search(line)
                if m:
                    chunk_i, chunk_n = int(m.group(1)), int(m.group(2))
                    st["chunk"] = [chunk_i, chunk_n]
                m = RE_BATCH.search(line)
                if m:
                    phase, bi, bn = m.group(1), int(m.group(2)), int(m.group(3))
                    base, wgt = PHASES[phase]
                    within = base + wgt * (bi / max(bn, 1))
                    st["phase"] = phase
                    st["batch"] = [bi, bn]
                    st["pct"] = ((chunk_i - 1) + within) / max(chunk_n, 1)
                m = RE_FPS.search(line)
                if m:
                    st.setdefault("fps", {})[name] = float(m.group(1))
                if RE_OOM.search(line):
                    oom = True
                if time.time() - last_write > 1.0:
                    log.flush()
                    write_status(st)
                    last_write = time.time()
            rc = proc.wait()
            log.flush()
        out = job["remote_out"]
        if rc == 0 and job.get("block"):
            # блок: PNG-кадры → отбросить контекст → кодировать так же, как это делает сам CLI (libx264 crf 12 medium)
            st["phase"] = "encode"
            write_status(st)
            ok_enc, msg = encode_block(job, out, log_path)
            if not ok_enc:
                st.setdefault("failed", {})[name] = msg[:2000]
                write_status(st)
                return False
        if rc == 0 and os.path.exists(out) and os.path.getsize(out) > 0:
            st["phase"] = "checksum"
            write_status(st)
            info = {"size": os.path.getsize(out), "sha256": sha256_file(out), "seconds": time.time() - st["started"],
                    "attempt": attempt + 1, "batch_size": batch, "fps": st.get("fps", {}).get(name)}
            with open(out + ".done", "w") as f:
                json.dump(info, f)
            st.setdefault("done", []).append(name)
            st["pct"] = 1.0
            write_status(st)
            return True
        if rc in (-9, 137) and attempt + 1 < max_attempts:
            # убит OOM-killer'ом контейнера: не хватило RAM, а не VRAM — уменьшаем чанк, batch не трогаем
            mem_state["ram_div"] = mem_state.get("ram_div", 1) * 2
            st.setdefault("events", []).append("%s: процесс убит по памяти контейнера (RAM) на попытке %d — чанк в %d раз меньше"
                                               % (name, attempt + 1, mem_state["ram_div"]))
            write_status(st)
            try:
                os.remove(out)
            except OSError:
                pass
            continue
        if oom and attempt + 1 < max_attempts:
            st.setdefault("events", []).append("%s: OOM VRAM на попытке %d (batch %d%s) — пробую щадящие настройки"
                                               % (name, attempt + 1, batch, ", tiled VAE" if mem.get("tiled") else ""))
            write_status(st)
            try:
                os.remove(out)
            except OSError:
                pass
            continue
        if not oom and attempt == 0 and rc != 0:
            st.setdefault("events", []).append("%s: ошибка (rc=%s) на 1-й попытке — повтор с тайлингом VAE" % (name, rc))
            write_status(st)
            continue
        msg = "rc=%s; %s" % (rc, " | ".join(tail[-8:]))
        st.setdefault("failed", {})[name] = msg[:2000]
        write_status(st)
        return False
    st.setdefault("failed", {})[name] = "исчерпаны попытки (OOM)"
    write_status(st)
    return False


def main():
    pidfile = os.path.join(ROOT, "runner.pid")
    try:
        old = int(open(pidfile).read().strip())
        os.kill(old, 0)
        print("runner.py уже работает (pid %d) — выходим" % old)
        return
    except (OSError, ValueError):
        pass
    with open(pidfile, "w") as f:
        f.write(str(os.getpid()))
    st = {"finished": False, "done": [], "failed": {}, "events": [], "fps": {}, "pid": os.getpid(), "queue": []}
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True).stdout.strip()
        st["gpu"] = gpu
    except Exception:
        pass
    write_status(st)
    processed = set()
    STOP = os.path.join(ROOT, "stop")
    # jobs.json перечитывается перед каждым файлом: оркестратор подгружает чанки из своей очереди по одному
    # (и может убрать ещё не начатый), пока идёт обработка; когда заданий нет — runner ждёт, а не выходит
    while not os.path.exists(STOP):
        try:
            if open(pidfile).read().strip() != str(os.getpid()):
                print("runner.py: pid-файл принадлежит другому процессу — выходим")
                return
        except OSError:
            return                      # каталог удалён — нам здесь больше нечего делать
        try:
            with open(JOBS) as f:
                data = json.load(f)
        except (OSError, ValueError):
            time.sleep(2)
            continue
        cfg, jobs = data["config"], data["jobs"]
        nxt = None
        for job in jobs:
            if job["name"] in processed:
                continue
            if os.path.exists(job["remote_out"] + ".done"):
                processed.add(job["name"])
                if job["name"] not in st["done"]:
                    st["done"].append(job["name"])
                continue
            nxt = job
            break
        st["queue"] = [j["name"] for j in jobs if j["name"] not in processed and (nxt is None or j["name"] != nxt["name"])]
        if nxt is None:
            if not st.get("idle"):
                st["idle"], st["job"], st["phase"] = True, None, "idle"
                write_status(st)
            time.sleep(2)
            continue
        st["idle"] = False
        processed.add(nxt["name"])
        if not os.path.exists(nxt["remote_in"]):
            st["failed"][nxt["name"]] = "входной файл не найден в контейнере"
            write_status(st)
            continue
        run_job(nxt, cfg, st, len(processed), len(jobs))
    st["finished"] = True
    st["idle"] = True
    st["job"] = None
    st["phase"] = "finished"
    write_status(st)


if __name__ == "__main__":
    main()
'''


def render_runner(root: str) -> str:
    return (RUNNER_PY.replace("@@ROOT@@", root)
            .replace("@@VRAM_MODEL@@", json.dumps({k: list(v) for k, v in VRAM_MODEL.items()}))
            .replace("@@VRAM_K_UNTILED@@", repr(VRAM_K_UNTILED)))


def render_bootstrap_env(model: dict, image: str = DEFAULT_IMAGE, prebuilt: bool = False) -> str:
    dit = model["dit"]
    d, v = MODEL_FILES[dit], MODEL_FILES[VAE_FILE]
    hf = "https://huggingface.co/{repo}/resolve/main/{file}"
    lines = [
        f"VU_IMAGE={image}",
        f"VU_PREBUILT={1 if prebuilt else 0}",
        f"SEEDVR2_REPO={SEEDVR2_REPO}",
        f"SEEDVR2_COMMIT={SEEDVR2_COMMIT}",
        f"DIT_FILE={dit}", f"DIT_URL={hf.format(repo=d['repo'], file=dit)}", f"DIT_SIZE={d['size']}", f"DIT_SHA={d['sha256']}",
        f"VAE_FILE={VAE_FILE}", f"VAE_URL={hf.format(repo=v['repo'], file=VAE_FILE)}", f"VAE_SIZE={v['size']}", f"VAE_SHA={v['sha256']}",
    ]
    return "\n".join(lines) + "\n"

# ----------------------------------------------------------------------------
# SSH / rsync к контейнеру
# ----------------------------------------------------------------------------
_RSYNC_PROGRESS = re.compile(r"(\d[\d,.]*)\s+(\d+)%\s+(\S+/s)")
RSYNC_FATAL_CODES = {1, 2, 3, 4}   # ошибка синтаксиса/протокола/выбора файлов — повторять бессмысленно


class SSH:
    def __init__(self, host: str, port: int, key: Path, tag: str):
        self.host, self.port, self.key = host, int(port), key
        ctl_dir = Path(os.environ.get("XDG_RUNTIME_DIR") or "/tmp")
        self.control = ctl_dir / f"vu-ssh-{tag}-{os.getpid()}"

    def opts(self) -> List[str]:
        return ["-i", str(self.key), "-p", str(self.port), "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", "-o", "LogLevel=ERROR", "-o", "BatchMode=yes",
                "-o", "ConnectTimeout=20", "-o", "ServerAliveInterval=15", "-o", "ServerAliveCountMax=4",
                "-o", "ControlMaster=auto", "-o", f"ControlPath={self.control}", "-o", "ControlPersist=600"]

    @property
    def target(self) -> str:
        return f"root@{self.host}"

    def run(self, cmd: str, timeout: float = 120) -> subprocess.CompletedProcess:
        full = ["ssh"] + self.opts() + [self.target, cmd]
        debug("ssh: " + cmd[:200])
        try:
            return subprocess.run(full, capture_output=True, text=True, timeout=timeout, errors="replace")
        except subprocess.TimeoutExpired:
            return subprocess.CompletedProcess(full, 255, "", "timeout")

    def test(self) -> bool:
        return self.run("echo ok", timeout=40).returncode == 0

    def write_file(self, remote_path: str, content: str, mode: str = "644") -> None:
        """Записывает текст в файл контейнера через stdin ssh (без scp — работает и до установки rsync)."""
        full = ["ssh"] + self.opts() + [self.target,
                f"mkdir -p $(dirname {shlex.quote(remote_path)}) && cat > {shlex.quote(remote_path)} && chmod {mode} {shlex.quote(remote_path)}"]
        cp = subprocess.run(full, input=content, capture_output=True, text=True, timeout=120)
        if cp.returncode != 0:
            raise FatalError(f"Не удалось записать {remote_path}: {cp.stderr.strip()[:300]}")

    force_pipe = False          # тесты: принудительно обходной канал без rsync
    _remote_rsync: Optional[bool] = None

    def remote_has_rsync(self) -> bool:
        if self._remote_rsync is None:
            cp = self.run("command -v rsync >/dev/null 2>&1 && echo yes || echo no", timeout=40)
            self._remote_rsync = "yes" in cp.stdout
        return self._remote_rsync

    def pipe_transfer(self, sources: List[str], dest: str, upload: bool, desc: str, total_bytes: Optional[int],
                      progress: Optional[ProgressBar] = None) -> Tuple[int, str]:
        """Передача с докачкой без rsync (образы без него): дописываем хвост файла через ssh-канал
        (tail -c +N | cat >>). Работает на любом контейнере с coreutils."""
        own = progress is None
        bar = progress or ProgressBar(total_bytes, desc, bytes_mode=True)
        done_total = 0
        try:
            for src in sources:
                if upload:
                    lp = Path(src)
                    size = lp.stat().st_size
                    rdest = dest.rstrip("/") + "/" + lp.name if dest.endswith("/") else dest
                    cp = self.run(f"mkdir -p $(dirname {shlex.quote(rdest)}); stat -c %s {shlex.quote(rdest)} 2>/dev/null || echo 0", timeout=60)
                    try:
                        have = int(cp.stdout.strip().splitlines()[-1])
                    except (ValueError, IndexError):
                        have = 0
                    if have > size:
                        self.run(f": > {shlex.quote(rdest)}", timeout=60)
                        have = 0
                    if have < size:
                        proc = subprocess.Popen(["ssh"] + self.opts() + [self.target, f"cat >> {shlex.quote(rdest)}"],
                                                stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
                        with open(lp, "rb") as f:
                            f.seek(have)
                            while True:
                                chunk = f.read(4 << 20)
                                if not chunk:
                                    break
                                proc.stdin.write(chunk)
                                have += len(chunk)
                                bar.set(done_total + have)
                        proc.stdin.close()
                        rc = proc.wait()
                        if rc != 0:
                            return rc, proc.stderr.read().decode("utf-8", "replace")[-300:]
                    done_total += size
                else:
                    cp = self.run(f"stat -c %s {shlex.quote(src)} 2>/dev/null || echo 0", timeout=60)
                    try:
                        size = int(cp.stdout.strip().splitlines()[-1])
                    except (ValueError, IndexError):
                        size = 0
                    lp = Path(dest)
                    if lp.is_dir():
                        lp = lp / Path(src).name
                    have = lp.stat().st_size if lp.exists() else 0
                    if have > size:
                        lp.unlink()
                        have = 0
                    if have < size:
                        proc = subprocess.Popen(["ssh"] + self.opts() + [self.target, f"tail -c +{have + 1} {shlex.quote(src)}"],
                                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        with open(lp, "ab") as f:
                            while True:
                                chunk = proc.stdout.read(4 << 20)
                                if not chunk:
                                    break
                                f.write(chunk)
                                have += len(chunk)
                                bar.set(done_total + have)
                        rc = proc.wait()
                        if rc != 0:
                            return rc, proc.stderr.read().decode("utf-8", "replace")[-300:]
                    done_total += size
            if own:
                bar.close()
            return 0, ""
        except (OSError, subprocess.SubprocessError) as e:
            if own:
                bar.render(force=True)
                _raw_print("")
            return 30, str(e)

    def rsync(self, sources: List[str], dest: str, upload: bool, desc: str, total_bytes: Optional[int],
              progress: Optional[ProgressBar] = None, timeout_idle: int = 90) -> Tuple[int, str]:
        """rsync с докачкой (--partial --inplace) и разбором прогресса. Возвращает (rc, последняя ошибка).
        Если rsync на инстансе нет (предсобранный образ без apt) — обходной канал pipe_transfer с той же семантикой."""
        if self.force_pipe or not self.remote_has_rsync():
            return self.pipe_transfer(sources, dest, upload, desc, total_bytes, progress)
        ssh_cmd = "ssh " + " ".join(shlex.quote(o) for o in self.opts())
        cmd = ["rsync", "-a", "--partial", "--inplace", "--info=progress2", "--no-inc-recursive",
               f"--timeout={timeout_idle}", "-e", ssh_cmd]
        if upload:
            cmd += sources + [f"{self.target}:{dest}"]
        else:
            cmd += [f"{self.target}:{s}" for s in sources] + [dest]
        debug(" ".join(shlex.quote(c) for c in cmd))
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, errors="replace",
                                bufsize=0)
        own_bar = progress is None
        bar = progress or ProgressBar(total_bytes, desc, bytes_mode=True)
        buf = ""
        assert proc.stdout is not None
        while True:
            ch = proc.stdout.read(1)
            if not ch:
                break
            if ch in ("\r", "\n"):
                m = _RSYNC_PROGRESS.search(buf)
                if m:
                    done = float(re.sub(r"[,.]", "", m.group(1)) or 0)
                    if total_bytes:
                        bar.set(min(total_bytes, done if done > 1 else total_bytes * int(m.group(2)) / 100.0),
                                extra=m.group(3))
                    else:
                        bar.set(done, extra=f"{m.group(2)}% {m.group(3)}")
                buf = ""
            else:
                buf += ch
        rc = proc.wait()
        errtxt = proc.stderr.read().strip() if proc.stderr else ""
        if own_bar:
            if rc == 0:
                bar.close()
            else:
                bar.render(force=True)
                _raw_print("")
        return rc, errtxt

    def close(self):
        try:
            subprocess.run(["ssh"] + self.opts() + ["-O", "exit", self.target], capture_output=True, timeout=15)
        except Exception:
            pass


# ----------------------------------------------------------------------------
# Жизненный цикл инстанса vast.ai
# ----------------------------------------------------------------------------
BAD_STATUSES = {"exited", "unknown", "offline", "error"}


def endpoints_of(inst: dict) -> List[Tuple[str, int, str]]:
    """Список кандидатов (host, port, вид): сначала прямое подключение, потом прокси."""
    eps: List[Tuple[str, int, str]] = []
    ports = inst.get("ports") or {}
    ip = inst.get("public_ipaddr")
    try:
        p22 = ports.get("22/tcp") or []
        if ip and p22 and p22[0].get("HostPort"):
            eps.append((ip, int(p22[0]["HostPort"]), "direct"))
    except (TypeError, ValueError, AttributeError):
        pass
    if inst.get("ssh_host") and inst.get("ssh_port"):
        eps.append((inst["ssh_host"], int(inst["ssh_port"]), "proxy"))
    return eps


def instance_state_line(inst: dict) -> str:
    return (f"actual={inst.get('actual_status')} intended={inst.get('intended_status')} "
            f"cur={inst.get('cur_state')} next={inst.get('next_state')} msg={str(inst.get('status_msg') or '')[:80]}")


_destroy_lock = threading.Lock()
_state_lock = threading.RLock()


class Session:
    """Оркестратор: провижининг, установка, загрузка, обработка, скачивание, очистка."""

    def __init__(self, client: VastClient, key: Path, plan: Plan, args, workdir: Path,
                 jobs: Optional[List[Job]] = None, tag: str = "main"):
        self.client, self.key, self.plan, self.args, self.workdir = client, key, plan, args, workdir
        self.jobs = jobs if jobs is not None else plan.jobs
        self.split_params: Optional[dict] = None     # блочная обработка: размер блока/контекст (для --resume)
        self.tag = tag
        self.is_worker = tag != "main"
        self.parent: Optional["Session"] = None      # у воркера — основная сессия (для state.json)
        self.root = REMOTE_ROOT
        self.image: str = args.image
        self.image_login: Optional[str] = None
        self.image_fn: Optional[Callable[[dict], Tuple[str, Optional[str], bool]]] = None  # образ под модель кандидата
        self.prebuilt = False
        self.link_mbps: Tuple[float, float] = (0.0, 0.0)
        self.last_status: dict = {}          # последний status.json (для планировщика пула)
        self.ready = False                   # runner запущен: инстанс может принимать файлы
        self.expensive = False               # дороже остального пула более чем на --expensive-factor
        self.retired = False                 # новых файлов не получает; удаляется, как только освободится
        self.on_poll: Optional[Callable[[dict], None]] = None      # оркестратор: скан рынка на каждом опросе
        self.orch: Optional["Orchestrator"] = None   # пул: очередь чанков, ребаланс, скан рынка
        self.jobs_by_name: Dict[str, Job] = {}
        self.board_line: Optional[Callable[[], str]] = None        # оркестратор: строка прогресса воркеров
        self.state_extra_fn: Optional[Callable[[], dict]] = None   # оркестратор: что добавить в state.json
        self.on_downloaded: Optional[Callable[[Job], None]] = None # сборщик: чанк скачан → собрать видео, если все готовы
        self.instance_id: Optional[int] = None
        self.instance: Optional[dict] = None
        self.cand: Optional[Candidate] = None
        self.model: Optional[dict] = None
        self.ssh: Optional[SSH] = None
        self.created_at: Optional[float] = None
        self.excluded_machines: set = set()
        self.we_stopped = False
        self.destroyed_at: Optional[float] = None
        self.calib_spf: Optional[float] = None
        self.state_file = workdir / "state.json"
        self.total_stages = 8
        self.credit_before: Optional[float] = None
        self.price_hint: float = 0.0
        self.t_ordered: Optional[float] = None       # когда заказан (для --boot-deadline)
        self.booted = False                          # provision + bootstrap завершены
        self.boot_error: str = ""
        self.boot_exc: Optional[BaseException] = None
        self.adopted = False                         # инстанс передан основной сессии (первым поднялся при наборе пула)
        self.interrupted = False                     # перебили по цене / хост забрал GPU: ждём возврата, бид растёт
        self.interrupted_at: Optional[float] = None
        self.bid_cap: Optional[float] = None         # потолок бида при возврате — цена соседнего взятого оффера
        self.lost = False                            # инстанс потерян окончательно; пул продолжает без него
        self.on_interrupt: Optional[Callable[["Session"], None]] = None       # оркестратор: замена + потолок бида
        self.on_interrupt_poll: Optional[Callable[["Session"], None]] = None  # оркестратор: пока машина недоступна
        self.on_lost: Optional[Callable[["Session"], bool]] = None            # оркестратор: машина не вернулась
        self._interrupt_lock = threading.Lock()
        self._waiter: Optional[threading.Thread] = None   # основной: ожидание возврата машины в фоне (общий цикл не блокируется)
        self._wait_result: Optional[bool] = None
        self.lost_since: Optional[float] = None
        self.external_lost = False                   # инстанс удалён в обход скрипта (замечено по списку инстансов)
        self.external_stopped = False                # инстанс не running по списку инстансов (перебили / остановили)

    def adopt(self, other: "Session") -> None:
        """Забирает поднятый инстанс у сессии-кандидата: он становится основным."""
        for a in ("instance_id", "instance", "created_at", "cand", "model", "ssh", "root", "image", "image_login", "prebuilt",
                  "link_mbps", "price_hint", "t_ordered", "booted"):
            setattr(self, a, getattr(other, a))
        other.adopted = True
        other.ssh = None
        other.instance_id = None
        self.save_state()

    def opt_dir(self) -> str:
        """Где в контейнере живут софт и веса (входит в снапшот образа)."""
        return REMOTE_OPT_TEMPLATE.format(iid=self.instance_id) if REMOTE_OPT_TEMPLATE else REMOTE_OPT

    # ---------- пути файлов на этом инстансе ----------
    def assign_paths(self, jobs: Optional[List[Job]] = None) -> None:
        for j in (jobs if jobs is not None else self.jobs):
            j.assigned_to = self.tag
            j.remote_in = f"{self.root}/in/{j.name}.cfr.mp4"
            j.remote_out = f"{self.root}/out/{j.name}.mp4"

    # ---------- состояние (для --resume) ----------
    def save_state(self):
        if self.is_worker:
            if self.parent:
                self.parent.save_state()
            return
        st = {
            "version": VERSION, "instance_id": self.instance_id, "created_at": self.created_at,
            "offer": {k: self.cand.offer.get(k) for k in ("id", "machine_id", "gpu_name", "gpu_ram", "dph_total", "min_bid",
                                                       "inet_down", "inet_up")} if self.cand else None,
            "kind": self.cand.kind if self.cand else None, "price": self.cand.price if self.cand else None,
            "model": self.model, "target": self.plan.target, "split": getattr(self, "split_params", None),
            "image": self.image, "prebuilt": self.prebuilt,
            "jobs": {j.name: {"src": str(j.src), "status": j.status, "assigned_to": j.assigned_to, "remote_in": j.remote_in,
                              "remote_out": j.remote_out, "downloaded": str(j.downloaded) if j.downloaded else None,
                              "out": str(j.out_path)} for j in self.jobs},
        }
        if self.state_extra_fn:
            try:
                st.update(self.state_extra_fn())
            except Exception:  # noqa
                pass
        with _state_lock:                     # пишут несколько потоков (воркеры через parent) — атомарно и по очереди
            tmp = self.state_file.with_suffix(".tmp")
            try:
                tmp.write_text(json.dumps(st, indent=2, ensure_ascii=False), encoding="utf-8")
                tmp.replace(self.state_file)
            except OSError as e:
                debug(f"state.json: {e}")

    def load_state(self) -> Optional[dict]:
        try:
            return json.loads(self.state_file.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None

    # ---------- провижининг ----------
    def provision(self, cands: List[Candidate]) -> None:
        """Заказывает инстанс по лучшему офферу; при неудаче — следующий. Ждёт статус running и SSH."""
        attempts = 0
        for cand in cands:
            if cand.offer.get("machine_id") in self.excluded_machines:
                continue
            if attempts >= self.args.max_attempts:
                break
            attempts += 1
            o = cand.offer
            info(f"Заказываю: {cand.gpu} {cand.vram_gb:.0f} ГБ, {cand.kind}, {fmt_money(cand.price)}/ч, "
                 f"оффер #{o.get('id')} (хост {o.get('host_id')}, машина {o.get('machine_id')}, {o.get('geolocation', '?')})")
            label = f"vast_upscale {_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}"
            if self.image_fn:
                self.image, self.image_login, self.prebuilt = self.image_fn(cand.model)
            note(f"образ: {self.image}{' (предсобранный)' if self.prebuilt else ''}")
            try:
                r = self.client.create_instance(
                    offer_id=int(o["id"]), image=self.image, disk_gb=self.plan.disk_gb, env={},
                    onstart=None, label=label, price=(cand.price if cand.kind == "bid" else None),
                    runtype="ssh_direc ssh_proxy" if o.get("direct_port_count") else "ssh_proxy", cancel_unavail=True,
                    image_login=self.image_login)
            except VastAPIError as e:
                warn(f"Создать инстанс не удалось: {e.msg}")
                if "credit" in e.msg.lower() or "balance" in e.msg.lower():
                    raise FatalError("Недостаточно средств на балансе vast.ai. Пополните: https://cloud.vast.ai/billing/")
                self.excluded_machines.add(o.get("machine_id"))
                continue
            if not r.get("success"):
                warn(f"vast.ai отказал: {r.get('msg') or r.get('error') or r}")
                self.excluded_machines.add(o.get("machine_id"))
                continue
            self.instance_id = int(r["new_contract"])
            self.root = remote_root_for(self.instance_id)
            self.cand, self.model = cand, cand.model
            self.created_at = time.time()
            self.we_stopped = False
            ok(f"Инстанс создан: #{self.instance_id} (метка «{label}»)")
            self.save_state()
            try:
                self.wait_running(timeout=self.args.boot_timeout)
                self.connect_ssh(timeout=600)
                self.check_link(strict=True)
                return
            except (FatalError, TimeoutError) as e:
                warn(f"Инстанс #{self.instance_id} не заработал: {e}. Удаляю и пробую следующий оффер.")
                self.destroy(quiet=True)
                self.excluded_machines.add(o.get("machine_id"))
                self.instance_id, self.cand, self.model = None, None, None
        raise FatalError("Не удалось получить рабочий инстанс ни по одному из офферов. "
                         "Попробуйте позже, ослабьте фильтры (--min-reliability, --min-inet, --gpu) или увеличьте --max-attempts.")

    def refresh_instance(self) -> Optional[dict]:
        try:
            self.instance = self.client.show_instance(self.instance_id)
        except VastAPIError as e:
            if e.status == 404:
                self.instance = None
            else:
                raise
        return self.instance

    def wait_running(self, timeout: float) -> dict:
        """Ждёт actual_status == running. Образ ~4 ГБ: обычно 1–5 мин, при холодном pull до 15."""
        inet = float((self.cand.offer.get("inet_down") if self.cand else 0) or 200) * 1e6 / 8
        sp = Spinner("Жду запуска контейнера", est_total_sec=min(timeout, 90 + IMAGE_SIZE_BYTES / inet))
        t0 = time.time()
        bad_since: Optional[float] = None
        last_line = ""
        while True:
            inst = self.refresh_instance()
            if inst is None:
                raise FatalError("инстанс исчез (удалён хостом или vast.ai)")
            st = inst.get("actual_status") or "created"
            line = instance_state_line(inst)
            if line != last_line:
                debug(line)
                last_line = line
            sp.tick(f"{st} {str(inst.get('status_msg') or '')[:60]}")
            if st == "running":
                sp.close(f"Контейнер запущен за {fmt_time(time.time() - t0)}: {inst.get('gpu_name')} на {inst.get('public_ipaddr') or inst.get('ssh_host')}")
                return inst
            if st in BAD_STATUSES or (inst.get("intended_status") == "stopped" and not self.we_stopped):
                bad_since = bad_since or time.time()
                if time.time() - bad_since > 120:
                    sp.close()
                    raise FatalError(f"состояние {st} более 2 минут ({inst.get('status_msg')})")
            else:
                bad_since = None
            if time.time() - t0 > timeout:
                sp.close()
                raise TimeoutError(f"контейнер не запустился за {fmt_time(timeout)} (статус {st})")
            time.sleep(self.args.poll_interval)

    def connect_ssh(self, timeout: float = 600) -> SSH:
        sp = Spinner("Проверяю SSH-доступ", est_total_sec=60)
        t0 = time.time()
        while time.time() - t0 < timeout:
            inst = self.instance or self.refresh_instance()
            if inst is None:
                raise FatalError("инстанс исчез")
            eps = endpoints_of(inst)
            for host, port, kind in eps:
                sp.tick(f"{kind} {host}:{port}")
                ssh = SSH(host, port, self.key, tag=str(self.instance_id))
                if ssh.test():
                    self.ssh = ssh
                    sp.close(f"SSH работает ({kind}): ssh -p {port} -i {self.key} root@{host}")
                    self.save_state()
                    return ssh
            time.sleep(8)
            self.refresh_instance()
        sp.close()
        raise FatalError("SSH не поднялся за отведённое время")

    # ---------- фактическая полоса инстанса ----------
    LINK_TEST_SH = r"""
DL_URL='https://speed.cloudflare.com/__down?bytes=150000000'
UP_URL='https://speed.cloudflare.com/__up'
HF_URL='https://huggingface.co/numz/SeedVR2_comfyUI/resolve/main/ema_vae_fp16.safetensors'
if command -v curl >/dev/null 2>&1; then
  D=$(curl -s -o /dev/null -w '%{speed_download}' --max-time 20 "$DL_URL" 2>/dev/null); D=${D:-0}
  # Cloudflare недоступен/заблокирован — меряем по CDN Hugging Face (оттуда же качаем веса)
  if [ "${D%.*}" -lt 1000000 ] 2>/dev/null; then
    D2=$(curl -sL -o /dev/null -w '%{speed_download}' --max-time 20 "$HF_URL" 2>/dev/null); D2=${D2:-0}
    [ "${D2%.*}" -gt "${D%.*}" ] 2>/dev/null && D=$D2
  fi
  head -c 30000000 /dev/zero > /tmp/vu_up.bin 2>/dev/null
  U=$(curl -s -o /dev/null -w '%{speed_upload}' --max-time 20 -X POST -H 'Content-Type: application/octet-stream' --data-binary @/tmp/vu_up.bin "$UP_URL" 2>/dev/null); U=${U:-0}
  rm -f /tmp/vu_up.bin
  echo "LINK $D $U"
else
  PY=""; for c in /opt/conda/bin/python3 /venv/main/bin/python3 /usr/bin/python3 python3; do command -v $c >/dev/null 2>&1 && PY=$c && break; done
  [ -n "$PY" ] && $PY - "$DL_URL" "$UP_URL" "$HF_URL" <<'EOF'
import sys, time, urllib.request
dl, up, hf = sys.argv[1], sys.argv[2], sys.argv[3]
def down_from(url):
    t0 = time.time(); n = 0
    with urllib.request.urlopen(url, timeout=25) as r:
        while time.time() - t0 < 20:
            b = r.read(1 << 20)
            if not b:
                break
            n += len(b)
    return n / max(time.time() - t0, 1e-3)
def down():
    try:
        d = down_from(dl)
    except Exception:
        d = 0.0
    if d < 1e6:
        try:
            d = max(d, down_from(hf))
        except Exception:
            pass
    return d
def upl():
    data = b"\0" * (30 << 20); t0 = time.time()
    try:
        urllib.request.urlopen(urllib.request.Request(up, data=data, method="POST",
                               headers={"Content-Type": "application/octet-stream"}), timeout=40).read()
    except Exception:
        return 0.0
    return len(data) / max(time.time() - t0, 1e-3)
try:
    d = down()
except Exception:
    d = 0.0
print("LINK %.0f %.0f" % (d, upl()))
EOF
fi
"""

    def check_link(self, strict: bool) -> Tuple[float, float]:
        """Измеряет реальную полосу инстанса (загрузка/отдача в интернет через speed.cloudflare.com)
        и канал от вас до инстанса по ssh. При strict и полосе ниже --min-link — FatalError
        (инстанс будет удалён, возьмём следующий оффер)."""
        assert self.ssh
        if self.args.min_link <= 0:
            return (0.0, 0.0)
        sp = Spinner("Проверяю фактическую полосу инстанса", est_total_sec=45)
        sp.tick("скачивание/отдача через Cloudflare")
        cp = self.ssh.run(self.LINK_TEST_SH, timeout=120)
        m = re.search(r"LINK\s+([\d.]+)\s+([\d.]+)", cp.stdout)
        down = float(m.group(1)) * 8 / 1e6 if m else 0.0
        up = float(m.group(2)) * 8 / 1e6 if m else 0.0
        # канал от вас к инстансу (ограничен вашим аплинком) — только для информации
        sp.tick("канал от вас к инстансу")
        t0 = time.time()
        try:
            proc = subprocess.run(["ssh"] + self.ssh.opts() + [self.ssh.target, "cat > /dev/null"],
                                  input=b"\0" * (12 << 20), capture_output=True, timeout=90)
            to_inst = (12 << 20) * 8 / 1e6 / max(time.time() - t0, 1e-3) if proc.returncode == 0 else 0.0
        except (subprocess.TimeoutExpired, OSError):
            to_inst = 0.0
        self.link_mbps = (down, up)
        sp.close()
        verdict = f"полоса инстанса: ↓{down:.0f} / ↑{up:.0f} Мбит/с; от вас к инстансу ~{to_inst:.0f} Мбит/с"
        if not m:
            warn(verdict + " (тест не отработал — curl/python в контейнере?)")
            return (down, up)
        if down <= 0:
            warn(verdict + " — тест скорости не отработал (Cloudflare и HF CDN недоступны из контейнера?), порог не применяю")
            return (down, up)
        bad = down < self.args.min_link or (0 < up < self.args.min_link)
        if bad:
            msg = f"{verdict} — ниже порога {self.args.min_link:.0f} Мбит/с"
            if strict:
                raise FatalError("медленный канал: " + msg)
            warn(msg)
        else:
            ok(verdict)
        return (down, up)

    # ---------- остановка/удаление ----------
    def destroy(self, quiet: bool = False) -> None:
        with _destroy_lock:
            if not self.instance_id:
                return
            iid, self.instance_id = self.instance_id, None      # повторный вызов (из другого потока) — no-op
        if self.ssh:
            self.ssh.close()
            self.ssh = None
        try:
            try:
                self.we_stopped = True
                self.client.stop_instance(iid)
            except VastAPIError:
                pass
            self.client.destroy_instance(iid)
            # проверка
            time.sleep(2)
            try:
                inst = self.client.show_instance(iid)
            except VastAPIError:
                inst = None
            if not quiet:
                if inst is None:
                    ok(f"Инстанс #{iid} остановлен и удалён (тарификация прекращена).")
                else:
                    warn(f"Инстанс #{iid}: запрос на удаление отправлен, текущее состояние {inst.get('actual_status')}. "
                         f"Проверьте https://cloud.vast.ai/instances/")
        except VastAPIError as e:
            if getattr(e, "status", None) == 404:
                if not quiet:
                    note(f"Инстанс #{iid} уже не существует (удалён в обход скрипта)")
            else:
                err(f"Не удалось удалить инстанс #{iid}: {e}. Удалите вручную: https://cloud.vast.ai/instances/ "
                    f"или `python3 {Path(sys.argv[0]).name} --destroy-instance {iid}`")
        finally:
            self.destroyed_at = time.time()
            self.save_state()

    # ---------- обработка прерывания (interruptible) ----------
    def ensure_alive(self, context: str) -> bool:
        """Проверяет, что контейнер существует и работает. Если нас перебили по цене (или хост забрал GPU) —
        ждёт возврата, каждые 5 с повышая бид до потолка. Возвращает False, если инстанс потерян."""
        if self.lost or not self.instance_id:
            return False
        inst = self.refresh_instance()
        if inst is None:
            err(f"[{context}] инстанс #{self.instance_id} больше не существует")
            return False
        st = inst.get("actual_status")
        if st == "running":
            return True
        if (inst.get("intended_status") == "stopped" and not self.we_stopped) or st in BAD_STATUSES:
            why = "перебили по цене" if (self.cand and self.cand.kind == "bid") else "остановлен vast.ai / хостом"
            warn(f"[{context}] инстанс недоступен ({st}/{inst.get('intended_status')}"
                 f"{(': ' + str(inst.get('status_msg'))[:60]) if inst.get('status_msg') else ''}) — {why}. Жду возврата…")
            return self.try_resume()
        return True  # loading/rebooting и т.п. — подождём

    def is_bid(self) -> bool:
        return bool(self.cand and self.cand.kind == "bid")

    def wait_back_async(self, context: str) -> None:
        """Основной инстанс: проверка/ожидание возврата машины в фоновом потоке — общий цикл (очередь, пул, прогресс)
        не останавливается из-за одной выбывшей машины."""
        if self._waiter is not None and self._waiter.is_alive():
            return
        self._wait_result = None

        def run():
            try:
                self._wait_result = self.ensure_alive(context)
            except Exception as e:  # noqa
                debug(f"wait_back: {e!r}")
                self._wait_result = False

        self._waiter = threading.Thread(target=run, daemon=True, name=f"wait-{self.tag}")
        self._waiter.start()

    def waiting_back(self) -> bool:
        return self._waiter is not None and self._waiter.is_alive()

    def raise_bid(self, price: float, cap: float, inst: dict) -> float:
        """Шаг повышения бида (раз в 5 с): к текущему min_bid машины с запасом, иначе +3 %, но не выше потолка."""
        mb = float(inst.get("min_bid") or 0.0)
        margin = float(getattr(self.args, "bid_margin", 0.1) or 0.0)
        new = round(max(price * 1.03 + 0.001, mb * (1.0 + margin) if mb else 0.0), 4)
        new = min(new, cap)
        if new <= price + 1e-9:
            return price
        try:
            self.client.change_bid(self.instance_id, new)
        except VastAPIError as e:
            note(f"Не удалось изменить бид: {e.msg}")
            return price
        if self.cand:
            self.cand.price = new
        return new

    def try_resume(self) -> bool:
        """Ждёт возвращения недоступного инстанса. Interruptible: раз в 5 с бид повышается — до потолка
        (цена соседнего взятого оффера пула или --max-price / +5 % к цене оффера). В пуле первый вызов уведомляет
        оркестратор (замена, если это была единственная готовая машина); повторные вызовы из других потоков
        (например, докачки) просто ждут. По возвращении SSH переподключается, очередь перезаписывается (часть блоков
        могли забрать другие машины) и runner перезапускается."""
        with self._interrupt_lock:
            first = not self.interrupted
            if first:
                self.interrupted, self.interrupted_at = True, time.time()
        if not first:
            t_end = (self.interrupted_at or time.time()) + self.args.resume_timeout + 30
            while self.interrupted and time.time() < t_end:
                time.sleep(2.0)
            return bool(self.ssh and self.instance_id and not self.lost)
        try:
            if self.on_interrupt:
                try:
                    self.on_interrupt(self)
                except Exception as e:  # noqa — замена не должна ломать ожидание
                    err(f"обработка прерывания: {e!r}")
            cap = float(self.bid_cap or 0) or float(self.args.max_price or 0) or \
                (float(self.cand.offer.get("dph_total") or 0) * 1.05 if self.cand else 0.0)
            price = self.cand.price if self.cand else 0.0
            sp = Spinner("Жду возвращения инстанса", est_total_sec=300)
            step_t = 0.0
            t0 = self.interrupted_at or time.time()
            while time.time() - t0 < self.args.resume_timeout:
                inst = self.refresh_instance()
                if inst is None:
                    sp.close()
                    return False
                if inst.get("actual_status") == "running":
                    sp.close(f"Инстанс снова работает (бид {fmt_money(price)}/ч)" if self.is_bid() else "Инстанс снова работает")
                    self.instance = inst
                    try:
                        self.connect_ssh(timeout=300)      # SSH мог смениться (порт/хост)
                    except FatalError:
                        return False
                    if self.ready and self.model:
                        try:
                            self.rewrite_jobs()            # очередь могла измениться, пока машина стояла
                            self.start_runner()
                        except FatalError as e:
                            warn(f"runner после возврата: {e}")
                            return False
                    self.bid_cap = None
                    return True
                if self.is_bid() and time.time() - step_t >= 5.0 and price < cap:
                    new = self.raise_bid(price, cap, inst)
                    if new > price:
                        price = new
                        note(f"Бид повышен до {fmt_money(price)}/ч (потолок {fmt_money(cap)}/ч"
                             f"{', min_bid машины ' + fmt_money(float(inst.get('min_bid'))) + '/ч' if inst.get('min_bid') else ''})")
                    step_t = time.time()
                if inst.get("intended_status") == "stopped" and inst.get("actual_status") in ("exited", "stopped", None) and not self.is_bid():
                    try:
                        self.client.start_instance(self.instance_id)     # on-demand инстанс остановлен — запускаем сами
                    except VastAPIError as e:
                        debug(f"start_instance: {e}")
                if self.on_interrupt_poll:
                    try:
                        self.on_interrupt_poll(self)
                    except Exception as e:  # noqa
                        debug(f"on_interrupt_poll: {e!r}")
                sp.tick(f"{inst.get('actual_status')}/{inst.get('intended_status')}"
                        + (f", бид {fmt_money(price)}/ч из {fmt_money(cap)}/ч" if self.is_bid() else ""))
                time.sleep(min(5.0, float(self.args.poll_interval)))
            sp.close()
            warn(f"Инстанс не вернулся за {fmt_time(self.args.resume_timeout)}.")
            return False
        finally:
            self.interrupted = False

    # ---------- установка софта в контейнере ----------
    BOOT_ALIVE_SH = "( [ -f bootstrap.pid ] && kill -0 $(cat bootstrap.pid) 2>/dev/null && echo BOOT_ALIVE )"
    BOOT_STEPS = {1: ("apt: ffmpeg/rsync/git", 0.08, 70), 2: ("git clone SeedVR2", 0.03, 15),
                  3: ("pip: зависимости", 0.19, 130), 4: ("веса модели", 0.62, 0), 5: ("проверка", 0.08, 15)}

    def bootstrap(self) -> None:
        assert self.ssh and self.model
        ssh = self.ssh
        root = self.root
        # уже установлено? (--resume / повторный запуск на том же инстансе)
        cp = ssh.run(f"cat {root}/bootstrap.status 2>/dev/null | tail -1", timeout=60)
        if cp.returncode == 0 and cp.stdout.strip().startswith("DONE"):
            cp2 = ssh.run(f"test -s {root}/models/{self.model['dit']} && test -d {root}/seedvr2 && echo yes", timeout=60)
            if "yes" in cp2.stdout:
                ok("SeedVR2 и веса уже установлены в контейнере — пропускаю установку.")
                return
        # установка уже идёт (например, прошлый запуск скрипта оборвался)? тогда не трогаем файлы
        cp = ssh.run(f"cd {root} 2>/dev/null && {self.BOOT_ALIVE_SH}; true", timeout=60)
        if "BOOT_ALIVE" in cp.stdout:
            ok("Установка уже идёт в контейнере — подключаюсь к её прогрессу")
        else:
            ssh.write_file(f"{root}/bootstrap.sh", BOOTSTRAP_SH.replace("@@ROOT@@", root).replace("@@OPT@@", self.opt_dir()), mode="755")
            ssh.write_file(f"{root}/bootstrap.env", render_bootstrap_env(self.model, self.image, self.prebuilt))
            # ВАЖНО: фоновый процесс запускаем в отдельной подоболочке со всеми перенаправленными
            # дескрипторами — иначе ssh ждёт завершения bootstrap.sh (десятки минут) и отваливается по таймауту
            cp = ssh.run(f"cd {root} && rm -f bootstrap.status bootstrap.pid && "
                         f"(setsid nohup bash bootstrap.sh > bootstrap.log 2>&1 < /dev/null &) ; sleep 2; "
                         f"{self.BOOT_ALIVE_SH}; true", timeout=90)
            if "BOOT_ALIVE" not in cp.stdout:
                tail = ssh.run(f"tail -n 20 {root}/bootstrap.log 2>/dev/null", timeout=60).stdout
                raise FatalError(f"Не удалось запустить bootstrap.sh: {cp.stderr[:300]}\n{tail}")
        weights_total = MODEL_FILES[self.model["dit"]]["size"] + MODEL_FILES[VAE_FILE]["size"]
        inet = float((self.cand.offer.get("inet_down") if self.cand else 0) or 100) * 1e6 / 8
        est_total = sum(s[2] for s in self.BOOT_STEPS.values()) + weights_total / min(inet, 120e6) * 1.15
        if self.prebuilt:
            est_total = 20.0
        bar = ProgressBar(100.0, "Проверка предустановленного ПО" if self.prebuilt else "Установка SeedVR2", unit="%", est_total_sec=est_total)
        step_started = time.time()
        cur_step = 0
        fails = 0
        dead_polls = 0
        while True:
            cp = ssh.run(f"cd {root} 2>/dev/null; cat bootstrap.status 2>/dev/null; echo ---; du -sb models 2>/dev/null | cut -f1; "
                         f"echo ---; {self.BOOT_ALIVE_SH}; true", timeout=60)
            if cp.returncode != 0:
                fails += 1
                if fails >= 3:
                    bar.render(force=True)
                    if not self.ensure_alive("установка"):
                        raise FatalError("контейнер потерян во время установки")
                    ssh = self.ssh
                    fails = 0
                time.sleep(self.args.poll_interval)
                continue
            fails = 0
            status_txt, _, rest = cp.stdout.partition("---")
            du, _, alive_txt = rest.partition("---")
            lines = [l for l in status_txt.strip().splitlines() if l.strip()]
            model_bytes = int(du.strip() or 0)
            done = False
            for l in lines:
                parts = l.split()
                if parts[0] == "STEP":
                    n = int(parts[1])
                    if n != cur_step:
                        cur_step, step_started = n, time.time()
                elif parts[0] == "DONE":
                    done = True
                elif parts[0] == "FAIL":
                    bar.render(force=True)
                    _raw_print("")
                    tail = ssh.run(f"tail -n 40 {root}/bootstrap.log", timeout=60).stdout
                    if self.prebuilt:
                        reasons = "; ".join(l.strip() for l in tail.splitlines() if l.startswith("prebuilt:") or "vast_upscale:" in l)
                        m = re.search(r"в контейнере Python (\d\.\d+)", tail)
                        raise PrebuiltBroken(f"Предсобранный образ {self.image} не прошёл проверку на инстансе — установка на "
                                             f"арендованной машине не выполняется. Причина: {reasons or tail[-400:]}",
                                             py_have=m.group(1) if m else None, reasons=reasons)
                    raise FatalError(f"Установка в контейнере не удалась на шаге «{' '.join(parts[1:-1])}». Хвост лога:\n{tail}")
            if not done and "BOOT_ALIVE" not in alive_txt:
                dead_polls += 1
                if dead_polls >= 3:
                    bar.render(force=True)
                    _raw_print("")
                    tail = ssh.run(f"tail -n 30 {root}/bootstrap.log", timeout=60).stdout
                    raise FatalError(f"Процесс установки в контейнере завершился без результата. Хвост лога:\n{tail}")
            else:
                dead_polls = 0
            if done:
                tail = ssh.run(f"tail -n 60 {root}/bootstrap.log", timeout=60).stdout
                fast = "BOOTSTRAP DONE (prebuilt)" in tail
                bar.close("Предустановленное ПО и веса на месте — установка не требовалась." if fast
                          else ("Недостающее доустановлено (готовое не повторялось)." if self.prebuilt
                                else "SeedVR2 установлен, веса скачаны и проверены."))
                for l in tail.splitlines():
                    if l.startswith("prebuilt:") or "vast_upscale:" in l:
                        warn(l.strip())
                m = re.search(r"в контейнере Python (\d\.\d+)", tail)
                if m and self.prebuilt:
                    warn(f"Образ собран под другую версию Python, чем в контейнере ({m.group(1)}): зависимости доустановил pip. "
                         f"Чтобы следующие инстансы стартовали без установки, пересоберите образ: --base-python {m.group(1)}")
                for l in tail.strip().splitlines()[-4:]:
                    if l and "BOOTSTRAP DONE" not in l and not l.startswith("prebuilt:"):
                        note(l)
                return
            pct = sum(self.BOOT_STEPS[i][1] for i in self.BOOT_STEPS if i < cur_step) * 100
            if cur_step in self.BOOT_STEPS:
                name, w, est = self.BOOT_STEPS[cur_step]
                if cur_step == 4:
                    frac = min(1.0, model_bytes / weights_total)
                    extra = f"[{name}: {fmt_bytes(model_bytes)}/{fmt_bytes(weights_total)}]"
                else:
                    frac = min(0.95, (time.time() - step_started) / max(est, 1))
                    extra = f"[{name}]"
                pct += w * frac * 100
                bar.set(pct, extra=extra)
            time.sleep(self.args.poll_interval)

    # ---------- загрузка входных файлов ----------
    def upload(self, jobs: Optional[List[Job]] = None, quiet: bool = False) -> None:
        assert self.ssh
        todo = jobs if jobs is not None else \
            [j for j in self.jobs if j.status == "pending" and j.assigned_to in (self.tag, f"pending-{self.tag}")]
        if not todo:
            return
        # что уже есть на сервере (при --resume)
        self.assign_paths(todo)
        cp = self.ssh.run(f"cd {self.root}/in 2>/dev/null && ls -l --block-size=1 | awk '{{print $5, $9}}'", timeout=60)
        present = {}
        for l in cp.stdout.splitlines():
            p = l.split()
            if len(p) == 2:
                present[p[1]] = int(p[0])
        need = [j for j in todo if present.get(Path(j.remote_in).name) != j.conv_bytes]
        for j in todo:
            if j not in need:
                j.status = "uploaded"
                (note if quiet else ok)(f"{j.label}: уже загружен в контейнер")
        if not need:
            return
        total = sum(j.conv_bytes for j in need)
        if not quiet:
            info(f"Загружаю {len(need)} файл(ов), {fmt_bytes(total)} (rsync, с докачкой при обрыве)")
        attempt = 0
        t0 = time.time()
        while True:
            attempt += 1
            cb = getattr(_tls, "board_cb", None)
            bar = ProgressBar(total, "⇧ " + (need[0].label if len(need) == 1 else "Загрузка"), bytes_mode=True,
                              on_render=(cb or (lambda l: None))) if quiet else None
            rc, e = self.ssh.rsync([str(j.converted) for j in need], f"{self.root}/in/", upload=True,
                                   desc="Загрузка", total_bytes=total, progress=bar)
            if bar:
                bar.close()
            if rc == 0:
                for j in need:
                    j.status = "uploaded"
                self.save_state()
                if quiet:
                    note(f"⇧ {', '.join(j.label for j in need)}: подгружен ({fmt_bytes(total)} за {fmt_time(time.time() - t0)})")
                else:
                    ok(f"Загрузка завершена ({fmt_bytes(total)})")
                return
            warn(f"rsync прервался (код {rc}): {e[-200:]}")
            if rc in RSYNC_FATAL_CODES:
                raise FatalError(f"rsync: неустранимая ошибка (код {rc}): {e[-300:]}")
            if not self.ensure_alive("загрузка"):
                raise FatalError("контейнер потерян во время загрузки")
            delay = min(60, 5 * attempt)
            note(f"Повтор загрузки через {delay} с (докачка с места обрыва)")
            time.sleep(delay)

    # ---------- очередь на инстансе ----------
    def slots(self) -> int:
        """Сколько чанков держать подгруженными заранее: не больше числа видеокарт инстанса (трафик за впустую
        загруженное при досрочном отключении инстанса не окупается)."""
        n = 0
        for src in (self.instance or {}, (self.cand.offer if self.cand else {}) or {}):
            try:
                n = int(src.get("num_gpus") or 0)
            except (TypeError, ValueError):
                n = 0
            if n > 0:
                break
        return max(1, min(n or 1, 8))

    def my_jobs(self) -> List[Job]:
        """Файлы, которые обрабатывает именно этот инстанс (и уже лежат на нём)."""
        return [j for j in self.jobs if j.assigned_to == self.tag and j.status in ("uploaded", "done")]

    def current_job_name(self) -> Optional[str]:
        st = self.last_status or {}
        return st.get("job") if not (st.get("finished") or st.get("idle")) else None

    def in_flight(self) -> List[Job]:
        """Чанки на инстансе, ещё не обработанные: текущий + подгруженные заранее."""
        st = self.last_status or {}
        done = set(st.get("done") or []) | set((st.get("failed") or {}).keys())
        return [j for j in self.jobs if j.assigned_to == self.tag and j.status == "uploaded" and j.name not in done]

    def preloaded(self) -> List[Job]:
        """Подгруженные заранее чанки (ещё не начатые). У недоступной машины runner не работает — и текущий тоже."""
        cur = self.current_job_name() if not (self.interrupted or self.lost) else None
        return [j for j in self.in_flight() if j.name != cur]

    def unstarted_jobs(self) -> List[Job]:
        return self.preloaded()

    def is_idle(self) -> bool:
        """Runner закончил очередь (или ждёт): свободен для новых файлов."""
        st = self.last_status or {}
        return bool(st.get("finished") or st.get("idle")) or (self.current_job_name() is None and not self.in_flight() and bool(st))

    def feed(self) -> int:
        """Подгрузка из очереди: пока подгруженных заранее меньше slots() и очередь не пуста — берём следующий чанк,
        загружаем, дописываем в jobs.json (runner подхватит, закончив текущий). Возвращает число подгруженных."""
        q = self.orch.queue if self.orch else None
        if q is None or not self.ssh or self.retired or self.interrupted or self.lost:
            return 0
        got = 0
        while len(self.preloaded()) < self.slots():
            j = q.pull(self)
            if j is None:
                break
            try:
                self.upload([j], quiet=True)
            except FatalError as e:
                q.push_front([j], why=f"{self.tag}: загрузка не удалась ({e})")
                raise
            got += 1
        if got:
            self.rewrite_jobs()
            self.start_runner(quiet=True)
            if self.orch:
                self.orch.after_feed(self)
        return got

    def ppt(self) -> float:
        """Цена в $/ч на TFLOP (если TFLOPS неизвестны — просто $/ч)."""
        if not self.cand:
            return 0.0
        v = price_per_tflop(self.cand.offer, self.cand.price)
        return v if v is not None else self.cand.price

    def value(self) -> float:
        """DLPerf/$ этой машины (чем больше, тем лучше); 0, если неизвестно."""
        if not self.cand:
            return 0.0
        return offer_value(self.cand.offer, self.cand.price) or (1.0 / self.cand.price if self.cand.price else 0.0)

    def rewrite_jobs(self) -> None:
        """Перезаписывает jobs.json на инстансе (runner перечитывает его перед каждым файлом)."""
        assert self.ssh and self.model
        max_long = self.args.max_long_side if self.args.max_long_side else 0
        cfg = {"dit": self.model["dit"], "batch": int(self.args.batch_size or self.model["batch"]),
               "user_batch": int(self.args.batch_size or 0),
               "extra": list(self.model["extra"]), "target": self.plan.target, "max_long": max_long,
               "seed": self.args.seed, "user_extra": shlex.split(self.args.extra_args or "")}
        jobs = []
        for j in self.my_jobs():
            d = {"name": j.name, "remote_in": j.remote_in, "remote_out": j.remote_out, "width": j.info.width,
                 "height": j.info.height, "frames": j.conv_frames or j.info.frames}
            if j.is_block:
                d["block"] = {"ctx": j.block_ctx, "frames": j.block_frames, "fps": fps_string(j.info.fps)}
            jobs.append(d)
        self.ssh.write_file(f"{self.root}/jobs.json", json.dumps({"config": cfg, "jobs": jobs}, ensure_ascii=False))

    def write_jobs_and_start_runner(self) -> None:
        assert self.ssh and self.model
        self.rewrite_jobs()
        self.ssh.write_file(f"{self.root}/runner.py", render_runner(self.root), mode="755")
        self.ssh.run(f"rm -f {self.root}/stop", timeout=30)
        self.start_runner()
        self.ready = True
        if self.orch:
            self.orch.on_ready(self)

    # живость runner.py проверяем по pid-файлу (pgrep -f ловил бы собственную командную строку ssh)
    RUNNER_ALIVE_SH = "( [ -f runner.pid ] && kill -0 $(cat runner.pid) 2>/dev/null && echo RUNNER_ALIVE )"

    def start_runner(self, quiet: bool = False) -> None:
        assert self.ssh
        cp = self.ssh.run(f"cd {self.root} && rm -f stop && export PY=$(cat python_path 2>/dev/null || echo python3); "
                          f"[ -f {self.opt_dir()}/env.sh ] && . {self.opt_dir()}/env.sh; "
                          f"if {self.RUNNER_ALIVE_SH} | grep -q RUNNER_ALIVE; then echo running; "
                          f"else rm -f runner.pid; "
                          f"(setsid nohup $PY -u runner.py > runner.log 2>&1 < /dev/null &) ; sleep 2; echo started; fi",
                          timeout=90)
        if "started" in cp.stdout:
            (note if quiet else ok)("Обработка запущена в контейнере (переживает обрывы SSH; лог: runner.log)")
        elif "running" in cp.stdout:
            if not quiet:
                ok("Обработка уже идёт в контейнере — подключаюсь к прогрессу")
        else:
            raise FatalError(f"Не удалось запустить runner.py: {cp.stderr[:300]} {cp.stdout[:300]}")

    def stop_runner(self) -> None:
        """Просит runner завершиться (он ждёт новых заданий, пока нет файла stop)."""
        if not self.ssh:
            return
        try:
            self.ssh.run(f"touch {self.root}/stop", timeout=30)
        except Exception as e:  # noqa
            debug(f"stop runner: {e!r}")

    def fetch_status(self) -> Optional[dict]:
        assert self.ssh
        cp = self.ssh.run(f"cd {self.root} 2>/dev/null; cat status.json 2>/dev/null; echo; {self.RUNNER_ALIVE_SH}; true", timeout=60)
        if cp.returncode != 0:
            return None
        txt = cp.stdout
        alive = "RUNNER_ALIVE" in txt
        body = txt.replace("RUNNER_ALIVE", "").strip()
        try:
            st = json.loads(body or "{}")
        except ValueError:
            st = {}
        st["_runner_alive"] = alive
        return st

    # ---------- мониторинг обработки + скачивание ----------
    def monitor_and_download(self) -> None:
        """Цикл машины пула: подгрузка чанков из очереди (не больше slots() заранее), опрос runner-а, скачивание
        готовых, перезапуск умершего runner-а, ожидание возврата при прерывании. Основной инстанс живёт, пока
        не обработана вся очередь (и координирует пул, даже потеряв свою машину); воркер — пока есть работа."""
        assert self.ssh
        orch = self.orch
        dl = Downloader(self)
        dl.start()
        spf_plan = (self.cand.model["spf_factor"] * GPU_PROFILES.get(self.cand.gpu, {"spf": 2.5})["spf"]) if self.cand else 2.5
        bar: Optional[ProgressBar] = None            # воркер: бар текущего чанка (на доску); основной: бар очереди
        cur_job = None
        seen_events = 0
        fails = 0
        reported_done: set = set()
        dead_polls = 0
        shown_mem = None
        idle_since: Optional[float] = None
        idle_max = 60.0
        released = False
        self.last_status = self.last_status or {}
        self.jobs_by_name = {j.name: j for j in self.jobs}

        def lost_primary() -> bool:
            """Основной потерял машину: если пул может продолжить — остаёмся координатором (цикл идёт дальше)."""
            if self.is_worker or not self.on_lost or not self.on_lost(self):
                return False
            self.lost_since = time.time()
            return True

        def unavailable(context: str) -> bool:
            """Машина недоступна. Воркер ждёт её возврата в своём потоке; основной — в фоне, не останавливая общий
            цикл. Возвращает True, если цикл должен просто продолжаться."""
            if self.is_worker:
                if self.ensure_alive(context):
                    return True
                raise FatalError(f"контейнер потерян ({context})")
            self.wait_back_async(context)
            return True

        try:
            while True:
                if self.external_lost and not self.lost:
                    self.external_lost = False
                    warn(f"{self.tag}: инстанса #{self.instance_id} больше нет в аккаунте (удалён в обход скрипта) — "
                         f"чанки возвращаются в очередь, нагрузка перераспределяется на оставшийся пул")
                    dl.stop()
                    if self.is_worker or not lost_primary():
                        raise FatalError("инстанс удалён в обход скрипта")
                waiting = self.waiting_back()
                if not self.is_worker and not waiting and self._waiter is not None:
                    res, self._waiter = self._wait_result, None            # фоновое ожидание закончилось
                    if not res:
                        if not lost_primary():
                            raise FatalError("контейнер потерян во время обработки")
                        dl.stop()
                live = bool(self.ssh) and not self.lost and not waiting and not self.interrupted
                if self.external_stopped and live:
                    self.external_stopped = False
                    warn(f"{self.tag}: инстанс #{self.instance_id} не работает по данным vast.ai — проверяю (перебили по цене / остановлен)")
                    unavailable("инстанс остановлен")
                    live = bool(self.ssh) and not self.lost and not self.waiting_back() and not self.interrupted
                # ---- подгрузка из очереди
                if live:
                    try:
                        self.feed()
                    except FatalError as e:
                        warn(f"подгрузка: {e}")
                        unavailable("подгрузка")
                        live = False
                # ---- опрос
                st = self.fetch_status() if live else None
                if st is None and live:
                    fails += 1
                    if fails >= 3:
                        if bar and self.is_worker:
                            bar.close()
                            bar = None
                        unavailable("обработка")
                        fails = 0
                    if self.is_worker:
                        time.sleep(self.args.poll_interval)
                        continue
                fails = 0 if st is not None else fails
                if st is not None:
                    self.last_status = st
                    self.jobs_by_name = {j.name: j for j in self.jobs}
                    for ev in (st.get("events") or [])[seen_events:]:
                        warn(ev)
                    seen_events = len(st.get("events") or [])
                    # завершённые чанки → в очередь скачивания
                    for name in st.get("done") or []:
                        j = self.jobs_by_name.get(name)
                        if not j or name in reported_done:
                            continue
                        reported_done.add(name)
                        if j.assigned_to != self.tag or j.status in ("done", "downloaded", "finished"):
                            continue                  # чанк уже отдан другой машине или обработан ранее
                        if bar and self.is_worker and cur_job is j:
                            bar.close()
                            bar = None
                        j.status = "done"
                        fps = (st.get("fps") or {}).get(name)
                        frames = j.conv_frames or j.info.frames
                        if fps:
                            j.proc_seconds = frames / fps
                            self.calib_spf = (1.0 / fps) / max(self.cand.model["spf_factor"], 1e-6) if self.cand else None
                            ok(f"{j.label}: апскейл готов, {fps:.2f} кадр/с ({1 / fps:.2f} с/кадр; план был {spf_plan:.2f} с/кадр)")
                        else:
                            ok(f"{j.label}: апскейл готов")
                        if orch:
                            orch.note_done(self, j)
                        self.save_state()
                        dl.enqueue(j)
                    for name, msg in (st.get("failed") or {}).items():
                        j = self.jobs_by_name.get(name)
                        if j and j.assigned_to == self.tag and j.status == "uploaded":
                            j.status, j.error = "failed", msg
                            err(f"{j.label}: ошибка обработки в контейнере: {msg[:400]}")
                            note(f"Полный лог: ssh ... 'cat {self.root}/logs/{name}.log'")
                            if bar and self.is_worker and cur_job is j:
                                bar.close()
                                bar = None
                    # текущий чанк
                    name = st.get("job")
                    j = self.jobs_by_name.get(name) if (name and not st.get("idle")) else None
                    if j and j.status == "uploaded" and j.assigned_to == self.tag:
                        frames = j.conv_frames or j.info.frames
                        if self.is_worker and (cur_job is not j or bar is None):
                            cur_job = j
                            spf_now = self.calib_spf * (self.cand.model["spf_factor"] if self.cand else 1.0) if self.calib_spf else spf_plan
                            bar = ProgressBar(frames, f"{j.label}", unit=" кадр",
                                              est_total_sec=frames * spf_now + (self.model or {}).get("load_sec", 60))
                        cur_job = j
                        mem = st.get("mem") or {}
                        if mem and (cur_job, st.get("attempt")) != shown_mem:
                            shown_mem = (cur_job, st.get("attempt"))
                            lim = f", лимит контейнера {mem['ram_limit_gb']} ГБ" if mem.get("ram_limit_gb") else ""
                            note(f"память: VRAM свободно {mem.get('vram_free_gb')}/{mem.get('vram_total_gb')} ГБ → batch {st.get('batch_size')}"
                                 f"{' + tiled VAE' if mem.get('tiled') else ''} (выход {mem.get('out_mp')} МП); "
                                 f"RAM доступно {mem.get('ram_avail_gb')} ГБ{lim} → чанк {st.get('chunk_size') or 'весь файл'} "
                                 f"(~{mem.get('chunk_gb')} ГБ, {mem.get('per_frame_mb')} МБ/кадр)")
                        if self.is_worker and bar:
                            pct = float(st.get("pct") or 0.0)
                            phase = st.get("phase") or ""
                            b = st.get("batch") or [0, 1]
                            extra = f"[{phase} {b[0]}/{b[1]}" + (f", попытка {st.get('attempt')}" if (st.get("attempt") or 1) > 1 else "") + "]"
                            if dl.current_line:
                                extra += " ⇩" + dl.current_line
                            bar.set(pct * frames, extra=extra)
                    elif self.is_worker and bar and cur_job and cur_job.status != "uploaded":
                        bar.close()
                        bar = None
                    # runner умер, не закончив? (два опроса подряд — чтобы не среагировать на момент старта)
                    if not st.get("finished") and not st.get("_runner_alive"):
                        dead_polls += 1
                    else:
                        dead_polls = 0
                    if dead_polls >= 2:
                        dead_polls = 0
                        warn("Процесс обработки в контейнере не найден — перезапускаю (готовые файлы не пересчитываются).")
                        alive_now = self.refresh_instance() if self.instance_id else None
                        if not alive_now or alive_now.get("actual_status") != "running":
                            unavailable("обработка")
                            time.sleep(self.args.poll_interval)
                            continue
                        try:
                            self.rewrite_jobs()
                            self.start_runner()
                        except FatalError as e:
                            warn(f"перезапуск runner-а: {e}")
                        time.sleep(self.args.poll_interval)
                        continue
                # ---- основной: прогресс очереди + управление пулом
                if not self.is_worker and orch:
                    try:
                        orch.on_poll(self.last_status or {})
                    except Exception as e:  # noqa — скан рынка не должен ронять обработку
                        debug(f"on_poll: {e!r}")
                    bar = orch.render_queue(bar, dl.current_line)
                # ---- условия выхода
                in_flight = self.in_flight()
                if self.is_worker:
                    if not in_flight:
                        if self.retired or (orch and orch.all_done()):
                            break
                        if orch and orch.queue.empty():
                            idle_since = idle_since or time.time()
                            if time.time() - idle_since > idle_max:
                                note(f"работы нет {fmt_time(idle_max)} — освобождаю инстанс")
                                break
                        else:
                            idle_since = None
                    else:
                        idle_since = None
                else:
                    if orch and orch.all_done():
                        break
                    if self.retired and not in_flight and not released and self.instance_id and not self.lost:
                        # основной выбыл как худший и свободен: отпускаем машину, остаёмся координатором
                        released = True
                        dl.finish_and_wait()
                        self.stop_runner()
                        ok("Основной инстанс выбыл из пула как худший и свободен — удаляю его, очередь доделывают остальные")
                        if not self.args.keep_instance:
                            self.destroy()
                        self.ssh = None
                        self.lost = True
                        self.lost_since = time.time()
                    if self.lost and not orch:
                        break
                    if self.lost and orch and not orch.active_workers() and orch.queue.empty():
                        break
                    if self.lost and orch and not orch.active_workers() and orch.provisioning <= 0 \
                            and time.time() - (self.lost_since or time.time()) > float(self.args.resume_timeout):
                        raise FatalError(f"машины потеряны, замены нет уже {fmt_time(self.args.resume_timeout)}: "
                                         f"в очереди {len(orch.queue)} чанк(ов)")
                    if self.lost:
                        time.sleep(self.args.poll_interval)
                        continue
                time.sleep(self.args.poll_interval)
        finally:
            if bar and self.is_worker:
                bar.close()
        if bar and not self.is_worker and orch:
            orch.close_queue_bar(bar)
        # ждём скачивание всех результатов
        dl.finish_and_wait()
        if self.ssh and not self.lost:
            self.stop_runner()
        self.save_state()

class Downloader(threading.Thread):
    """Фоновая докачка результатов: rsync --partial --inplace с бесконечными повторами,
    пока контейнер жив. Проверка размера и sha256 по маркеру .done."""

    def __init__(self, session: Session):
        super().__init__(daemon=True)
        self.s = session
        self.queue: List[Job] = []
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.closing = False
        self.stopped = False
        self.current_line = ""
        self.errors: Dict[str, str] = {}
        self._prefix = getattr(_tls, "prefix", "")   # поток-воркер: сохраняем префикс строк

    def enqueue(self, job: Job):
        with self.cv:
            self.queue.append(job)
            self.cv.notify()

    def stop(self):
        with self.cv:
            self.stopped = True
            self.cv.notify_all()

    def finish_and_wait(self):
        with self.cv:
            self.closing = True
            self.cv.notify_all()
        sp = None
        while self.is_alive():
            self.join(0.5)
            if self.current_line:
                sp = sp or Spinner("Скачивание результатов")
                sp.tick(self.current_line)
        if sp:
            sp.close()

    def run(self):
        if self._prefix:
            _tls.prefix = self._prefix
        while True:
            with self.cv:
                while not self.queue and not self.closing and not self.stopped:
                    self.cv.wait(1.0)
                if self.stopped or (not self.queue and self.closing):
                    return
                job = self.queue.pop(0)
            try:
                self.download_one(job)
            except Exception as e:  # noqa
                if job.status == "done":                 # блок ещё наш (не ушёл в очередь пула после потери машины)
                    job.status, job.error = "failed", f"скачивание: {e}"
                    err(f"{job.label}: не удалось скачать результат: {e}")
                else:
                    debug(f"{job.label}: скачивание прервано ({e}); блок переназначен")

    def download_one(self, job: Job):
        s = self.s
        assert s.ssh
        # чанк — сразу в кэш чанков (переживает перезапуски; удаляется после сборки видео), целый файл — в папку запуска
        if job.is_block and job.cache_dir:
            job.cache_dir.mkdir(parents=True, exist_ok=True)
            local = job.cache_dir / f"{job.name}.mp4"
        else:
            local = s.workdir / f"{job.name}.up.mp4"
        # маркер .done с размером и sha256
        meta = None
        for _ in range(5):
            cp = s.ssh.run(f"cat {shlex.quote(job.remote_out)}.done", timeout=60)
            if cp.returncode == 0:
                try:
                    meta = json.loads(cp.stdout)
                    break
                except ValueError:
                    pass
            time.sleep(3)
        if not meta:
            raise FatalError("маркер .done не прочитан")
        size = int(meta["size"])
        attempt = 0
        bar = ProgressBar(size, f"⇩ {job.label}", bytes_mode=True, on_render=lambda l: setattr(self, "current_line", l))
        t_giveup = time.time() + (s.args.download_timeout if s.args.download_timeout else 10 ** 9)
        while True:
            attempt += 1
            if local.exists() and local.stat().st_size == size and sha256_of(local) == meta["sha256"]:
                bar.close(f"{job.label}: результат скачан и проверен ({fmt_bytes(size)})")
                break
            self.current_line = f"⇩ {job.label}: попытка {attempt}"
            rc, e = s.ssh.rsync([job.remote_out], str(local), upload=False, desc="Скачивание", total_bytes=size,
                                progress=bar)
            if rc == 0 and local.exists() and local.stat().st_size == size:
                if sha256_of(local) == meta["sha256"]:
                    bar.close(f"{job.label}: результат скачан и проверен ({fmt_bytes(size)})")
                    break
                warn(f"{job.label}: sha256 не совпал после скачивания — скачиваю заново")
                try:
                    local.unlink()
                except OSError:
                    pass
            else:
                have = local.stat().st_size if local.exists() else 0
                warn(f"{job.label}: обрыв скачивания (код {rc}, есть {fmt_bytes(have)} из {fmt_bytes(size)}): {e[-160:]}")
                if rc in RSYNC_FATAL_CODES:
                    raise FatalError(f"rsync: неустранимая ошибка (код {rc}): {e[-300:]}")
            if self.stopped:
                raise FatalError("остановлено")
            if time.time() > t_giveup:
                raise FatalError("превышен --download-timeout")
            # жив ли контейнер? если нет — ждём возобновления/выходим
            if not s.ensure_alive("скачивание"):
                raise FatalError("контейнер завершил работу — докачка невозможна")
            delay = min(90, 5 * attempt)
            self.current_line = f"⇩ {job.label}: повтор через {delay} с"
            time.sleep(delay)
        self.current_line = ""
        job.downloaded = local
        job.status = "downloaded"
        chunk_cache_store(job, size, meta["sha256"])
        s.save_state()
        cb = s.on_downloaded or (s.parent.on_downloaded if s.parent else None)
        if cb:
            try:
                cb(job)
            except Exception as e:  # noqa — сборка не должна ронять скачивание
                err(f"{job.label}: сборка после скачивания: {e!r}")


# ----------------------------------------------------------------------------
# Оркестратор параллельных воркеров: пока основной инстанс обрабатывает очередь,
# периодически сканирует рынок; найдя оффер в N раз дешевле медианы по $/TFLOP,
# заказывает дополнительный инстанс и отдаёт ему ещё не начатые файлы из очереди.
# ----------------------------------------------------------------------------
LABEL_PREFIX = "vast_upscale"


def instances_in_use() -> set:
    """Инстансы, занятые живыми фоновыми запусками (по их state.json из реестра демонов)."""
    used: set = set()
    for wd, ent in _registry_load().items():
        if not _pid_alive(int(ent.get("pid") or 0)) or ent.get("pid") == os.getpid():
            continue
        try:
            st = json.loads((Path(wd) / "state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if st.get("instance_id"):
            used.add(int(st["instance_id"]))
        for w in st.get("workers") or []:
            if w.get("instance_id"):
                used.add(int(w["instance_id"]))
    return used


def find_our_instances(client: VastClient, images: List[str], repo: str = "", exclude: Optional[set] = None) -> List[dict]:
    """Работающие инстансы этого скрипта с прошлых запусков: метка «vast_upscale …» и наш образ (базовый или любой тег
    репозитория предсобранных); занятые другими живыми запусками и уже используемые исключаются."""
    try:
        insts = client.show_instances() or []
    except Exception as e:  # noqa
        debug(f"show_instances: {e!r}")
        return []
    busy = instances_in_use() | set(exclude or ())
    out = []
    for i in insts:
        try:
            iid = int(i.get("id"))
        except (TypeError, ValueError):
            continue
        label = str(i.get("label") or "")
        img = str(i.get("image_uuid") or i.get("image") or "")
        if not label.startswith(LABEL_PREFIX) or iid in busy:
            continue
        if i.get("actual_status") != "running" or (i.get("intended_status") or "running") != "running":
            continue
        if not (img in images or (repo and img.startswith(repo + ":"))):
            continue
        out.append(i)
    return out


class WorkQueue:
    """Очередь чанков на обработку (живёт на этой машине). Машины пула берут из неё по одному чанку (`pull`), держа
    подгруженными заранее не больше числа своих видеокарт; чанки потерянной машины возвращаются в голову очереди."""

    def __init__(self, jobs: List[Job]):
        self.lock = threading.RLock()
        self.order = {j.name: i for i, j in enumerate(jobs)}
        self.items: List[Job] = [j for j in jobs if j.status == "pending"]
        self.t0 = time.time()
        self.done_log: List[Tuple[float, int]] = []      # (когда, кадров) — для оценки скорости пула

    def empty(self) -> bool:
        with self.lock:
            return not self.items

    def __len__(self) -> int:
        with self.lock:
            return len(self.items)

    def frames(self) -> int:
        with self.lock:
            return sum(j.conv_frames or j.info.frames for j in self.items)

    def pull(self, s: "Session") -> Optional[Job]:
        with self.lock:
            if not self.items:
                return None
            j = self.items.pop(0)
            j.assigned_to, j.status, j.error, j.in_transfer = s.tag, "pending", "", False
            if j not in s.jobs:
                s.jobs.append(j)
            return j

    def push_front(self, jobs: List[Job], why: str = "") -> None:
        """Чанки возвращаются в очередь (в исходном порядке, вперёд): их подхватит первая свободная машина."""
        with self.lock:
            back = [j for j in jobs if j.status not in ("downloaded", "finished") and j not in self.items]
            for j in back:
                j.assigned_to, j.status, j.error, j.in_transfer = "pool", "pending", "", False
            self.items = sorted(back + self.items, key=lambda j: self.order.get(j.name, 1 << 30))
        if back:
            info(f"В очередь возвращено {len(back)} чанк(ов)" + (f": {why}" if why else "") + " — " + ", ".join(j.label for j in back[:6])
                 + (" …" if len(back) > 6 else ""))

    def note_done(self, frames: int) -> None:
        with self.lock:
            self.done_log.append((time.time(), frames))
            self.done_log = self.done_log[-200:]

    def rate(self) -> float:
        """Скорость пула, кадров/с, по завершённым за последние ~10 минут чанкам (0, если данных нет)."""
        with self.lock:
            now = time.time()
            recent = [(t, f) for t, f in self.done_log if now - t < 600]
            if len(recent) < 1:
                return 0.0
            span = now - min(t for t, f in recent) if len(recent) > 1 else now - self.t0
            span = max(span, 1.0)
            return sum(f for t, f in recent) / span


class Orchestrator:
    """Пул машин: очередь чанков, подгрузка не больше числа видеокарт, ребаланс при подключении/отключении,
    скан рынка и регулятор числа машин, замена при прерывании, прогресс очереди с общим эстимейтом."""

    def __init__(self, primary: Session, client: VastClient, key: Path, args, plan: Plan, workdir: Path):
        self.primary, self.client, self.key, self.args, self.plan, self.workdir = primary, client, key, args, plan, workdir
        self.workers: List[Session] = []
        self.threads: Dict[str, threading.Thread] = {}
        self.board: Dict[str, str] = {}
        self.lock = threading.RLock()
        self.t_last_scan = 0.0
        self.tried_machines: set = set()
        self.image_fn: Optional[Callable[[dict], Tuple[str, Optional[str], bool]]] = None
        self.enabled = not args.no_parallel and args.pool_scan > 0 and args.max_workers > 0 and len(plan.jobs) > 1
        self.primary_chosen = threading.Event()
        self.primary_chosen.set()
        self.image_dit: Optional[str] = None       # модель предсобранного образа: пул набирается только из подходящих ей офферов
        self.rebuild_fn: Optional[Callable[[str], bool]] = None   # пересборка образа под другую версию Python контейнера
        self.image_broken: str = ""                # образ не прошёл проверку на инстансе и починить нечем — воркеры не заказываем
        self.pool_size = max(1, int(getattr(args, "pool_size", 1) or 1))   # регулятор: сколько активных машин держать
        self.scan_requested = False                # клавиша r: внеплановый скан рынка
        self.all_jobs: List[Job] = list(plan.jobs)
        self.queue = WorkQueue(self.all_jobs)
        self.total_frames = sum(j.conv_frames or j.info.frames for j in self.all_jobs)
        self.queue_bar: Optional[ProgressBar] = None
        self._last_queue_line = ""
        self._last_queue_print = 0.0
        primary.orch = self
        primary.on_poll = self.on_poll
        primary.board_line = self.board_line
        primary.state_extra_fn = self.state_extra
        self.attach_hooks(primary)

    def attach_hooks(self, s: Session) -> None:
        """Хуки машины пула: очередь, прерывание (перебили по цене / хост забрал GPU), потеря."""
        s.orch = self
        s.on_interrupt = self.on_interrupt
        s.on_lost = self.on_lost

    # ---- регулятор числа машин ----
    def set_pool_size(self, n: int, who: str = "") -> int:
        cap = 1 + int(self.args.max_workers)
        if int(n) > cap:
            warn(f"Пул: больше {cap} машин не даёт --max-workers {self.args.max_workers} — поднимите его при запуске")
        n = max(1, min(int(n), cap))
        if n != self.pool_size:
            self.pool_size = n
            info(f"Пул: требуется активных машин — {n}{(' (' + who + ')') if who else ''}; {self.pool_status()}")
        else:
            info(f"Пул: требуется активных машин — {n} (без изменений); {self.pool_status()}")
        return self.pool_size

    def request_scan(self, who: str = "") -> None:
        """Клавиша r: внеплановая проверка листинга и пригодности лучшего оффера к захвату."""
        self.scan_requested = True
        info(f"Внеплановая проверка рынка{(' (' + who + ')') if who else ''} — на ближайшем опросе")

    def pool_status(self) -> str:
        ready = self.ready_members()
        waiting = [m for m in self.members() if m.interrupted]
        return (f"нужно {self.pool_size} · активно {len(ready)} · разворачивается {self.provisioning}"
                + (f" · ждут возврата {len(waiting)}" if waiting else ""))

    # ---- служебное ----
    def state_extra(self) -> dict:
        return {"workers": [{"tag": w.tag, "instance_id": w.instance_id, "gpu": w.cand.gpu if w.cand else None,
                             "jobs": [j.name for j in w.jobs]} for w in self.workers if w.instance_id],
                "queue": [j.name for j in self.queue.items]}

    def board_line(self) -> str:
        with self.lock:
            return " ‖ ".join(f"{k}: {v}" for k, v in self.board.items() if v)

    def active_workers(self) -> List[Session]:
        return [w for w in self.workers if self.threads.get(w.tag) and self.threads[w.tag].is_alive()]

    @property
    def provisioning(self) -> int:
        """Сколько машин сейчас разворачивается: живые потоки воркеров/кандидатов, у которых runner ещё не запущен."""
        return sum(1 for w in self.active_workers() if not w.ready and not w.lost)

    def members(self) -> List[Session]:
        """Машины пула с runner-ом (в т.ч. недоступные/прерванные)."""
        ms = [self.primary] if self.primary.ssh and self.primary.instance_id and self.primary.ready and not self.primary.lost else []
        ms += [w for w in self.active_workers() if w.ssh and w.instance_id and w.cand and w.ready and not w.lost]
        return ms

    def ready_members(self) -> List[Session]:
        """Машины, готовые брать чанки из очереди прямо сейчас: SSH есть, runner запущен, не выбывают, не прерваны."""
        return [m for m in self.members() if not m.retired and not m.interrupted]

    # ---- очередь: состояние и прогресс ----
    def done_frames(self) -> float:
        done = 0.0
        for j in self.all_jobs:
            fr = j.conv_frames or j.info.frames
            if j.status in ("done", "downloaded", "finished"):
                done += fr
        for m in self.members():
            st = m.last_status or {}
            cur = m.current_job_name()
            j = next((x for x in m.jobs if x.name == cur), None) if cur else None
            if j and j.status == "uploaded" and j.assigned_to == m.tag:
                done += float(st.get("pct") or 0.0) * (j.conv_frames or j.info.frames)
        return min(done, float(self.total_frames))

    def counts(self) -> Dict[str, int]:
        c = {"total": len(self.all_jobs), "done": 0, "failed": 0, "processing": 0, "preloaded": 0, "queue": len(self.queue)}
        for j in self.all_jobs:
            if j.status in ("done", "downloaded", "finished"):
                c["done"] += 1
            elif j.status == "failed":
                c["failed"] += 1
        for m in self.members():
            infl = m.in_flight()
            pre = m.preloaded()
            c["preloaded"] += len(pre)
            c["processing"] += len(infl) - len(pre)
        return c

    def eta(self) -> Optional[float]:
        rate = self.queue.rate()
        left = float(self.total_frames) - self.done_frames()
        if rate > 0:
            return left / rate
        # нет данных: по профилям готовых машин (последовательно на каждой) либо по плану основной
        spf = [((m.calib_spf or GPU_PROFILES.get(m.cand.gpu, {"spf": 2.5})["spf"]) * m.cand.model["spf_factor"]) for m in self.ready_members() if m.cand]
        if spf:
            return left / sum(1.0 / s for s in spf)
        return None

    def queue_line(self) -> str:
        c = self.counts()
        eta = self.eta()
        return (f"очередь: готово {c['done']}/{c['total']} · в работе {c['processing']} · подгружено {c['preloaded']} · ждут {c['queue']}"
                + (f" · ошибок {c['failed']}" if c["failed"] else "") + f" · осталось ~{fmt_time(eta)}")

    def machines_line(self) -> str:
        parts = []
        for m in self.members():
            st = m.last_status or {}
            cur = m.current_job_name()
            j = next((x for x in m.jobs if x.name == cur), None) if cur else None
            if m.interrupted:
                parts.append(f"{m.tag}: ждёт возврата")
            elif j:
                parts.append(f"{m.tag}: {j.label.split(' · ')[-1]} {float(st.get('pct') or 0) * 100:.0f}%")
            else:
                parts.append(f"{m.tag}: свободна")
        return " · ".join(parts)

    def render_queue(self, bar: Optional[ProgressBar], dl_line: str = "") -> Optional[ProgressBar]:
        """Бар очереди у основного: кадры всех чанков, общий эстимейт по фактической скорости пула."""
        if bar is None:
            bar = ProgressBar(float(self.total_frames or 1), "Очередь", unit=" кадр", est_total_sec=None)
        c = self.counts()
        extra = (f"[чанки {c['done']}/{c['total']} · в работе {c['processing']} · подгружено {c['preloaded']} · ждут {c['queue']}"
                 + (f" · ошибок {c['failed']}" if c["failed"] else "") + "]")
        ml = self.machines_line()
        if ml:
            extra += " " + ml
        if dl_line:
            extra += " ⇩" + dl_line
        bl = self.board_line()
        if bl:
            extra += " ‖ " + bl
        eta = self.eta()
        bar.est_total_sec = (time.time() - bar.t0) + eta if eta is not None else None
        bar.set(self.done_frames(), extra=extra)
        return bar

    def close_queue_bar(self, bar: ProgressBar) -> None:
        bar.close()

    def all_done(self) -> bool:
        """Вся очередь обработана: нет чанков в ожидании, в работе или на скачивании."""
        if not self.queue.empty():
            return False
        return all(j.status in ("downloaded", "finished", "failed", "skipped") for j in self.all_jobs)

    def note_done(self, s: Session, j: Job) -> None:
        self.queue.note_done(j.conv_frames or j.info.frames)

    def after_feed(self, s: Session) -> None:
        with self.lock:
            self.board["пул"] = self.pool_status()

    # ---- ребаланс ----
    def on_ready(self, s: Session) -> None:
        """Машина подключилась (runner запущен): ребаланс — если очередь пуста, а у других есть подгруженные,
        но не начатые чанки, они возвращаются в очередь, и новая машина берёт их сама."""
        self.rebalance(f"подключилась {s.tag}")

    def rebalance(self, why: str = "") -> None:
        """Ребаланс при подключении/отключении машины. Чанки в процессинге не трогаются; подгруженные заранее
        (ещё не начатые) уходят обратно в очередь только тогда, когда очередь пуста и есть свободная машина —
        иначе они и так следующие в работу на своей машине."""
        with self.lock:
            self.board["пул"] = self.pool_status()
        if not self.queue.empty():
            return
        idle = [m for m in self.ready_members() if not m.in_flight()]
        if not idle:
            return
        donors = sorted((m for m in self.members() if m.preloaded() and (m.retired or m.interrupted or m is not idle[0])),
                        key=lambda m: (not (m.retired or m.interrupted), m.value()))
        for d in donors:
            if not idle:
                break
            pre = d.preloaded()
            if not pre:
                continue
            got = self.reclaim(d, pre[-len(idle):] if len(pre) > len(idle) else pre)
            if got:
                self.queue.push_front(got, why=f"ребаланс ({why}): у {d.tag} они ещё не начаты")
                idle = idle[len(got):]

    def reclaim(self, src: Session, jobs: List[Job]) -> List[Job]:
        """Убирает подгруженные, но не начатые чанки из очереди машины src (с проверкой гонки с runner-ом)."""
        with self.lock:
            jobs = [j for j in jobs if j.assigned_to == src.tag and j.status == "uploaded"]
            if not jobs:
                return []
            if src.interrupted or src.lost or not src.ssh:
                return jobs                         # runner не работает — гонки нет
            for j in jobs:
                j.assigned_to = "pool"
            try:
                src.rewrite_jobs()
                st = src.fetch_status() or src.last_status or {}
            except Exception as e:  # noqa
                warn(f"Не удалось перераспределить очередь {src.tag}: {e}")
                for j in jobs:
                    j.assigned_to = src.tag
                return []
            src.last_status = st or src.last_status
            cur = st.get("job") if not (st.get("finished") or st.get("idle")) else None
            racing = [j for j in jobs if j.name == cur]
            if racing:
                for j in racing:
                    j.assigned_to = src.tag
                    jobs.remove(j)
                try:
                    src.rewrite_jobs()
                except Exception:  # noqa
                    pass
            return jobs

    def wait_workers(self) -> None:
        """Ждёт завершения всех потоков-воркеров, показывая их прогресс."""
        sp = None
        while self.active_workers():
            sp = sp or Spinner("Жду параллельные воркеры")
            sp.tick(self.queue_line() + " ‖ " + (self.board_line() or "…"))
            time.sleep(1.0)
        if sp:
            sp.close()

    def primary_ppt(self) -> Optional[float]:
        c = self.primary.cand
        if not c:
            return None
        return price_per_tflop(c.offer, c.price)

    def used_machines(self) -> set:
        s = set(self.tried_machines)
        if self.primary.cand:
            s.add(self.primary.cand.offer.get("machine_id"))
        for w in self.workers:
            if w.cand:
                s.add(w.cand.offer.get("machine_id"))
        return s

    # ---- вызывается из цикла мониторинга основного инстанса ----
    def on_poll(self, st: dict) -> None:
        if not self.enabled:
            return
        self.primary.last_status = st or self.primary.last_status
        if time.time() - self.t_last_scan < self.args.pool_scan and not self.scan_requested:
            try:
                self.evaluate_pool()
            except Exception as e:  # noqa
                debug(f"evaluate_pool: {e!r}")
            return
        forced, self.scan_requested = self.scan_requested, False
        self.t_last_scan = time.time()
        try:
            self.check_instances()
        except Exception as e:  # noqa
            debug(f"check_instances: {e!r}")
        try:
            self.scan_market(st, verbose=forced)
        except Exception as e:  # noqa
            debug(f"scan_market: {e!r}")

    def check_instances(self) -> None:
        """Раз в --pool-scan: список инстансов аккаунта сверяется с пулом. Удалённый в обход скрипта инстанс помечается
        потерянным сразу (его чанки — в очередь, замена — по регулятору), остановленный — как прерванный."""
        members = [m for m in [self.primary] + self.active_workers() if m.instance_id and not m.lost]
        if not members:
            return
        insts = {}
        lst = self.client.show_instances()
        if not isinstance(lst, list):
            return
        for i in lst:
            try:
                insts[int(i.get("id"))] = i
            except (TypeError, ValueError):
                pass
        for m in members:
            inst = insts.get(int(m.instance_id))
            if inst is None:
                m.missing_polls = getattr(m, "missing_polls", 0) + 1
                if m.missing_polls >= 2:                     # два скана подряд — не сбой API, а удаление
                    m.external_lost = True
                    m.instance = None
            else:
                m.missing_polls = 0
                m.instance = inst
                if inst.get("actual_status") != "running" and not m.we_stopped and not m.interrupted and not m.waiting_back():
                    m.external_stopped = True

    def evaluate_pool(self, market: Optional[List[Bargain]] = None) -> None:
        """Выбывание худших: машина помечается retired (новых чанков не берёт, доделывает начатое и подгруженное,
        освободится — удалим), если есть готовая машина пула лучше её по DLPerf/$ не менее чем в --pool-improve раз,
        и после её ухода в пуле останется не меньше --pool-min готовых и не меньше --pool-size активных."""
        ms = [m for m in self.members() if m.cand]
        with self.lock:
            self.board["пул"] = self.pool_status()
        if len(ms) < 2:
            return
        f = float(self.args.pool_improve)
        for m in sorted(ms, key=lambda x: x.value()):
            if m.retired or m.value() <= 0:
                continue
            ready_ok = [x for x in ms if x is not m and not x.retired and not x.interrupted and x.value() > 0]
            if len(ready_ok) < max(self.args.pool_min, self.pool_size):
                continue                        # регулятор требует столько машин — худшую не отпускаем без лучшей замены
            best = max(ready_ok, key=lambda x: x.value())
            if best.value() >= m.value() * f:
                m.expensive = m.retired = True
                warn(f"{m.tag}: {m.cand.gpu} за {fmt_money(m.cand.price)}/ч — худшая в пуле по DLPerf/$ "
                     f"({m.value():.0f} против {best.value():.0f} у {best.tag}, порог ×{f:g}): новых чанков не берёт, освободится — удалю")
                self.rebalance(f"выбывает {m.tag}")
        # регулятор: активных (готовых, не прерванных) машин больше, чем требуется — худшие выбывают (не ниже --pool-min)
        active = sorted((m for m in ms if not m.retired and not m.interrupted), key=lambda m: m.value())
        while len(active) > self.pool_size and len(active) > self.args.pool_min:
            m = active.pop(0)
            m.retired = True
            warn(f"{m.tag}: активных машин {len(active) + 1} > требуемых {self.pool_size} — худшая по DLPerf/$ выбывает "
                 f"(новых чанков не берёт, освободится — удалю)")
            self.rebalance(f"выбывает {m.tag}")

    def scan_market(self, st: dict, verbose: bool = False) -> None:
        """Раз в --pool-scan секунд (или по клавише r): листинг офферов → лучший по DLPerf/$ берётся в пул, если он
        лучше худшей машины пула не менее чем в --pool-improve раз, или готовых меньше --pool-min, или активных
        меньше --pool-size, — и в очереди есть работа для него; худшие помечаются на выбывание (evaluate_pool)."""
        market = list_market(self.client, self.args, self.plan, self.used_machines(), self.image_dit)
        for w in self.active_workers():
            if w.instance_id and not w.booted and w.t_ordered and time.time() - w.t_ordered > self.args.boot_deadline:
                warn(f"{w.tag}: не поднялся за {fmt_time(self.args.boot_deadline)} — удаляю")
                try:
                    w.destroy(quiet=True)
                except Exception:  # noqa
                    pass
        self.evaluate_pool(market)
        ready = self.ready_members()
        best = market[0] if market else None
        worst = min((m for m in ready if m.value() > 0), key=lambda m: m.value(), default=None)
        with self.lock:
            self.board["рынок"] = (f"лучший {best.offer.get('gpu_name')} {fmt_money(best.price)}/ч DLPerf/$ {best.value:.0f}"
                                  + (f" (пул: худший {worst.value():.0f})" if worst else "")) if best else "нет офферов"
            self.board["пул"] = self.pool_status()
        deficit = self.pool_size - len(ready) - self.provisioning     # регулятор: сколько машин не хватает
        verdict = ""
        if not best:
            verdict = "подходящих офферов нет"
        elif self.image_broken:
            verdict = "образ не прошёл проверку — новых машин не беру"
        elif len(self.active_workers()) >= self.args.max_workers:
            verdict = f"достигнут --max-workers {self.args.max_workers}"
        elif self.provisioning > 0 and deficit <= 0:
            verdict = f"уже разворачивается {self.provisioning}"
        elif self.queue.empty():
            verdict = "очередь пуста — новой машине нечего дать"
        else:
            need_min = len(ready) < self.args.pool_min
            need_size = deficit > 0
            if not need_min and not need_size and worst and best.value < worst.value() * float(self.args.pool_improve):
                verdict = (f"лучший оффер ({best.value:.0f}) не лучше худшей машины пула ({worst.value():.0f}) "
                           f"в {float(self.args.pool_improve):g}× — не беру")
            else:
                why = (f"в пуле {len(ready)} готовых машин (< --pool-min {self.args.pool_min})" if need_min
                       else f"активных машин {len(ready) + self.provisioning} < требуемых {self.pool_size}" if need_size
                       else f"лучше худшей машины пула ({worst.value():.0f}) в {best.value / max(worst.value(), 1e-9):.2f}×")
                if self.try_spawn(best, why, force=need_min or need_size):
                    return
                verdict = "оффер не подошёл (профиль/модель/мало работы)"
        if verbose:
            info(f"Рынок: {self.board.get('рынок')}; {self.queue_line()}; решение: {verdict}")

    def _empty_plan(self) -> Plan:
        return Plan(jobs=[], target=self.plan.target, model_pref=self.plan.model_pref, total_frames=0, upload_bytes=0,
                    est_download_bytes=0, local_up_bps=self.plan.local_up_bps, local_down_bps=self.plan.local_down_bps,
                    time_value=self.plan.time_value, bid_margin=self.plan.bid_margin, disk_gb=self.plan.disk_gb,
                    optimize=self.plan.optimize, req=self.plan.req)

    def try_spawn(self, b: Bargain, why: str = "", force: bool = False) -> bool:
        """Заказ ещё одной машины под очередь. Без force — только если ей достанется не меньше --min-worker-minutes
        обработки (по её профилю) с учётом запуска."""
        self.tried_machines.add(b.offer.get("machine_id"))
        gpu = b.offer.get("gpu_name", "?")
        prof = GPU_PROFILES.get(gpu)
        if not prof:
            return False
        vram = round((b.offer.get("gpu_ram") or 0) / 1024.0)
        model_w = pick_model(self.args.model, vram)
        if not model_w:
            return False
        c0 = estimate_candidate(b.offer, b.kind, self._empty_plan())
        if c0 is None:
            return False
        spf_w = prof["spf"] * model_w["spf_factor"]
        # сколько работы достанется новичку: доля очереди пропорционально его скорости среди готовых машин
        rates = [1.0 / max(((m.calib_spf or GPU_PROFILES.get(m.cand.gpu, {"spf": 2.5})["spf"]) * m.cand.model["spf_factor"]), 1e-6)
                 for m in self.ready_members() if m.cand]
        share = (1.0 / spf_w) / (sum(rates) + 1.0 / spf_w)
        proc_w = self.queue.frames() * share * spf_w
        if not force and proc_w < self.args.min_worker_minutes * 60:
            debug(f"оффер {gpu}: на воркер набирается лишь {fmt_time(proc_w)} обработки — не стоит запуска")
            return False
        warn(f"Беру в пул: {gpu} {vram} ГБ, {b.kind}, {fmt_money(b.price)}/ч, DLPerf/$ {b.value:.0f} — {why}. "
             f"В очереди {len(self.queue)} чанк(ов); ему достанется ~{fmt_time(proc_w)} обработки, запуск ~{fmt_time(c0.t_setup)}")
        tag = f"W{len(self.workers) + 1} {gpu}"
        self.spawn_worker(c0, tag)
        return True

    def spawn_worker(self, cand: Candidate, tag: str) -> Session:
        w = Session(self.client, self.key, self._empty_plan(), self.args, self.workdir, jobs=[], tag=tag)
        w.parent = self.primary
        w.image_fn = self.image_fn
        self.attach_hooks(w)
        self.workers.append(w)
        with self.lock:
            self.board["пул"] = self.pool_status()
        t = threading.Thread(target=self.worker_main, args=(w, cand), daemon=True, name=tag)
        self.threads[tag] = t
        t.start()
        return w

    # ---- прерывание машины пула (перебили по цене / хост забрал GPU) ----
    def on_interrupt(self, s: Session) -> None:
        """Машина s недоступна: её чанки (текущий и подгруженные) возвращаются в очередь — их перезапустит другая
        машина; если s была единственной готовой в пуле — немедленно заказывается следующий по DLPerf/$ оффер;
        потолок бида s — цена этого (соседнего взятого) оффера, иначе цена ближайшей по DLPerf/$ машины пула."""
        if not self.enabled:
            return
        lost = s.in_flight()
        if lost:
            self.queue.push_front(lost, why=f"{s.tag} недоступна, чанки перезапустятся на другой машине")
        others = [m for m in self.ready_members() if m is not s]
        with self.lock:
            self.board["пул"] = self.pool_status()
        if others:
            nb = min(others, key=lambda m: abs(m.value() - s.value()))
            s.bid_cap = nb.cand.price if nb.cand else None
            info(f"{s.tag}: в пуле остаются {len(others)} готовых машин; потолок бида — цена соседней {nb.tag}: "
                 f"{fmt_money(s.bid_cap)}/ч")
            return
        if self.provisioning > 0 or len(self.active_workers()) >= self.args.max_workers:
            info(f"{s.tag}: замена уже разворачивается" if self.provisioning else f"{s.tag}: лимит --max-workers, замену не заказываю")
            return
        if self.queue.empty():
            info(f"{s.tag}: очередь пуста — замену не заказываю, жду возврата")
            return
        try:
            market = list_market(self.client, self.args, self.plan, self.used_machines(), self.image_dit)
        except Exception as e:  # noqa
            warn(f"листинг для замены: {e!r}")
            market = []
        cand = None
        for b in market:
            c = estimate_candidate(b.offer, b.kind, self._empty_plan())
            if c is not None:
                cand = (b, c)
                break
        if not cand:
            warn(f"{s.tag}: единственная машина пула недоступна, а подходящих офферов на рынке нет — жду её возврата")
            return
        b, c = cand
        warn(f"{s.tag} была единственной готовой машиной пула — немедленно беру следующий оффер по DLPerf/$: "
             f"{c.gpu} {c.vram_gb:.0f} ГБ, {b.kind}, {fmt_money(c.price)}/ч, DLPerf/$ {b.value:.0f}; в очереди {len(self.queue)} чанк(ов); "
             f"бид {s.tag} будет расти раз в 5 с, но не выше {fmt_money(c.price)}/ч")
        s.bid_cap = c.price
        self.tried_machines.add(b.offer.get("machine_id"))
        self.spawn_worker(c, f"W{len(self.workers) + 1} {c.gpu}")

    def on_lost(self, s: Session) -> bool:
        """Машина s не вернулась: её чанки уже в очереди (on_interrupt); инстанс удаляется. True, если пул может
        продолжить без неё (есть другие машины или разворачивается замена)."""
        if not self.enabled:
            return False
        s.lost = True
        s.ssh = None
        lost = [j for j in s.jobs if j.assigned_to == s.tag and j.status in ("pending", "uploaded", "done")]
        if lost:
            self.queue.push_front(lost, why=f"{s.tag} потеряна")
        others = [m for m in self.members() if m is not s]
        if not others and self.provisioning <= 0 and not self.active_workers():
            if self.queue.empty() and self.all_done():
                return False
            warn(f"{s.tag}: инстанс потерян, других машин в пуле нет — очередь ({len(self.queue)} чанк(ов)) ждёт новую машину: "
                 f"беру следующий оффер по DLPerf/$ на ближайшем скане рынка")
            self.scan_requested = True
        else:
            warn(f"{s.tag}: инстанс потерян; пул продолжает без него ({self.queue_line()})")
        # удалить сразу: иначе поднятый бид может позже «выиграть» машину, и простаивающий инстанс начнёт тарифицироваться
        if s.instance_id and not self.args.keep_instance:
            try:
                s.destroy()
            except Exception as e:  # noqa
                err(f"{s.tag}: не удалось удалить потерянный инстанс: {e}")
        with self.lock:
            self.board["пул"] = self.pool_status()
        return True

    def adopt_existing(self, inst: dict, tag: str, prebuild_repo: str = "") -> Optional[Tuple[Session, Candidate]]:
        """Сессия для уже работающего инстанса прошлого запуска (без заказа): подключение, проверка, включение в пул."""
        gpu = str(inst.get("gpu_name") or "")
        vram = round((inst.get("gpu_ram") or 0) / 1024.0)
        model = pick_model(self.args.model, vram) or pick_model("3b", max(vram, 24))
        if not model:
            return None
        kind = "bid" if inst.get("is_bid") else "on-demand"
        fake = dict(inst)
        fake.setdefault("reliability2", 1.0)
        fake.setdefault("min_bid", inst.get("dph_total"))
        cand = estimate_candidate(fake, kind, self._empty_plan()) if gpu in GPU_PROFILES else None
        if cand is None:
            prof_price = float(inst.get("dph_total") or 0.0)
            cand = Candidate(offer=fake, kind=kind, price=prof_price, model=model, t_setup=0.0, t_upload=0.0, t_proc=0.0,
                             t_download=0.0, cost_total=0.0, score=0.0)
        h = Session(self.client, self.key, self._empty_plan(), self.args, self.workdir, jobs=[], tag=tag)
        h.parent, h.image_fn, h.t_ordered = self.primary, self.image_fn, time.time()
        h.instance_id, h.instance, h.created_at = int(inst["id"]), inst, float(inst.get("start_date") or time.time())
        h.root, h.cand, h.model = remote_root_for(h.instance_id), cand, model
        h.price_hint = float(inst.get("dph_total") or 0.0)
        img = str(inst.get("image_uuid") or inst.get("image") or "")
        if prebuild_repo and img.startswith(prebuild_repo + ":"):
            h.image, h.prebuilt = img, True
        self.attach_hooks(h)
        return h, cand

    def adopt_existing_workers(self, existing: List[dict], prebuild_repo: str = "") -> None:
        """Основной уже выбран (--instance/--resume): остальные инстансы прошлых запусков — в пул воркерами."""
        for k, inst in enumerate(existing, 1):
            got = self.adopt_existing(inst, f"E{k} {inst.get('gpu_name') or '?'}", prebuild_repo)
            if not got:
                continue
            h, cand = got
            info(f"Переиспользую инстанс #{h.instance_id} прошлого запуска: {cand.gpu} {cand.vram_gb:.0f} ГБ, "
                 f"{fmt_money(h.price_hint)}/ч — в пул воркером")
            self.workers.append(h)
            t = threading.Thread(target=self.hedge_main, args=(h, cand, True), daemon=True, name=h.tag)
            self.threads[h.tag] = t
            t.start()

    def acquire_pool(self, cands: List[Candidate], existing: Optional[List[dict]] = None, prebuild_repo: str = "") -> None:
        """Набор первичного пула без ожидания. Сначала — уже работающие инстансы этого скрипта с прошлых запусков
        (метка + наш образ): они подключаются без заказа. Затем заказывается лучший по DLPerf/$ оффер (при
        --pool-size N — столько, чтобы всего было N) и, пока ни одна машина не готова, каждые --pool-scan с
        перезапрашивается листинг; оффер лучше лучшего из заказанных в --pool-improve раз заказывается тоже
        (до 1+--max-workers одновременно). Первая готовая машина становится основной (её инстанс переходит к основной
        сессии); остальные входят в пул воркерами и берут чанки из очереди; не поднявшиеся за --boot-deadline удаляются."""
        ranked = sorted((c for c in cands if not c.offer.get("_existing")), key=lambda c: -(offer_value(c.offer, c.price) or 0.0))
        hedges: List[Tuple[Session, Candidate, threading.Thread]] = []
        self.primary_chosen = threading.Event()

        def order(cand: Candidate) -> None:
            n = len(hedges) + 1
            h = Session(self.client, self.key, self._empty_plan(), self.args, self.workdir, jobs=[], tag=f"H{n} {cand.gpu}")
            h.parent, h.image_fn, h.t_ordered = self.primary, self.image_fn, time.time()
            self.attach_hooks(h)
            self.tried_machines.add(cand.offer.get("machine_id"))
            t = threading.Thread(target=self.hedge_main, args=(h, cand), daemon=True, name=h.tag)
            hedges.append((h, cand, t))
            t.start()

        n_existing = 0
        for inst in (existing or []):
            got = self.adopt_existing(inst, f"E{n_existing + 1} {inst.get('gpu_name') or '?'}", prebuild_repo)
            if not got:
                continue
            h, cand = got
            n_existing += 1
            info(f"Переиспользую инстанс #{h.instance_id} прошлого запуска: {cand.gpu} {cand.vram_gb:.0f} ГБ, "
                 f"{fmt_money(h.price_hint)}/ч, метка «{inst.get('label')}» — подключаюсь без заказа")
            t = threading.Thread(target=self.hedge_main, args=(h, cand, True), daemon=True, name=h.tag)
            hedges.append((h, cand, t))
            t.start()
        n_init = max(0 if n_existing else 1, min(self.pool_size - n_existing, 1 + int(self.args.max_workers) - n_existing, len(ranked)))
        if n_init > 0 and ranked:
            info(f"Набор пула: заказываю лучший оффер по DLPerf/$ ({ranked[0].gpu}, {fmt_money(ranked[0].price)}/ч, "
                 f"DLPerf/$ {offer_value(ranked[0].offer, ranked[0].price) or 0:.0f}); пока он поднимается, каждые "
                 f"{self.args.pool_scan:.0f} с смотрю листинг и беру офферы лучше него в ≥{self.args.pool_improve:g}×")
            order(ranked[0])
            for c in ranked[1:n_init]:               # регулятор: требуется несколько машин — заказываем их сразу
                info(f"Регулятор пула (--pool-size {self.pool_size}): заказываю и {c.gpu} за {fmt_money(c.price)}/ч, "
                     f"DLPerf/$ {offer_value(c.offer, c.price) or 0:.0f}")
                order(c)
        elif n_existing:
            info(f"Набор пула: {n_existing} инстанс(ов) прошлого запуска — новые не заказываю (требуется {self.pool_size})")
        n_init = max(n_init, 0)
        t_scan = time.time()
        sp = Spinner("Жду первую готовую машину пула")
        next_cands = ranked[n_init:]
        handled_broken: set = set()
        while True:
            failed = [h for h, c, t in hedges if not t.is_alive() and not h.booted]
            for h in failed:
                e = h.boot_exc
                if isinstance(e, PrebuiltBroken) and id(h) not in handled_broken:
                    handled_broken.add(id(h))
                    if e.py_have and self.rebuild_fn:
                        sp.close()
                        warn(f"{h.tag}: {e}")
                        info(f"Пересобираю образ под Python {e.py_have} контейнера (слой весов и базовые слои переиспользуются)")
                        if not self.rebuild_fn(e.py_have):
                            raise FatalError("Образ под нужную версию Python собрать не удалось.")
                        sp = Spinner("Жду первую готовую машину пула")
                    else:
                        sp.close()
                        raise FatalError(str(e))
            if len(failed) >= max(1, int(self.args.max_attempts)) and not any(t.is_alive() for h, c, t in hedges):
                sp.close()
                raise FatalError(f"Ни один из {len(failed)} заказанных инстансов не поднялся (--max-attempts {self.args.max_attempts}): "
                                 + "; ".join(f"{h.tag}: {h.boot_error[:120]}" for h in failed[-3:]))
            alive = [(h, c, t) for h, c, t in hedges if t.is_alive() or h.booted]
            booted = [h for h, c, t in alive if h.booted and not h.adopted]
            if booted:
                winner = booted[0]
                sp.close(f"Первая готовая машина: {winner.tag} — становится основной")
                self.primary.adopt(winner)
                self.primary_chosen.set()
                break
            # не поднявшиеся за --boot-deadline — удалить
            for h, c, t in alive:
                if h.instance_id and not h.booted and time.time() - (h.t_ordered or time.time()) > self.args.boot_deadline:
                    warn(f"{h.tag}: не поднялся за {fmt_time(self.args.boot_deadline)} — удаляю")
                    try:
                        h.destroy(quiet=True)
                    except Exception:  # noqa
                        pass
            alive = [(h, c, t) for h, c, t in hedges if t.is_alive()]
            if not alive:
                if not next_cands:
                    cands2 = list_market(self.client, self.args, self.plan, self.used_machines(), self.image_dit)
                    if not cands2:
                        sp.close()
                        raise FatalError("Ни один инстанс не поднялся, подходящих офферов больше нет.")
                    b = cands2[0]
                    c = estimate_candidate(b.offer, b.kind, self._empty_plan())
                    if c is None:
                        time.sleep(self.args.pool_scan)
                        continue
                    order(c)
                else:
                    order(next_cands.pop(0))
                t_scan = time.time()
            elif (time.time() - t_scan >= self.args.pool_scan or self.scan_requested) and len(alive) < 1 + self.args.max_workers:
                t_scan = time.time()
                self.scan_requested = False
                try:
                    market = list_market(self.client, self.args, self.plan, self.used_machines(), self.image_dit)
                except Exception as e:  # noqa
                    market = []
                    debug(f"листинг при наборе пула: {e!r}")
                best_ordered = max((offer_value(c.offer, c.price) or 0.0) for h, c, t in alive)
                if market and market[0].value >= best_ordered * float(self.args.pool_improve):
                    b = market[0]
                    c = estimate_candidate(b.offer, b.kind, self._empty_plan())
                    if c is not None:
                        warn(f"На рынке оффер лучше заказанных: {c.gpu} {fmt_money(c.price)}/ч, DLPerf/$ {b.value:.0f} "
                             f"(против {best_ordered:.0f}) — заказываю и его, первый готовый станет основным")
                        order(c)
            sp.tick(" ‖ ".join(f"{h.tag}: {self.board.get(h.tag) or ('заказан' if not h.instance_id else 'поднимается')}"
                               for h, c, t in alive) or "…")
            time.sleep(1.0)
        # остальные кандидаты остаются в пуле воркерами (их потоки продолжают после primary_chosen)
        for h, c, t in hedges:
            if h is not self.primary and not h.adopted and t.is_alive():
                self.workers.append(h)
                self.threads[h.tag] = t

    def hedge_main(self, h: Session, cand: Candidate, existing: bool = False) -> None:
        """Кандидат при наборе пула: поднимается (или, если это инстанс прошлого запуска, — подключается);
        если стал основным — поток заканчивается (инстанс передан); иначе ждёт выбора основного и работает воркером
        (берёт чанки из очереди)."""
        _tls.prefix = f"[{h.tag}] "
        _tls.board_cb = lambda line, _t=h.tag: self.board.__setitem__(_t, line)
        try:
            try:
                if existing:
                    h.connect_ssh(timeout=180)
                    h.check_link(strict=False)
                else:
                    h.provision([cand])
                h.bootstrap()
                h.booted = True
            except (FatalError, TimeoutError) as e:
                h.boot_error, h.boot_exc = str(e), e
                if existing:
                    warn(f"инстанс #{h.instance_id} прошлого запуска не подключился: {e} — оставляю его "
                         f"(проверьте: --list-instances, удалить: --destroy-instance {h.instance_id})")
                    h.instance_id = None                      # не удалять в finally: он не наш заказ
                else:
                    warn(f"не поднялся: {e}")
                return
            # ждём, пока основной выбран (возможно, это мы)
            while not self.primary_chosen.is_set():
                time.sleep(0.5)
            if h.adopted:
                with self.lock:
                    self.board.pop(h.tag, None)
                return
            self.worker_serve(h)
        except Exception as e:  # noqa
            err(f"Кандидат пула упал: {e!r}")
        finally:
            if h.instance_id and not h.adopted:
                if self.args.keep_instance:
                    warn(f"--keep-instance: инстанс #{h.instance_id} оставлен")
                else:
                    h.destroy()
            with self.lock:
                self.board.pop(h.tag, None)

    def worker_serve(self, w: Session) -> None:
        """Воркер после установки: runner (пустая очередь на инстансе), затем цикл — чанки берутся из общей очереди."""
        try:
            w.write_jobs_and_start_runner()
            w.monitor_and_download()
            failed = [j for j in w.jobs if j.assigned_to == w.tag and j.status == "failed"]
            if failed:
                warn(f"Воркер не справился с {len(failed)} чанк(ами): " + ", ".join(j.label for j in failed))
            elif w.jobs:
                ok("Все чанки воркера скачаны и проверены")
            if w.retired:
                ok("Инстанс выбывает из пула как худший — освобождаю")
        except (FatalError, TimeoutError) as e:
            if isinstance(e, PrebuiltBroken):
                self.image_broken = str(e)
                err(f"Воркер: {e} — новых воркеров не заказываю")
            else:
                warn(f"Воркер остановлен: {e}. Его чанки возвращаются в очередь.")
            self.queue.push_front([j for j in w.jobs if j.assigned_to == w.tag and j.status in ("pending", "uploaded", "done")],
                                  why=f"{w.tag} остановлен")
        finally:
            self.rebalance(f"отключился {w.tag}")

    def worker_main(self, w: Session, cand: Candidate) -> None:
        _tls.prefix = f"[{w.tag}] "
        _tls.board_cb = lambda line, _t=w.tag: self.board.__setitem__(_t, line)
        w.t_ordered = time.time()
        try:
            try:
                w.provision([cand])
                w.bootstrap()
                w.booted = True
            except (FatalError, TimeoutError) as e:
                if isinstance(e, PrebuiltBroken):
                    self.image_broken = str(e)
                    err(f"Воркер: {e} — новых воркеров не заказываю")
                else:
                    warn(f"Воркер не поднялся: {e}")
                return
            self.worker_serve(w)
        except Exception as e:  # noqa
            err(f"Воркер упал: {e!r}. Его чанки возвращаются в очередь.")
            self.queue.push_front([j for j in w.jobs if j.assigned_to == w.tag and j.status in ("pending", "uploaded", "done")],
                                  why=f"{w.tag} упал")
        finally:
            if w.instance_id:
                if self.args.keep_instance:
                    warn(f"--keep-instance: инстанс воркера #{w.instance_id} оставлен")
                else:
                    w.destroy()
            with self.lock:
                self.board.pop(w.tag, None)

    def destroy_all(self) -> None:
        for w in self.workers:
            if w.instance_id and not self.args.keep_instance:
                try:
                    w.destroy()
                except Exception as e:  # noqa
                    err(f"Не удалось удалить инстанс воркера #{w.instance_id}: {e}")

    def rent_cost(self) -> Tuple[float, float]:
        """(часы, $) по всем воркерам (оценка по цене × времени жизни)."""
        hours = cost = 0.0
        for w in self.workers:
            if w.created_at and w.cand:
                h = ((w.destroyed_at or time.time()) - w.created_at) / 3600
                hours += h
                cost += h * w.cand.price
        return hours, cost


def drop_page_cache(f, start: int = 0, length: int = 0, sync: bool = False) -> None:
    """Просит ядро не держать прочитанное/записанное в page cache (POSIX_FADV_DONTNEED). Важно в WSL2:
    страничный кэш от 16-ГБ файлов раздувает память VM до лимита, а Windows её обратно не получает."""
    try:
        if sync:
            os.fdatasync(f.fileno())
        os.posix_fadvise(f.fileno(), start, length, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError, ValueError):
        pass


def sha256_of(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        n = 0
        for chunk in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(chunk)
            n += len(chunk)
            if n % (256 << 20) == 0:
                drop_page_cache(f, 0, n)
        drop_page_cache(f)
    return h.hexdigest()


# ----------------------------------------------------------------------------
# Сборка заданий, аргументы командной строки, main
# ----------------------------------------------------------------------------
DAEMON_REGISTRY = VAST_KEY_FILE.parent / "vast_upscale_daemons.json"


def _registry_load() -> dict:
    try:
        return json.loads(DAEMON_REGISTRY.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def _registry_save(reg: dict) -> None:
    try:
        DAEMON_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
        DAEMON_REGISTRY.write_text(json.dumps(reg, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError:
        pass


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (OSError, TypeError):
        return False


def derive_workdir(args) -> Optional[Path]:
    """Папка запуска: --workdir, иначе <рабочая папка скрипта>/runs/<имя входа>-<hash пути>
    (служебные файлы — сконвертированные видео, состояние, логи — не растят папку с исходниками и, под WSL2,
    ext4.vhdx). Для запусков старых версий, чьё состояние лежит в <папка входа>/.vast_upscale, берётся оно."""
    if args.workdir:
        return Path(args.workdir).expanduser().resolve()
    if args.inputs:
        first = Path(args.inputs[0]).expanduser().resolve()
        base = first if first.is_dir() else first.parent
        legacy = base / ".vast_upscale"
        h = hashlib.sha1(str(first).encode("utf-8")).hexdigest()[:8]
        name = re.sub(r"[^\w.-]+", "_", first.stem if first.is_file() else first.name) or "run"
        wd = LOCAL_CACHE_DIR / "runs" / f"{name}-{h}"
        if (legacy / "state.json").exists() and not (wd / "state.json").exists():
            return legacy
        return wd
    return None


def apply_cache_setting(args) -> None:
    """Молча выставляет LOCAL_CACHE_DIR из --cache-dir или сохранённых настроек (вопросы и перенос — в ensure_cache_dir);
    нужно до derive_workdir, т.е. до --attach/--stop/--status/--daemon."""
    global LOCAL_CACHE_DIR
    if getattr(args, "cache_dir", None):
        LOCAL_CACHE_DIR = Path(args.cache_dir).expanduser().resolve()
        return
    saved = load_settings().get("cache_dir")
    if saved:
        LOCAL_CACHE_DIR = Path(saved).expanduser()


def pick_daemon(args) -> Optional[Tuple[Path, dict]]:
    """Запись реестра для --attach/--stop/--status: по workdir из аргументов или самый свежий живой запуск."""
    reg = _registry_load()
    wd = derive_workdir(args)
    if wd and str(wd) in reg:
        return wd, reg[str(wd)]
    alive = [(k, v) for k, v in reg.items() if _pid_alive(int(v.get("pid") or 0))]
    if alive:
        k, v = max(alive, key=lambda kv: kv[1].get("started", 0))
        return Path(k), v
    if reg:
        k, v = max(reg.items(), key=lambda kv: kv[1].get("started", 0))
        return Path(k), v
    return None


def daemonize(workdir: Path, argv_desc: str) -> bool:
    """Двойной fork: возвращает True в потомке (продолжаем работу в фоне), False в родителе."""
    workdir.mkdir(parents=True, exist_ok=True)
    log = workdir / "run.log"
    pid = os.fork()
    if pid > 0:
        # родитель: ждём, пока внук зарегистрируется
        for _ in range(50):
            time.sleep(0.1)
            reg = _registry_load()
            if str(workdir) in reg and _pid_alive(int(reg[str(workdir)].get("pid") or 0)):
                break
        reg = _registry_load().get(str(workdir), {})
        ok(f"Запущено в фоне (pid {reg.get('pid', '?')}). Лог: {log}")
        info(f"Подключиться к прогрессу:  python3 {Path(sys.argv[0]).name} --attach" + (f" --workdir {workdir}" if True else ""))
        info(f"Остановить (инстансы будут удалены):  python3 {Path(sys.argv[0]).name} --stop --workdir {workdir}")
        if is_wsl():
            warn("WSL2: фоновый процесс живёт, пока открыто хотя бы одно окно/сессия WSL — при закрытии последнего "
                 "окна Windows останавливает всю VM вместе с задачей. Оставьте окно открытым или вернитесь через --attach.")
        return False
    os.setsid()
    if os.fork() > 0:
        os._exit(0)
    # внук: отвязываем терминал
    sys.stdout.flush(); sys.stderr.flush()
    fd = os.open(str(log), os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o644)
    null = os.open(os.devnull, os.O_RDONLY)
    os.dup2(null, 0); os.dup2(fd, 1); os.dup2(fd, 2)
    os.close(fd); os.close(null)
    sys.stdout = io.TextIOWrapper(os.fdopen(1, "wb", 0), encoding="utf-8", write_through=True)
    sys.stderr = io.TextIOWrapper(os.fdopen(2, "wb", 0), encoding="utf-8", write_through=True)
    global _IS_TTY, _progress_file
    _IS_TTY = False
    _progress_file = workdir / "progress.json"
    reg = _registry_load()
    reg[str(workdir)] = {"pid": os.getpid(), "started": time.time(), "cmd": argv_desc, "log": str(log)}
    _registry_save(reg)
    _raw_print("")
    _raw_print(bold(f"=== фоновый запуск {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} pid {os.getpid()}: {argv_desc}"))

    def _term(signum, frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, _term)
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    return True


def daemon_finish(workdir: Path, exit_code: int) -> None:
    reg = _registry_load()
    ent = reg.get(str(workdir))
    if ent:
        ent["finished"] = time.time()
        ent["exit_code"] = exit_code
        _registry_save(reg)
    _write_progress("")


def cmd_attach(args) -> int:
    """Показывает лог фонового запуска и живую строку прогресса; Ctrl+C — отсоединиться (задача продолжается)."""
    pick = pick_daemon(args)
    if not pick:
        err("Фоновых запусков не найдено (реестр пуст). Запустите с --daemon.")
        return 1
    wd, ent = pick
    log = Path(ent.get("log") or (wd / "run.log"))
    pid = int(ent.get("pid") or 0)
    alive = _pid_alive(pid)
    info(f"Фоновый запуск pid {pid} ({'работает' if alive else 'завершён'}), рабочая папка {wd}")
    if not log.exists():
        err(f"Лог {log} не найден.")
        return 1
    with open(log, "rb") as f:
        f.seek(0, 2)
        size = f.tell()
        f.seek(max(0, size - 6000))
        tail = f.read().decode("utf-8", "replace")
        for line in tail.splitlines()[-25:]:
            _raw_print(line)
        pos = f.tell()
        if not alive:
            info(f"Задача завершена (код {ent.get('exit_code')}).")
            return 0
        info(dim("--- подключено; Ctrl+C — отсоединиться, задача продолжит работать ---"))
        prog = wd / "progress.json"
        last_line = ""
        try:
            while True:
                f.seek(pos)
                chunk = f.read()
                if chunk:
                    pos = f.tell()
                    for line in chunk.decode("utf-8", "replace").splitlines():
                        _raw_print(line)
                try:
                    pl = json.loads(prog.read_text(encoding="utf-8")).get("line", "")
                except (OSError, ValueError):
                    pl = ""
                if pl != last_line and _IS_TTY:
                    last_line = pl
                    with _print_lock:
                        cols = shutil.get_terminal_size((120, 20)).columns
                        sys.stdout.write("\r" + ("  " + pl)[:cols - 1] + "\033[K")
                        sys.stdout.flush()
                        global _line_open
                        _line_open = bool(pl)
                if not _pid_alive(pid):
                    time.sleep(0.5)
                    f.seek(pos)
                    for line in f.read().decode("utf-8", "replace").splitlines():
                        _raw_print(line)
                    ent = _registry_load().get(str(wd), {})
                    info(f"Задача завершена (код {ent.get('exit_code')}).")
                    return int(ent.get("exit_code") or 0)
                time.sleep(0.5)
        except KeyboardInterrupt:
            _raw_print("")
            info("Отсоединился. Задача продолжает работать в фоне; --attach — подключиться снова, --stop — остановить.")
            return 0


def cmd_stop(args) -> int:
    pick = pick_daemon(args)
    if not pick:
        err("Фоновых запусков не найдено.")
        return 1
    wd, ent = pick
    pid = int(ent.get("pid") or 0)
    if not _pid_alive(pid):
        info(f"Фоновый запуск pid {pid} уже завершён (код {ent.get('exit_code')}).")
        return 0
    os.kill(pid, signal.SIGTERM)
    info(f"Отправил SIGTERM pid {pid}: задача остановится и удалит свои инстансы (см. --attach).")
    return 0


def cmd_status(args) -> int:
    reg = _registry_load()
    if not reg:
        info("Фоновых запусков нет.")
        return 0
    for k, v in sorted(reg.items(), key=lambda kv: -kv[1].get("started", 0)):
        pid = int(v.get("pid") or 0)
        alive = _pid_alive(pid)
        line = ""
        try:
            line = json.loads((Path(k) / "progress.json").read_text(encoding="utf-8")).get("line", "")
        except (OSError, ValueError):
            pass
        st = "работает" if alive else f"завершён (код {v.get('exit_code')})"
        _raw_print(f"  pid {pid:<7} {st:<20} {_dt.datetime.fromtimestamp(v.get('started', 0)).strftime('%Y-%m-%d %H:%M')}  {k}")
        note(v.get("cmd", ""))
        if line and alive:
            note("→ " + line)
    return 0


def safe_name(p: Path) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", p.stem).strip("_")[:48] or "video"
    return f"{stem}-{hashlib.sha1(str(p.resolve()).encode()).hexdigest()[:6]}"


def make_jobs(files: List[Path], args, workdir: Path) -> List[Job]:
    jobs: List[Job] = []
    for f in files:
        try:
            inf = ffprobe(f)
        except FatalError as e:
            warn(str(e))
            continue
        j = Job(src=f, info=inf, name=safe_name(f))
        out_dir = Path(args.out_dir).expanduser() if args.out_dir else f.parent
        j.out_path = out_dir / f"{f.stem}{args.suffix}.mp4"
        j.remote_in = f"{REMOTE_ROOT}/in/{j.name}.cfr.mp4"
        j.remote_out = f"{REMOTE_ROOT}/out/{j.name}.mp4"
        short = min(inf.width, inf.height)
        desc = (f"{f.name}: {inf.width}x{inf.height} {inf.fps:.3f} fps, {fmt_time(inf.duration)}, ~{inf.frames} кадров, "
                f"{inf.codec}/{inf.pix_fmt}{', VFR' if inf.vfr else ''}{', без звука' if not inf.has_audio else ''}, {fmt_bytes(inf.size_bytes)}")
        if j.out_path.exists() and not args.force:
            j.status = "skipped"
            warn(desc + f" → результат уже есть ({j.out_path.name}), пропускаю (используйте --force)")
        elif short >= args.target and not args.force:
            j.status = "skipped"
            warn(desc + f" → короткая сторона уже ≥ {args.target}, пропускаю (используйте --force)")
        else:
            info(desc)
        jobs.append(j)
    return jobs


class PoolKeys(threading.Thread):
    """Интерактивный регулятор пула (только в терминале, не в --daemon): клавиши «+»/«-» — требуемое число активных
    машин, «s» — состояние пула (нужно · активно · разворачивается), «?» — подсказка. Терминал переводится в
    cbreak-режим на время обработки и восстанавливается в stop()."""

    HELP = "клавиши: «+»/«-» — больше/меньше машин в пуле, «r» — проверить рынок сейчас, «s» — состояние пула и очереди, «?» — подсказка"

    def __init__(self, orch: "Orchestrator"):
        super().__init__(daemon=True, name="pool-keys")
        self.orch = orch
        self.fd = sys.stdin.fileno()
        self._saved = None
        self._stop_evt = threading.Event()        # не _stop: так называется внутренний метод Thread

    @staticmethod
    def available(args) -> bool:
        return sys.stdin.isatty() and _IS_TTY and not getattr(args, "daemon", False) and os.name != "nt"

    def handle(self, ch: str) -> None:
        """Обработка одной клавиши (вынесено для тестов)."""
        if ch in ("+", "="):
            self.orch.set_pool_size(self.orch.pool_size + 1, "клавиша +")
        elif ch in ("-", "_"):
            self.orch.set_pool_size(self.orch.pool_size - 1, "клавиша -")
        elif ch in ("s", "p", "ы", "з"):
            info(f"Пул: {self.orch.pool_status()}; {self.orch.queue_line()}")
        elif ch in ("r", "к"):
            self.orch.request_scan("клавиша r")
        elif ch in ("?", "h"):
            info(self.HELP)

    def run(self) -> None:
        import select
        import termios
        import tty
        try:
            self._saved = termios.tcgetattr(self.fd)
            tty.setcbreak(self.fd)
        except (termios.error, OSError, ValueError):
            self._saved = None
            return
        try:
            while not self._stop_evt.is_set():
                r, _, _ = select.select([self.fd], [], [], 0.5)
                if not r:
                    continue
                try:
                    ch = os.read(self.fd, 1).decode("utf-8", "ignore")
                except OSError:
                    break
                if ch:
                    self.handle(ch)
        finally:
            self.restore()

    def restore(self) -> None:
        import termios
        if self._saved is not None:
            try:
                termios.tcsetattr(self.fd, termios.TCSADRAIN, self._saved)
            except (termios.error, OSError):
                pass
            self._saved = None

    def stop(self) -> None:
        self._stop_evt.set()
        try:
            self.join(timeout=2.0)
        except RuntimeError:
            pass
        self.restore()


def ask_with_timeout(question: str, default: bool, timeout: float) -> bool:
    """Да/нет с таймаутом (для обработчика Ctrl+C: не оставлять инстанс тарифицироваться вечно)."""
    if not sys.stdin.isatty():
        return default
    import select
    _raw_print(f"  {question} [{'Y/n' if default else 'y/N'}] (через {int(timeout)} с — «{'да' if default else 'нет'}»): ", end="")
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    if not r:
        _raw_print("")
        return default
    a = sys.stdin.readline().strip().lower()
    if not a:
        return default
    return a in ("y", "yes", "д", "да")


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="vast_upscale.py",
        description="AI-апскейл видео (SeedVR2) на арендованном GPU vast.ai: от проверки зависимостей до скачивания результата.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            Примеры:
              %(prog)s --check                          проверить зависимости и сопряжение с vast.ai
              %(prog)s clip.mp4                         апскейл до 1080p, результат clip_1080p.mp4 рядом
              %(prog)s ./videos --recursive --yes       вся папка, без вопросов
              %(prog)s clip.mp4 --target 1440 --model 7b --optimize speed
              %(prog)s clip.mp4 --dry-run               только анализ, план и таблица офферов
              %(prog)s clip.mp4 --resume                продолжить после обрыва (тот же инстанс)
            """))
    p.add_argument("inputs", nargs="*", help="видеофайл(ы) и/или директории")
    g = p.add_argument_group("результат")
    g.add_argument("--target", type=int, default=1080, help="целевая короткая сторона в пикселях (по умолчанию 1080 → 1080p)")
    g.add_argument("--max-long-side", type=int, default=0, help="ограничить длинную сторону (0 = без ограничения)")
    g.add_argument("--model", choices=["auto", "3b", "7b", "7b-sharp"], default="auto",
                   help="SeedVR2: auto (7B при VRAM ≥ 40 ГБ, иначе 3B), 3b, 7b, 7b-sharp (резче; для мягких источников)")
    g.add_argument("--suffix", default=None, help="суффикс имени результата (по умолчанию _<target>p)")
    g.add_argument("--out-dir", default=None, help="куда класть результаты (по умолчанию рядом с исходником)")
    g.add_argument("--recursive", action="store_true", help="искать видео в поддиректориях")
    g.add_argument("--force", action="store_true", help="обрабатывать даже если результат есть или видео уже ≥ цели")
    g.add_argument("--pre-downscale", type=int, default=0, help="сначала уменьшить короткую сторону до N (для сильно испорченных источников)")
    g.add_argument("--seed", type=int, default=42)
    g.add_argument("--batch-size", type=int, default=0, help="принудительный batch_size SeedVR2 (4n+1); 0 = авто по VRAM")
    g.add_argument("--extra-args", default="", help="дополнительные флаги inference_cli.py, строкой")
    g = p.add_argument_group("выбор GPU / цена")
    g.add_argument("--optimize", choices=["cost", "ratio", "balanced", "speed"], default="cost",
                   help="cost (по умолчанию): минимальная стоимость всего run (аренда × полное время прогона + диск + трафик); "
                        "ratio: стоимость run × длительность run; balanced: стоимость + $/ч ожидания × время; speed: только время run")
    g.add_argument("--time-value", type=float, default=None, help="для balanced: сколько $ стоит час вашего ожидания (по умолчанию 1)")
    g.add_argument("--rental", choices=["auto", "on-demand", "bid"], default="auto",
                   help="on-demand (надёжно), bid (interruptible, дешевле, с риском пауз), auto — по оценке")
    g.add_argument("--bid-margin", type=float, default=0.25, help="надбавка к min_bid при биде (0.25 = +25%%)")
    g.add_argument("--max-price", type=float, default=0.0, help="потолок $/ч (0 = нет)")
    g.add_argument("--gpu", default="", help="ограничить список GPU, через запятую (напр. 'RTX 4090,RTX 5090,H100 SXM')")
    g.add_argument("--min-vram", type=float, default=24, help="минимум VRAM, ГБ")
    g.add_argument("--min-reliability", type=float, default=0.95)
    g.add_argument("--min-inet", type=float, default=200, help="минимум входящей скорости хоста, Мбит/с")
    g.add_argument("--min-cuda", type=float, default=12.6, help="минимальная версия CUDA драйвера хоста")
    g.add_argument("--country", default="", help="коды стран через запятую (напр. 'DE,NL,FI')")
    g.add_argument("--image", default=DEFAULT_IMAGE, help="docker-образ контейнера")
    g.add_argument("--disk", type=float, default=0, help="размер диска инстанса, ГБ (0 = авто)")
    g.add_argument("--max-attempts", type=int, default=4, help="сколько офферов перебрать при неудачном запуске")
    g = p.add_argument_group("предсобранный образ и проверка канала")
    g.add_argument("--prebuild", choices=["auto", "local", "strict", "inline", "never"], default="auto",
                   help="local: если образа с SeedVR2+весами нет — собрать его НА ЭТОЙ машине без Docker (слои OCI + Registry API, "
                        "push в ваш Docker Hub) до выбора и покупки инстанса; strict: собрать снапшотом на самом дешёвом арендуемом инстансе; "
                        "inline: снапшот с рабочего инстанса в фоне; never: ставить всё при старте. "
                        "auto = local при наличии Docker и учётных данных Docker Hub, иначе never")
    g.add_argument("--docker-user", default=None, help="логин Docker Hub (или env DOCKER_USER)")
    g.add_argument("--docker-pass", default=None, help="пароль/access-token Docker Hub (или env DOCKER_PASS)")
    g.add_argument("--image-repo", default=None, help="репозиторий для образов, напр. user/vast-upscale (по умолчанию <docker-user>/vast-upscale)")
    g.add_argument("--snapshot-wait", type=float, default=1500, help="сколько ждать публикации снапшота в Docker Hub, с")
    g.add_argument("--base-python", default=None, help="версия Python в базовом образе (для подготовки зависимостей; по умолчанию 3.11 для образа по умолчанию)")
    g.add_argument("--cache-dir", "--work-root", dest="cache_dir", default=None,
                   help="рабочая папка скрипта: кэш сборки образа (веса, исходники, зависимости, digest слоёв, сессии загрузки) и "
                        "служебные файлы запусков (runs/: сконвертированные видео, состояние, логи). Спрашивается при первом запуске и "
                        "сохраняется в настройках; этот ключ меняет путь (см. --cache-migrate)")
    g.add_argument("--cache-migrate", choices=["move", "delete", "keep"], default=None,
                   help="при смене --cache-dir: перенести старые данные кэша в новый путь, удалить их или оставить (без терминала по умолчанию keep)")
    g.add_argument("--ignore-disk-check", action="store_true", help="не останавливаться, если места для сборки образа мало")
    g.add_argument("--min-link", type=float, default=100, help="минимальная фактическая полоса инстанса, Мбит/с (0 = не проверять); медленные удаляются")
    g = p.add_argument_group("пул машин (блоки и файлы раздаются по нескольким инстансам)")
    g.add_argument("--pool-scan", "--bargain-scan", dest="pool_scan", type=float, default=15,
                   help="период проверки листинга офферов, с (0 = выкл.): лучший по DLPerf/$ берётся в пул, худшие выбывают")
    g.add_argument("--pool-improve", "--expensive-factor", "--bargain-factor", dest="pool_improve", type=float, default=1.3,
                   help="во сколько раз оффер должен быть лучше худшей машины пула по DLPerf/$, чтобы её заменить (1.3 = +30%%); "
                        "худшая помечается на выбывание и удаляется, как только освободится")
    g.add_argument("--pool-min", type=int, default=1, help="минимум готовых (онлайн, с запущенным runner-ом) машин в пуле: "
                   "худшая не выбывает, если после этого готовых останется меньше")
    g.add_argument("--pool-size", type=int, default=1, help="регулятор: сколько активных машин держать в пуле параллельно "
                   "(готовых + разворачивающихся; не больше 1+--max-workers); пока их меньше — берётся следующий лучший оффер, "
                   "если больше — худшие выбывают; в терминале меняется клавишами «+»/«-»")
    g.add_argument("--boot-deadline", type=float, default=600, help="инстанс, не прошедший установку за N с, удаляется (набор пула и воркеры)")
    g.add_argument("--interactive", action="store_true", help="меню выбора оффера и подтверждение заказа (по умолчанию — "
                   "автоматический выбор лучшего по DLPerf/$ без ожидания)")
    g.add_argument("--no-reuse-instances", action="store_true", help="не подхватывать работающие инстансы прошлых запусков "
                   "(метка «vast_upscale …» + наш образ); по умолчанию они включаются в пул без заказа")
    g.add_argument("--max-workers", type=int, default=32, help="жёсткий потолок дополнительных инстансов одновременно (регулятор "
                   "--pool-size и клавиша «+» не поднимут пул выше 1+N)")
    g.add_argument("--min-worker-minutes", type=float, default=10, help="не запускать воркер, если ему достанется меньше N минут обработки")
    g.add_argument("--no-parallel", action="store_true", help="отключить параллельные воркеры и управление пулом")
    g.add_argument("--local-upload-mbps", type=float, default=100, help="оценка вашего исходящего канала для ETA")
    g.add_argument("--local-download-mbps", type=float, default=200, help="оценка вашего входящего канала для ETA")
    g = p.add_argument_group("vast.ai / служебное")
    g.add_argument("--api-key", default=None)
    g.add_argument("--ssh-key", default=None, help="путь к приватному SSH-ключу (по умолчанию ~/.ssh/id_ed25519)")
    g.add_argument("--yes", "-y", action="store_true", help="не задавать вопросов (авто-подтверждение)")
    g.add_argument("--no-install", action="store_true", help="не предлагать apt-get install")
    g.add_argument("--check", action="store_true", help="только проверка зависимостей и сопряжение с vast.ai")
    g.add_argument("--dry-run", action="store_true", help="всё до выбора оффера включительно, без заказа")
    g.add_argument("--instance", type=int, default=0, help="использовать уже существующий инстанс с этим ID")
    g.add_argument("--resume", action="store_true", help="продолжить по state.json из рабочей папки")
    g.add_argument("--keep-instance", action="store_true", help="не удалять инстанс по окончании/при ошибке")
    g.add_argument("--workdir", default=None, help="папка этого запуска: сконвертированные видео, состояние, логи "
                   "(по умолчанию <рабочая папка скрипта>/runs/<имя>-<hash>; см. --cache-dir)")
    g.add_argument("--keep-converted", action="store_true", help="не удалять промежуточные файлы после успеха")
    g = p.add_argument_group("Блочная обработка (один файл — на несколько инстансов)")
    g.add_argument("--block-seconds", type=float, default=8.0, help="максимальная длительность чанка (блока), с видео; файл длиннее "
                   "режется на равные чанки не длиннее этого (по умолчанию 8)")
    g.add_argument("--block-frames", default="auto", help="размер блока в кадрах: auto — по --block-seconds и fps файла; N — явно "
                   "(не длиннее --block-seconds); 0 — без блоков (файл целиком)")
    g.add_argument("--block-minutes", type=float, default=8.0, help=argparse.SUPPRESS)
    g.add_argument("--block-ctx", type=int, default=8, help="кадров контекста перед блоком (обрабатываются, но в результат не идут)")
    g.add_argument("--poll-interval", type=float, default=6.0, help="период опроса статуса, с")
    g.add_argument("--boot-timeout", type=float, default=1200, help="сколько ждать запуска контейнера, с")
    g.add_argument("--resume-timeout", type=float, default=1800, help="сколько ждать возобновления interruptible-инстанса, с")
    g.add_argument("--download-timeout", type=float, default=0, help="лимит на докачку одного файла, с (0 = пока жив контейнер)")
    g.add_argument("--daemon", "-d", action="store_true", help="запустить задачу в фоне (лог и прогресс в рабочей папке; подразумевает --yes)")
    g.add_argument("--attach", "-a", action="store_true", help="подключиться к фоновому запуску (лог + живой прогресс); Ctrl+C — отсоединиться")
    g.add_argument("--stop", action="store_true", help="остановить фоновый запуск (его инстансы будут удалены)")
    g.add_argument("--status", action="store_true", help="список фоновых запусков и их прогресс")
    g.add_argument("--list-instances", action="store_true", help="показать мои инстансы и выйти")
    g.add_argument("--clear-chunks", action="store_true", help="очистить кэш очереди и все чанки (chunks/ и runs/: готовые чанки, "
                   "нарезка, конвертированные копии, состояние запусков) и выйти; кэш сборки образа и веса остаются")
    g.add_argument("--clear-cache", action="store_true", help="очистить ВСЕ кэши скрипта (рабочая папка целиком: веса, "
                   "зависимости, слои образа, чанки, запуски) и выйти")
    g.add_argument("--destroy-instance", type=int, default=0, metavar="ID", help="удалить инстанс и выйти")
    g.add_argument("--verbose", "-v", action="store_true")
    g.add_argument("--version", action="version", version=f"%(prog)s {VERSION}")
    args = p.parse_args(argv)
    if args.suffix is None:
        args.suffix = f"_{args.target}p"
    if args.batch_size and (args.batch_size - 1) % 4 != 0:
        p.error("--batch-size должен быть вида 4n+1 (5, 9, 13, 21, 33, 81 …)")
    if args.target < 240 or args.target > 4320:
        p.error("--target вне разумного диапазона (240–4320)")
    return args


def cmd_list_instances(client: VastClient) -> None:
    insts = client.show_instances()
    if not insts:
        info("Инстансов нет.")
        return
    _raw_print(bold(f"  {'ID':>9} {'GPU':<18} {'статус':<10} {'$/ч':>7} {'метка':<32} {'SSH'}"))
    for i in insts:
        eps = endpoints_of(i)
        ssh = f"ssh -p {eps[0][1]} root@{eps[0][0]}" if eps else "-"
        _raw_print(f"  {i.get('id'):>9} {str(i.get('gpu_name')):<18} {str(i.get('actual_status')):<10} "
                   f"{float(i.get('dph_total') or 0):>7.3f} {str(i.get('label') or '')[:32]:<32} {ssh}")


def cmd_clear(args) -> int:
    """--clear-chunks: chunks/ + runs/ (готовые чанки, нарезка, конвертированные копии, состояние очереди);
    --clear-cache: вся рабочая папка (плюс веса, зависимости, слои образа, сессии загрузки)."""
    root = LOCAL_CACHE_DIR
    alive = [(k, v) for k, v in _registry_load().items() if _pid_alive(int(v.get("pid") or 0))]
    if alive:
        err("Есть работающие фоновые запуски (--status): остановите их (--stop) перед очисткой кэша.")
        return 2
    if args.clear_cache:
        targets = [root] if root.exists() else []
        what = "все кэши скрипта"
    else:
        targets = [d for d in (root / "chunks", root / "runs") if d.exists()]
        what = "кэш очереди и все чанки"
        for raw in (args.inputs or []):                     # старые запуски со состоянием рядом с исходниками
            pth = Path(raw).expanduser()
            legacy = (pth if pth.is_dir() else pth.parent) / ".vast_upscale"
            if legacy.exists() and legacy not in targets:
                targets.append(legacy)
    if not targets:
        ok(f"Очищать нечего: {what} ({root})")
        return 0
    total = 0
    for t in targets:
        sz = cache_dir_size(t)
        total += sz
        info(f"{t}: {fmt_bytes(sz)}")
    if not ask_yes_no(f"Удалить {what} — {fmt_bytes(total)}?", default=False, auto=(True if args.yes else None)):
        info("Отменено.")
        return 1
    for t in targets:
        shutil.rmtree(t, ignore_errors=True)
    ok(f"Удалено: {what} ({fmt_bytes(total)})")
    return 0


def print_banner():
    _raw_print(bold(f"vast_upscale.py v{VERSION}") + dim("  — SeedVR2 на vast.ai: апскейл видео с людьми до 1080p и выше"))


def main(argv: Optional[List[str]] = None) -> int:
    global VERBOSE
    args = parse_args(argv)
    VERBOSE = args.verbose
    apply_cache_setting(args)
    if args.clear_cache or args.clear_chunks:
        print_banner()
        return cmd_clear(args)
    if args.attach:
        print_banner()
        return cmd_attach(args)
    if args.stop:
        print_banner()
        return cmd_stop(args)
    if args.status:
        print_banner()
        return cmd_status(args)
    daemon_wd: Optional[Path] = None
    if args.daemon:
        if not args.inputs:
            err("Для --daemon укажите видеофайл или директорию.")
            return 1
        daemon_wd = derive_workdir(args)
        print_banner()
        if not daemonize(daemon_wd, " ".join(shlex.quote(a) for a in (argv if argv is not None else sys.argv[1:]))):
            return 0
        args.yes = True
        args.daemon = False
    auto = True if args.yes else None
    print_banner()
    session: Optional[Session] = None
    orchestrator: Optional[Orchestrator] = None
    prebuild: Optional[Prebuild] = None
    t_start = time.time()
    exit_code = 0
    try:
        # ---- Этап 1
        stage(1, 8, "Проверка зависимостей")
        check_dependencies(auto, allow_install=not args.no_install)

        # ---- Этап 2
        stage(2, 8, "Сопряжение с vast.ai и Docker Hub")
        client, key, user = ensure_pairing(args.api_key, args.ssh_key, auto)
        credit_before = float(user.get("credit") or 0.0)
        if not (args.list_instances or args.destroy_instance):
            prebuild_possible = ensure_dockerhub(args, auto, allow_install=not args.no_install)
            big = "7b" in (args.model or "")
            est = (MODEL_FILES["seedvr2_ema_7b_fp16.safetensors" if big else "seedvr2_ema_3b_fp16.safetensors"]["size"] +
                   MODEL_FILES[VAE_FILE]["size"] + 2_000_000_000) if prebuild_possible else 0
            ensure_cache_dir(args, auto, need_bytes=est)

        if args.list_instances:
            cmd_list_instances(client)
            return 0
        if args.destroy_instance:
            info(f"Удаляю инстанс #{args.destroy_instance}…")
            client.destroy_instance(args.destroy_instance)
            ok("Запрос отправлен. Проверьте: --list-instances")
            return 0
        if args.check:
            ok("Готово: зависимости и сопряжение в порядке" + (" (предсобранный образ: " + args.prebuild + ")." if args.prebuild != "never" else "."))
            return 0
        if not args.inputs:
            raise FatalError("Укажите видеофайл или директорию (см. --help).")

        # ---- Этап 3
        stage(3, 8, "Анализ и подготовка видео")
        files = collect_inputs(args.inputs, args.recursive, args.suffix)
        if not files:
            raise FatalError("Видеофайлы не найдены.")
        workdir = derive_workdir(args)
        workdir.mkdir(parents=True, exist_ok=True)
        info(f"Найдено файлов: {len(files)}; рабочая папка: {workdir}")
        jobs = make_jobs(files, args, workdir)
        active = [j for j in jobs if j.status != "skipped"]
        if not active:
            ok("Нечего обрабатывать.")
            return 0
        for j in active:
            convert_for_upscaler(j, workdir, pre_downscale=args.pre_downscale)

        # ---- Этап 4
        stage(4, 8, "Подбор GPU-конфигурации на vast.ai")
        plan = build_plan(active, args)
        info(f"Требования к инстансу: {describe_requirements(plan.req)}")
        info(f"Кадров всего: {plan.total_frames}; загрузить: {fmt_bytes(plan.upload_bytes)}; "
             f"скачать: ~{fmt_bytes(plan.est_download_bytes)}; диск инстанса: {plan.disk_gb:.0f} ГБ; "
             f"цель: короткая сторона {args.target}px; модель: {args.model}; критерий: {plan.optimize}"
             + (f" (час ожидания = {fmt_money(plan.time_value)})" if plan.optimize == "balanced" else ""))
        session = Session(client, key, plan, args, workdir)
        session.credit_before = credit_before

        # возобновление на существующем инстансе?
        reuse_id = args.instance
        restore_prev = False
        prev = session.load_state() if (args.resume or not reuse_id) else None
        if not reuse_id and prev and prev.get("instance_id"):
            try:
                inst = client.show_instance(int(prev["instance_id"]))
            except VastAPIError:
                inst = None
            if inst and inst.get("actual_status") not in BAD_STATUSES:
                info(f"Найден прошлый инстанс #{prev['instance_id']} ({inst.get('gpu_name')}, {inst.get('actual_status')}, "
                     f"{fmt_money(float(inst.get('dph_total') or 0))}/ч)")
                if args.resume or ask_yes_no("Продолжить на нём (иначе он останется тарифицироваться отдельно)?", default=True, auto=auto):
                    reuse_id = int(prev["instance_id"])
                    restore_prev = True
        cands: List[Candidate] = []
        if reuse_id:
            inst = client.show_instance(reuse_id)
            if not inst:
                raise FatalError(f"Инстанс #{reuse_id} не найден в аккаунте.")
            session.instance_id, session.instance, session.created_at = reuse_id, inst, float(inst.get("start_date") or time.time())
            session.root = remote_root_for(reuse_id)
            vram = round((inst.get("gpu_ram") or 0) / 1024.0)
            session.model = pick_model(args.model, vram) or pick_model("3b", max(vram, 24))
            fake_offer = dict(inst)
            fake_offer.setdefault("reliability2", 1.0)
            c = estimate_candidate(fake_offer, "on-demand", plan) if inst.get("gpu_name") in GPU_PROFILES else None
            session.cand = c
            session.price_hint = float(inst.get("dph_total") or 0.0)
            ok(f"Использую инстанс #{reuse_id}: {inst.get('gpu_name')} {vram} ГБ, модель {session.model['label']}")
        else:
            sp = Spinner("Ищу офферы (on-demand и interruptible)")
            sp.tick("запрос к vast.ai")
            cands = search_candidates(client, plan, args)
            sp.close(f"Подходящих офферов: {len(cands)}")
            if not cands and not args.no_reuse_instances:
                # рынок пуст, но работают свои инстансы прошлых запусков — продолжаем на них (как на «офферах»)
                repo0 = args.image_repo or load_dockerhub_creds().get("repo") or ""
                for inst in find_our_instances(client, [args.image], repo0):
                    fake = dict(inst)
                    fake.setdefault("reliability2", 1.0)
                    fake.setdefault("min_bid", inst.get("dph_total"))
                    fake["_existing"] = True
                    c = estimate_candidate(fake, "bid" if inst.get("is_bid") else "on-demand", plan)
                    if c is not None:
                        cands.append(c)
                if cands:
                    info(f"Офферов нет, но работают {len(cands)} свой(их) инстанс(ов) прошлых запусков — продолжаю на них")
            if not cands:
                raise FatalError("Нет офферов под фильтры. Ослабьте --min-inet/--min-reliability/--min-vram, --gpu, --country или --max-price.")
            # ---- образ: проверка наличия и (при необходимости) сборка — ДО выбора и покупки рабочего инстанса
            prebuild = Prebuild(client, key, args, session.plan)
            session.image_fn = prebuild.image_for
            image_model = cands[0].model      # целевая модель: явная (--model) или по лучшему офферу при auto
            if prebuild.enabled:
                note(f"Предсобранный образ: режим {prebuild.mode}, репозиторий docker.io/{prebuild.repo}; "
                     f"целевая модель {image_model['label']}" + (" (по лучшему офферу)" if args.model == "auto" else ""))
                if args.dry_run:
                    ref = prebuild.existing_image(image_model)
                    info(f"Образ {prebuild.repo}:{prebuilt_tag(image_model['dit'], args.image, prebuild.py_ver)}: "
                         + ("есть" if ref else "нет — будет собран перед покупкой рабочего инстанса"))
                elif prebuild.mode in ("local", "strict"):
                    prebuild.ensure_image(image_model, cands)
                    # рабочий инстанс берём среди офферов с той же моделью, чтобы образ подошёл
                    same = [c for c in cands if c.model["dit"] == image_model["dit"]]
                    if same:
                        cands = same
            pool_mode = not args.no_parallel and args.pool_scan > 0 and args.max_workers > 0
            if pool_mode:
                # пул управляется по DLPerf/$ — им же и выбираем стартовую машину (таблица — в этом порядке)
                cands.sort(key=lambda c: -(offer_value(c.offer, c.price) or 0.0))
            if args.interactive and sys.stdin.isatty() and not args.yes and not args.dry_run:
                # интерактивное уточнение: номер, подробности, ещё офферы, фильтры/критерий, повторный поиск
                best, cands, plan = interactive_offer_selection(client, plan, args, cands, lambda: build_plan(active, args))
                session.plan = plan
                if prebuild.mode in ("local", "strict") and best.model["dit"] != image_model["dit"]:
                    info(f"Выбранный оффер требует другую модель ({best.model['label']}) — проверяю/собираю образ для неё перед покупкой")
                    prebuild.ensure_image(best.model, cands)
            else:
                print_candidates(cands, plan)
                best = cands[0]
                if pool_mode:
                    note("выбор автоматический, по DLPerf/$ (меню и подтверждение: --interactive)")
            _raw_print("")
            info(bold(f"Выбор: {best.gpu} {best.vram_gb:.0f} ГБ ({best.kind}, {fmt_money(best.price)}/ч), модель {best.model['label']}, "
                      f"batch {best.model['batch']}; оценка: ~{fmt_time(best.t_total)} и ~{fmt_money(best.cost_total)} за весь run"))
            note(f"из них запуск+установка ~{fmt_time(best.t_setup)}, загрузка ~{fmt_time(best.t_upload)}, "
                 f"обработка ~{fmt_time(best.t_proc)}, скачивание ~{fmt_time(best.t_download)}")
            if best.cost_total > credit_before:
                warn(f"Оценка стоимости ({fmt_money(best.cost_total)}) больше баланса ({fmt_money(credit_before)}). Пополните счёт.")
            if args.dry_run:
                ok("Режим --dry-run: заказ не выполняется.")
                return 0
            if args.interactive and sys.stdin.isatty() and not args.yes:
                if not ask_yes_no(f"Заказать {best.gpu} ({best.kind}, {fmt_money(best.price)}/ч) и начать?", default=True, auto=auto):
                    raise UserAbort()

        # ---- блочная обработка: файл режется на блоки под скорость выбранного GPU; блок — единица очереди,
        # переезда между инстансами, докачки и возобновления
        if reuse_id:
            gpu_name = (session.instance or {}).get("gpu_name", "")
            spf_est = GPU_PROFILES.get(gpu_name, {"spf": 2.5})["spf"] * (session.model or {}).get("spf_factor", 1.0)
            load_sec = (session.model or {}).get("load_sec", 60)
        else:
            spf_est = GPU_PROFILES.get(best.gpu, {"spf": 2.5})["spf"] * best.model["spf_factor"]
            load_sec = best.model["load_sec"]
        block_ctx = int(args.block_ctx)
        proc_jobs: List[Job] = []
        blocks_by_parent: Dict[str, List[Job]] = {}
        cached_total = 0
        for j in active:
            bf = block_frames_for(j, args)
            blocks = split_into_blocks(j, workdir, bf, block_ctx) if bf else []
            if blocks:
                blocks_by_parent[j.name] = blocks
                # готовые чанки из кэша прошлых запусков (те же исходник и выходные требования) — не пересчитываются
                kept = restore_cached_chunks(j, blocks, chunk_requirements(j, args, bf, block_ctx))
                cached_total += kept
                if kept:
                    info(f"{j.src.name}: из кэша чанков готово {kept} из {len(blocks)} блоков ({j.cache_dir})")
            proc_jobs.extend(blocks or [j])
        nblk = sum(1 for j in proc_jobs if j.is_block)
        assembler = Assembler(active, blocks_by_parent, workdir, args)
        session.on_downloaded = assembler.notify
        if nblk:
            session.plan = build_plan(proc_jobs, args)
            session.jobs = session.plan.jobs
            session.split_params = {"block_seconds": float(args.block_seconds), "ctx": block_ctx}
            longest = max((b.block_frames / b.info.fps for b in proc_jobs if b.is_block), default=0.0)
            info(f"Блочная обработка: {nblk} чанк(ов) не длиннее {args.block_seconds:g} с (самый длинный {longest:.1f} с, "
                 f"~{fmt_time(longest * (spf_est * max(b.info.fps for b in proc_jobs if b.is_block)))} GPU-времени), +{block_ctx} кадров контекста; "
                 f"каждый чанк скачивается сразу по готовности, видео собирается на этой машине, как только скачаны все его чанки"
                 + (f"; из кэша уже готово {cached_total}" if cached_total else ""))
        else:
            session.split_params = None
        if prev and (restore_prev or args.resume):
            # уже скачанные результаты (в т.ч. отдельные блоки) не пересчитываются даже на новом инстансе;
            # «загружен»/«обработан» имеют смысл только на прежнем инстансе
            kept = 0
            for j in session.jobs:
                pj = (prev.get("jobs") or {}).get(j.name)
                if not pj:
                    continue
                if pj.get("status") == "downloaded" and pj.get("downloaded") and Path(pj["downloaded"]).exists():
                    j.status, j.downloaded = "downloaded", Path(pj["downloaded"])
                    kept += 1
                elif restore_prev and pj.get("status") in ("uploaded", "done"):
                    j.status, j.assigned_to = pj["status"], session.tag      # лежит на прежнем инстансе — в очередь не идёт
            if kept:
                info(f"Из прошлого запуска уже скачано: {kept} из {len(session.jobs)} — пересчитываться не будут")
        need_gpu = any(j.status in ("pending", "uploaded", "done") for j in session.jobs)

        # ---- Этап 5
        stage(5, 8, "Запуск контейнера и установка SeedVR2")
        if prebuild is None:                       # путь --instance/--resume: образ не собираем, только используем готовый
            prebuild = Prebuild(client, key, args, session.plan)
            session.image_fn = prebuild.image_for
        if reuse_id and session.instance and prebuild.enabled:
            img = str(session.instance.get("image_uuid") or session.instance.get("image") or "")   # vast.ai: image_uuid
            if not img.startswith(prebuild.repo + ":") and restore_prev and prev and prev.get("prebuilt") and prev.get("image"):
                img = str(prev["image"])                          # из состояния прошлого запуска
            if img.startswith(prebuild.repo + ":"):
                session.image, session.prebuilt = img, True      # bootstrap: только проверка, без установки
        orchestrator = Orchestrator(session, client, key, args, session.plan, workdir)
        orchestrator.image_fn = prebuild.image_for
        if prebuild.enabled and cands:
            orchestrator.image_dit = cands[0].model["dit"]
            orchestrator.rebuild_fn = lambda py, _m=cands[0].model: prebuild.rebuild_for_python(_m, py)
        pool_mode = not args.no_parallel and args.pool_scan > 0 and args.max_workers > 0
        existing: List[dict] = []
        if need_gpu and not args.no_reuse_instances:
            # работающие инстансы этого скрипта с прошлых запусков (метка «vast_upscale …» + наш образ) — в пул без заказа
            existing = find_our_instances(client, [args.image], prebuild.repo if prebuild.enabled else "",
                                          exclude={int(reuse_id)} if reuse_id else set())
            if existing:
                info(f"Найдено {len(existing)} работающих инстанс(ов) прошлых запусков (метка «{LABEL_PREFIX} …», наш образ): "
                     + ", ".join(f"#{i.get('id')} {i.get('gpu_name')} {fmt_money(float(i.get('dph_total') or 0))}/ч" for i in existing)
                     + " — переиспользую (отключить: --no-reuse-instances)")
        if not need_gpu:
            ok("Все результаты уже скачаны — GPU не нужен, перехожу к сборке")
        elif not reuse_id:
            if pool_mode:
                # без ожидания: свои прошлые инстансы + лучший оффер + догон лучшими, первый готовый — основной
                orchestrator.acquire_pool(cands, existing, prebuild.repo if prebuild.enabled else "")
            elif existing or (cands and cands[0].offer.get("_existing")):
                inst = existing[0] if existing else cands[0].offer
                session.instance_id, session.instance = int(inst["id"]), inst
                session.created_at = float(inst.get("start_date") or time.time())
                session.root = remote_root_for(session.instance_id)
                vram = round((inst.get("gpu_ram") or 0) / 1024.0)
                session.model = pick_model(args.model, vram) or pick_model("3b", max(vram, 24))
                session.price_hint = float(inst.get("dph_total") or 0.0)
                img = str(inst.get("image_uuid") or inst.get("image") or "")
                if prebuild.enabled and img.startswith(prebuild.repo + ":"):
                    session.image, session.prebuilt = img, True
                ok(f"Использую инстанс #{session.instance_id} прошлого запуска: {inst.get('gpu_name')} {vram} ГБ")
                session.connect_ssh(timeout=300)
                session.check_link(strict=False)
            else:
                session.provision(cands)
        else:
            if not session.ensure_alive("подключение"):
                raise FatalError("инстанс недоступен")
            session.connect_ssh(timeout=300)
            session.check_link(strict=False)
            if existing and pool_mode:
                orchestrator.adopt_existing_workers(existing, prebuild.repo if prebuild.enabled else "")
        if need_gpu:
            if not session.booted:
                session.bootstrap()
            if prebuild.enabled and not session.prebuilt and not reuse_id and session.model:
                # образа не было (или сборщик не нашёлся) — снимаем снапшот с рабочего инстанса в фоне
                prebuild.request_snapshot(session, session.model)

        # ---- Этап 6
        stage(6, 8, "Загрузка видео и апскейл")
        keys: Optional[PoolKeys] = None
        if need_gpu:
            if orchestrator.enabled:
                note(f"Пул машин: каждые {args.pool_scan:.0f} с проверяю листинг офферов — лучший по DLPerf/$ беру в пул, если он "
                     f"в ≥{args.pool_improve:g}× лучше худшей машины пула или активных машин меньше требуемых; худшие выбывают, "
                     f"освободившись; в пуле не меньше {args.pool_min} готовых машин; до {args.max_workers} воркеров (отключить: --no-parallel)")
                note(f"Регулятор пула: требуется активных машин — {orchestrator.pool_size} (--pool-size N)"
                     + (f"; {PoolKeys.HELP}" if PoolKeys.available(args) else ""))
                if PoolKeys.available(args):
                    keys = PoolKeys(orchestrator)
                    keys.start()
            try:
                info(f"Очередь: {len(orchestrator.queue)} чанк(ов), {orchestrator.total_frames} кадров; на каждой машине заранее "
                     f"подгружается не больше чанков, чем у неё видеокарт (обычно 1): следующий уходит в обработку сразу, "
                     f"а следующий за ним подгружается из очереди")
                session.write_jobs_and_start_runner()      # runner ждёт заданий; чанки подгружаются из очереди по одному
                session.monitor_and_download()
                orchestrator.wait_workers()
            finally:
                if keys:
                    keys.stop()

        # ---- Этап 7
        stage(7, 8, "Сборка результатов")
        assembler.wait()                           # видео из чанков собираются сразу по готовности — дождаться последних
        finished = []
        by_name = {j.name: j for j in session.jobs}
        for j in active:
            if j.status == "finished":             # собрано сборщиком по готовности чанков
                finished.append(j)
                note(f"{j.out_path}: собран ранее, сразу по готовности чанков")
                continue
            if j.status == "split":
                blocks = [by_name[n] for n in j.blocks if n in by_name]
                bad = [b for b in blocks if b.status == "failed"]
                if bad or len(blocks) != len(j.blocks):
                    j.status, j.error = "failed", "; ".join(f"блок {b.block_idx + 1}: {b.error[:120]}" for b in bad) or "не все блоки обработаны"
                    continue
                if all(b.status == "downloaded" and b.downloaded for b in blocks):
                    try:
                        j.downloaded = assemble_blocks(j, blocks, workdir)
                        j.status = "downloaded"
                        ok(f"{j.src.name}: {len(blocks)} блоков склеены без перекодирования")
                    except FatalError as e:
                        j.status, j.error = "failed", str(e)
                        err(str(e))
                        continue
                else:
                    j.status, j.error = "failed", "не все блоки скачаны"
                    continue
            if j.status == "downloaded" and j.downloaded:
                try:
                    out = mux_audio(j, f"{args.target}p")
                    j.status = "finished"
                    pi = ffprobe(out)
                    finished.append(j)
                    ok(f"{out}  ({pi.width}x{pi.height}, {fmt_bytes(pi.size_bytes)}{', со звуком' if j.info.has_audio else ''})")
                    if j.blocks:
                        chunk_cache_clear(j)           # полный файл собран — чанки этого видео из кэша удаляются
                    if not args.keep_converted and j.downloaded:
                        try:
                            j.downloaded.unlink()
                        except OSError:
                            pass
                except FatalError as e:
                    j.status, j.error = "failed", str(e)
                    err(str(e))
        session.save_state()
        failed = [j for j in active if j.status == "failed"]
        if failed:
            for j in failed:
                err(f"{j.src.name}: {j.error[:300]}")
        # нарезка на чанки и CFR-копии живут в рабочей папке до полного завершения всей очереди (переживают перезапуск:
        # следующий запуск переиспользует их); удаляются только когда обработано всё
        if failed or not all(j.status == "finished" for j in active):
            note(f"Нарезка и конвертированные копии сохранены в {workdir} — следующий запуск продолжит с них")
        elif not args.keep_converted:
            n_rm = 0
            for j in active:
                for b in blocks_by_parent.get(j.name, []):
                    try:
                        if b.converted and b.converted.exists():
                            b.converted.unlink()
                            n_rm += 1
                    except OSError:
                        pass
                try:
                    if j.converted and j.converted.exists():
                        j.converted.unlink()
                        n_rm += 1
                except OSError:
                    pass
            if n_rm:
                note(f"Вся очередь обработана — удалены промежуточные файлы ({n_rm}) из {workdir}")

        # ---- Этап 8
        stage(8, 8, "Остановка и удаление контейнера")
        if orchestrator:
            for t in list(orchestrator.threads.values()):
                t.join(timeout=120)
            orchestrator.destroy_all()
        if prebuild and prebuild.snapshot_pending:
            prebuild.finish_pending()
        all_ok = not failed and len(finished) == len(active)
        if args.keep_instance:
            warn(f"--keep-instance: инстанс #{session.instance_id} оставлен и продолжает тарифицироваться "
                 f"({fmt_money(session.cand.price if session.cand else 0)}/ч). Удалить: --destroy-instance {session.instance_id}")
        elif all_ok or ask_yes_no("Не все файлы обработаны. Всё равно удалить инстанс?", default=True, auto=auto):
            session.destroy()
        # итог
        hours = (((session.destroyed_at or time.time()) - (session.created_at or t_start)) / 3600) if need_gpu else 0.0
        price = session.cand.price if session.cand else session.price_hint
        if orchestrator and orchestrator.workers:
            wh, wc = orchestrator.rent_cost()
            note(f"Воркеры: {len(orchestrator.workers)} шт., ≈ {fmt_money(wc)} за {fmt_time(wh * 3600)} аренды")
            price_extra = wc
        else:
            price_extra = 0.0
        try:
            credit_after = float(client.show_user().get("credit") or 0.0)
            spent = f"по балансу: {fmt_money(credit_before - credit_after)} (может обновляться с задержкой)"
        except VastAPIError:
            spent = ""
        _raw_print("")
        ok(bold(f"Готово: {len(finished)} из {len(active)} файл(ов) за {fmt_time(time.time() - t_start)}; "
                f"аренда ≈ {fmt_money(price * hours + price_extra)} (основной {fmt_time(hours * 3600)} × {fmt_money(price)}/ч"
                f"{' + воркеры' if price_extra else ''}){'; ' + spent if spent else ''}"))
        exit_code = 0 if all_ok else 2
    except UserAbort:
        _raw_print("")
        warn("Остановлено пользователем.")
        exit_code = 130
    except KeyboardInterrupt:
        _raw_print("")
        warn("Прервано (Ctrl+C).")
        exit_code = 130
    except FatalError as e:
        _raw_print("")
        err(str(e))
        exit_code = 1
    except Exception as e:                      # неожиданная ошибка: полный traceback — в crash.log, пользователю — суть
        import traceback
        _raw_print("")
        crash = None
        try:
            LOCAL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            crash = LOCAL_CACHE_DIR / "crash.log"
            with open(crash, "a", encoding="utf-8") as f:
                f.write(f"=== {_dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')} v{VERSION} {' '.join(sys.argv[1:])}\n")
                f.write(traceback.format_exc() + "\n")
        except OSError:
            crash = None
        err(f"Внутренняя ошибка: {type(e).__name__}: {e}" + (f" — подробности в {crash}" if crash else ""))
        if VERBOSE:
            traceback.print_exc()
        exit_code = 1
    finally:
        if orchestrator and orchestrator.workers:
            alive = orchestrator.active_workers()
            if alive:
                warn(f"Останавливаю {len(alive)} параллельных воркер(ов)…")
            orchestrator.destroy_all()
        if session and session.instance_id:
            iid = session.instance_id
            if args.keep_instance:
                warn(f"Инстанс #{iid} оставлен (--keep-instance) и тарифицируется. Удалить: --destroy-instance {iid}; "
                     f"продолжить: --resume")
            else:
                do_destroy = True if args.yes else ask_with_timeout(
                    f"Удалить инстанс #{iid}, чтобы не платить дальше? (нет = оставить для --resume)", default=True, timeout=45)
                if do_destroy:
                    try:
                        session.destroy()
                    except Exception as e:  # noqa
                        err(f"Не удалось удалить инстанс #{iid}: {e}. Удалите вручную: https://cloud.vast.ai/instances/")
                else:
                    warn(f"Инстанс #{iid} оставлен. Продолжить: --resume; удалить: --destroy-instance {iid}")
        if session and session.ssh:
            session.ssh.close()
        if daemon_wd:
            daemon_finish(daemon_wd, exit_code)
            try:
                sys.stdout.flush(); sys.stderr.flush()
            except Exception:  # noqa
                pass
            os._exit(exit_code)   # фоновый потомок никогда не возвращается в вызывающий код
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
