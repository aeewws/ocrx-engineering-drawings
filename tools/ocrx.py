from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import fnmatch
import json
import importlib
import logging
import os
import queue
import re
import shutil
import subprocess
import sys
import tempfile
import threading
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Iterator

import cv2
import fitz
import numpy as np
import onnxruntime as ort
from PIL import Image
from rapid_latex_ocr import LaTeXOCR
from rapidocr_onnxruntime import RapidOCR


GENERAL_ENGINE: RapidOCR | None = None
MATH_ENGINE: LaTeXOCR | None = None
PADDLE_GENERAL_ENGINE: Any | None = None
PADDLE_MODULE: Any | None = None
PADDLE_OCR_CLASS: Any | None = None
ORT_PREPARED = False
ORT_GPU_READY = False
MATH_ORT_PATCHED = False
PADDLE_PREPARED = False
PADDLE_IMPORT_ERROR: Exception | None = None

DRAWING_SOURCE_ORDER = [
    "full",
    "full.line_free",
    "full.rot90",
    "full.rot270",
    "bottom_right",
    "bottom_right.line_free",
    "bottom_left",
    "bottom_left.line_free",
    "top_right",
    "top_right.line_free",
]
DRAWING_REGION_SPECS: list[tuple[str, tuple[float, float, float, float]]] = [
    ("full", (0.0, 0.0, 1.0, 1.0)),
    ("bottom_right", (0.54, 0.64, 1.0, 1.0)),
    ("bottom_left", (0.0, 0.70, 0.46, 1.0)),
    ("top_right", (0.60, 0.0, 1.0, 0.28)),
]
TITLE_BLOCK_KEYWORDS: list[tuple[str, tuple[str, ...]]] = [
    ("发包人", ("发包人", "建设单位", "业主")),
    ("监理人", ("监理人", "监理单位")),
    ("工程名称", ("工程名称", "工程名", "项目名称", "项目名", "工程", "项目")),
    ("图名", ("图名", "标题", "监理合同", "施工图", "平面图", "剖面图")),
    ("图号", ("图号", "编号", "图纸编号", "drawingno")),
    ("比例", ("比例", "scale")),
    ("日期", ("日期", "date")),
    ("设计", ("设计", "designedby", "designer")),
    ("校核", ("校核", "checkedby", "checker")),
    ("审核", ("审核", "审定", "approvedby", "reviewer", "approver")),
    ("页码", ("页码", "第页", "page")),
]
OCR_ENGINE_CHOICES = ("auto", "rapid", "paddle")
DRAWING_DETAIL_CHOICES = ("auto", "drawing", "full", "rebar")
REBAR_TEXT_PATTERN = re.compile(
    r"(%%132|[Φφ@]|HRB\s*\d{3,4}|\d+\s*[@xX×]\s*\d+)",
    re.IGNORECASE,
)
ENGINEERING_DETAIL_PATTERN = re.compile(
    r"(%%132|[Φφ@#]|HRB\s*\d{3,4}|\b[NCG]\d{1,3}(?:-\d+)?\b|\b[A-Za-z]{1,4}-?\d{1,4}\b|"
    r"\d+\s*[@xX×/]\s*\d+|\b[Kk]\d+\+\d+(?:\.\d+)?\b|\b[Rr]\d+(?:\.\d+)?\b|\d+(?:\.\d+)?\s*(?:mm|cm|m)\b)",
    re.IGNORECASE,
)
HQ_SOURCE_PREFIXES = ("hq.title_block", "hq.table", "hq.detail", "hq.note", "hq.rebar")
TITLE_BLOCK_REQUIRED_FIELDS = ("工程名称", "图号", "比例")
DRAWING_NOTE_KEYWORDS = ("说明", "备注", "做法", "要求", "技术", "材料", "坐标", "高程")
DRAWING_DETAIL_KEYWORDS = ("详图", "大样", "节点", "剖面", "断面", "尺寸", "标高", "轴线", "配筋", "钢筋")
STRONG_DRAWING_DETAIL_KEYWORDS = ("详图", "大样", "节点", "装配", "配筋", "钢筋", "明细")
PDF_NATIVE_VECTOR_CHAR_THRESHOLD = 80
PDF_NATIVE_VECTOR_WORD_THRESHOLD = 20
PDF_NATIVE_SAMPLE_LIMIT = 6
CAD_SUFFIXES = (".dwg", ".dxf")
CAD_DEFAULT_DPI = 320
CAD_DEFAULT_MIN_WIDTH = 2600
CAD_RENDER_MAX_SIDE_PX = 5200
CAD_RENDER_MIN_SIDE_INCHES = 3.0
CAD_RENDER_BG = "#FFFFFF"
CAD_RENDER_FG = "#000000"


@dataclass
class OCRLine:
    text: str
    score: float
    box: list[list[float]]
    page: int
    source: str = "full"


@dataclass
class RenderedPage:
    page: int
    image: np.ndarray
    path: Path | None = None


@dataclass
class PreparedDrawingPage:
    page: int
    base_regions: dict[str, np.ndarray]
    line_free_regions: dict[str, np.ndarray] | None = None
    rotated_fulls: dict[str, np.ndarray] | None = None
    path: Path | None = None


@dataclass
class NativePDFPage:
    page: int
    text_class: str
    chars: int
    words: int
    images: int
    lines: list[OCRLine]


def prepare_onnxruntime() -> None:
    global ORT_PREPARED, ORT_GPU_READY
    if ORT_PREPARED:
        return

    site_packages = Path(ort.__file__).resolve().parent.parent
    add_nvidia_library_dirs(site_packages)

    if hasattr(ort, "preload_dlls"):
        try:
            ort.preload_dlls(directory="")
        except TypeError:
            ort.preload_dlls()
        except Exception:
            pass

    try:
        ORT_GPU_READY = str(ort.get_device()).upper() == "GPU"
    except Exception:
        providers = ort.get_available_providers()
        ORT_GPU_READY = "CUDAExecutionProvider" in providers
    ORT_PREPARED = True


def iter_nvidia_library_dirs(site_packages: Path) -> list[Path]:
    nvidia_root = site_packages / "nvidia"
    if not nvidia_root.exists():
        return []

    dirs: set[Path] = set()
    for path in nvidia_root.rglob("*"):
        if not path.is_dir():
            continue
        if path.name == "bin":
            dirs.add(path)
        elif path.name.lower() == "x86_64" and path.parent.name == "bin":
            dirs.add(path)
    return sorted(dirs)


def add_nvidia_library_dirs(site_packages: Path) -> None:
    for bin_dir in iter_nvidia_library_dirs(site_packages):
        try:
            os.add_dll_directory(str(bin_dir))
        except (AttributeError, FileNotFoundError, OSError):
            pass
        path_value = os.environ.get("PATH", "")
        bin_dir_str = str(bin_dir)
        if bin_dir_str.lower() not in path_value.lower():
            os.environ["PATH"] = f"{bin_dir_str}{os.pathsep}{path_value}"


def prepare_paddle_runtime() -> None:
    global PADDLE_PREPARED, PADDLE_MODULE, PADDLE_OCR_CLASS, PADDLE_IMPORT_ERROR
    if PADDLE_PREPARED:
        return

    site_packages = Path(ort.__file__).resolve().parent.parent
    add_nvidia_library_dirs(site_packages)

    try:
        PADDLE_MODULE = importlib.import_module("paddle")
        paddleocr_module = importlib.import_module("paddleocr")
        PADDLE_OCR_CLASS = getattr(paddleocr_module, "PaddleOCR", None)
        PADDLE_IMPORT_ERROR = None
    except Exception as exc:
        PADDLE_MODULE = None
        PADDLE_OCR_CLASS = None
        PADDLE_IMPORT_ERROR = exc
    finally:
        PADDLE_PREPARED = True


def gpu_provider_options() -> list[Any]:
    prepare_onnxruntime()
    providers: list[Any] = ["CPUExecutionProvider"]
    if ORT_GPU_READY:
        providers.insert(
            0,
            (
                "CUDAExecutionProvider",
                {
                    "device_id": 0,
                    "arena_extend_strategy": "kNextPowerOfTwo",
                    "cudnn_conv_algo_search": "EXHAUSTIVE",
                    "do_copy_in_default_stream": True,
                },
            ),
        )
    return providers


def current_python_scripts_dir() -> Path | None:
    python_path = Path(sys.executable).resolve()
    parent = python_path.parent
    if parent.name.lower() == "scripts":
        return parent
    scripts_dir = parent / "Scripts"
    return scripts_dir if scripts_dir.exists() else None


def resolve_env_executable(name: str) -> str | None:
    scripts_dir = current_python_scripts_dir()
    if scripts_dir is None:
        return None
    for candidate in (scripts_dir / f"{name}.exe", scripts_dir / f"{name}.cmd", scripts_dir / name):
        if candidate.exists():
            return str(candidate)
    return None


def resolve_tesseract_executable() -> str | None:
    exe = shutil.which("tesseract")
    if exe:
        return exe
    if os.name == "nt":
        for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
            root = os.environ.get(env_name)
            if not root:
                continue
            candidate = Path(root) / "Tesseract-OCR" / "tesseract.exe"
            if candidate.exists():
                return str(candidate.resolve())
    return None


def resolve_ocrmypdf_executable() -> str | None:
    env_exe = resolve_env_executable("ocrmypdf")
    if env_exe:
        return env_exe
    return shutil.which("ocrmypdf")


def subprocess_search_path(extra_dirs: Iterable[Path | str]) -> str:
    parts: list[str] = []
    seen: set[str] = set()
    for entry in extra_dirs:
        text = str(entry)
        if not text:
            continue
        lowered = text.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        parts.append(text)
    existing = os.environ.get("PATH", "")
    if existing:
        parts.append(existing)
    return os.pathsep.join(parts)


def patch_math_onnx_sessions() -> None:
    global MATH_ORT_PATCHED
    if MATH_ORT_PATCHED:
        return

    prepare_onnxruntime()

    import rapid_latex_ocr.main as latex_main
    import rapid_latex_ocr.models as latex_models

    class GPUOrtInferSession:
        def __init__(self, model_path: str | Path, num_threads: int = -1):
            self.num_threads = num_threads
            self.sess_opt = ort.SessionOptions()
            self.sess_opt.log_severity_level = 4
            self.sess_opt.enable_cpu_mem_arena = False
            if self.num_threads != -1:
                self.sess_opt.intra_op_num_threads = self.num_threads
            self.sess_opt.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            self.session = ort.InferenceSession(
                str(model_path),
                sess_options=self.sess_opt,
                providers=gpu_provider_options(),
            )

        def __call__(self, input_content: list[np.ndarray]) -> np.ndarray:
            input_dict = dict(zip(self.get_input_names(), input_content))
            return self.session.run(None, input_dict)

        def get_input_names(self) -> list[str]:
            return [v.name for v in self.session.get_inputs()]

        def get_output_name(self, output_idx: int = 0) -> str:
            return self.session.get_outputs()[output_idx].name

        def get_metadata(self) -> dict[str, Any]:
            return self.session.get_modelmeta().custom_metadata_map

        def get_providers(self) -> list[str]:
            return self.session.get_providers()

    latex_main.OrtInferSession = GPUOrtInferSession
    latex_models.OrtInferSession = GPUOrtInferSession
    MATH_ORT_PATCHED = True


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="ocrx",
        description="Global OCR hub for images, screenshots, PDFs, formulas, drawings, and CAD files.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("list", help="List OCR modes")
    subparsers.add_parser("doctor", help="Show OCR environment health")

    add_ocr_subcommand(
        subparsers,
        name="auto",
        help_text="Run auto OCR",
        default_profile="general",
        default_dpi=220,
        default_min_width=1400,
    )
    add_ocr_subcommand(
        subparsers,
        name="general",
        help_text="Run general OCR",
        default_profile="general",
        default_dpi=220,
        default_min_width=1400,
    )
    add_ocr_subcommand(
        subparsers,
        name="math",
        help_text="Run math OCR",
        default_profile="general",
        default_dpi=220,
        default_min_width=1400,
    )
    add_ocr_subcommand(
        subparsers,
        name="pdf",
        help_text="Run PDF OCR",
        default_profile="general",
        default_dpi=220,
        default_min_width=1400,
    )
    add_ocr_subcommand(
        subparsers,
        name="drawing",
        help_text="Run engineering drawing OCR",
        default_profile="drawing",
        default_dpi=300,
        default_min_width=2200,
    )
    add_ocr_subcommand(
        subparsers,
        name="cad",
        help_text="Run CAD extraction with OCR fallback",
        default_profile="drawing",
        default_dpi=CAD_DEFAULT_DPI,
        default_min_width=CAD_DEFAULT_MIN_WIDTH,
    )
    batch = subparsers.add_parser("batch", help="Batch OCR a directory")
    batch.add_argument("input", help="Input directory")
    batch.add_argument(
        "--mode",
        choices=["auto", "general", "drawing", "cad", "pdf", "math"],
        default="auto",
        help="OCR mode for matched files",
    )
    batch.add_argument(
        "-o",
        "--output-dir",
        default="ocr_output",
        help="Batch output directory, default: ./ocr_output",
    )
    batch.add_argument(
        "--profile",
        choices=["general", "scan", "drawing"],
        default="general",
        help="Preprocess profile for OCR",
    )
    batch.add_argument("--dpi", type=int, default=300, help="PDF render DPI")
    batch.add_argument(
        "--min-width",
        type=int,
        default=2200,
        help="Upscale images narrower than this width before OCR",
    )
    batch.add_argument(
        "--keep-pages",
        action="store_true",
        help="Keep rendered PDF pages under the output folder for debugging",
    )
    batch.add_argument(
        "--searchable-pdf",
        action="store_true",
        help="Also produce searchable PDFs for PDF inputs",
    )
    batch.add_argument(
        "--ocrmypdf-lang",
        default="chi_sim+eng",
        help="Language string for OCRmyPDF/Tesseract",
    )
    batch.add_argument(
        "--engine",
        choices=OCR_ENGINE_CHOICES,
        default="auto",
        help="OCR backend: auto prefers Paddle when its GPU backend is ready, otherwise RapidOCR",
    )
    batch.add_argument(
        "--recursive",
        action="store_true",
        help="Recurse into subdirectories",
    )
    batch.add_argument(
        "--glob",
        action="append",
        default=[],
        help="Optional filename glob, can be repeated",
    )

    return parser.parse_args()


def add_ocr_subcommand(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    name: str,
    help_text: str,
    default_profile: str,
    default_dpi: int,
    default_min_width: int,
) -> None:
    sub = subparsers.add_parser(name, help=help_text)
    sub.add_argument("input", help="Input path")
    sub.add_argument(
        "-o",
        "--output-dir",
        default="ocr_output",
        help="Output directory, default: ./ocr_output",
    )
    sub.add_argument(
        "--profile",
        choices=["general", "scan", "drawing"],
        default=default_profile,
        help="Preprocess profile for OCR",
    )
    sub.add_argument("--dpi", type=int, default=default_dpi, help="PDF render DPI")
    sub.add_argument(
        "--min-width",
        type=int,
        default=default_min_width,
        help="Upscale images narrower than this width before OCR",
    )
    sub.add_argument(
        "--keep-pages",
        action="store_true",
        help="Keep rendered PDF pages under the output folder for debugging",
    )
    sub.add_argument(
        "--searchable-pdf",
        action="store_true",
        help="Also produce a searchable PDF when input is a PDF and OCRmyPDF exists",
    )
    sub.add_argument(
        "--ocrmypdf-lang",
        default="chi_sim+eng",
        help="Language string for OCRmyPDF/Tesseract",
    )
    sub.add_argument(
        "--engine",
        choices=OCR_ENGINE_CHOICES,
        default="auto",
        help="OCR backend: auto prefers Paddle when its GPU backend is ready, otherwise RapidOCR",
    )
    sub.add_argument(
        "--detail",
        choices=DRAWING_DETAIL_CHOICES,
        default="auto",
        help="Drawing detail mode; auto uses the smart drawing flow, full enables heavier rescans",
    )


def get_general_engine() -> RapidOCR:
    global GENERAL_ENGINE
    if GENERAL_ENGINE is None:
        prepare_onnxruntime()
        GENERAL_ENGINE = RapidOCR(
            det_use_cuda=ORT_GPU_READY,
            cls_use_cuda=ORT_GPU_READY,
            rec_use_cuda=ORT_GPU_READY,
        )
    return GENERAL_ENGINE


def paddle_available() -> bool:
    prepare_paddle_runtime()
    return PADDLE_OCR_CLASS is not None and PADDLE_MODULE is not None


def paddle_gpu_ready() -> bool:
    if not paddle_available():
        return False
    try:
        return bool(PADDLE_MODULE.is_compiled_with_cuda())
    except Exception:
        return False


def resolve_ocr_engine(requested: str) -> str:
    if requested == "rapid":
        return "rapid"
    if requested == "paddle":
        if not paddle_available():
            raise RuntimeError(f"PaddleOCR unavailable: {PADDLE_IMPORT_ERROR}")
        return "paddle"
    auto_pref = os.environ.get("OCRX_AUTO_ENGINE", "").strip().lower()
    if auto_pref == "paddle":
        if not paddle_available():
            raise RuntimeError(f"PaddleOCR unavailable: {PADDLE_IMPORT_ERROR}")
        return "paddle"
    if auto_pref == "rapid":
        return "rapid"
    # Prefer Paddle when its GPU backend is ready on this workstation, but still allow
    # an explicit rapid override via `OCRX_AUTO_ENGINE=rapid`.
    if paddle_available() and paddle_gpu_ready():
        return "paddle"
    return "rapid"


def get_paddle_engine() -> Any:
    global PADDLE_GENERAL_ENGINE
    if not paddle_available():
        raise RuntimeError(f"PaddleOCR unavailable: {PADDLE_IMPORT_ERROR}")
    if PADDLE_GENERAL_ENGINE is None:
        device = "gpu:0" if paddle_gpu_ready() else "cpu"
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")
        # Drawing-heavy PDFs run better on this workstation when we keep the
        # server det/rec models but skip the extra orientation/unwarping passes.
        PADDLE_GENERAL_ENGINE = PADDLE_OCR_CLASS(
            device=device,
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
            text_recognition_batch_size=get_paddle_rec_batch_size(),
        )
    return PADDLE_GENERAL_ENGINE


def env_positive_int(name: str, default: int) -> int:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def default_drawing_prepare_workers() -> int:
    cpu_count = os.cpu_count() or 4
    if cpu_count >= 16:
        return 3
    if cpu_count >= 8:
        return 2
    return 1


def get_paddle_rec_batch_size() -> int:
    return env_positive_int("OCRX_PADDLE_REC_BATCH_SIZE", 16)


def get_drawing_prepare_workers() -> int:
    return env_positive_int("OCRX_DRAWING_PREPARE_WORKERS", default_drawing_prepare_workers())


def get_drawing_queue_size(prepare_workers: int | None = None) -> int:
    workers = prepare_workers if prepare_workers is not None else get_drawing_prepare_workers()
    return env_positive_int("OCRX_DRAWING_QUEUE_SIZE", max(3, workers * 2))


def get_math_engine() -> LaTeXOCR:
    global MATH_ENGINE
    if MATH_ENGINE is None:
        patch_math_onnx_sessions()
        MATH_ENGINE = LaTeXOCR()
    return MATH_ENGINE


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def preprocess_image(image: np.ndarray, profile: str, min_width: int) -> np.ndarray:
    if image is None:
        raise ValueError("Unable to read image")

    height, width = image.shape[:2]
    if width < min_width:
        scale = min_width / max(width, 1)
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_CUBIC,
        )

    if profile == "general":
        return image

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    if profile == "scan":
        gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if profile == "drawing":
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        gray = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
        gray = cv2.medianBlur(gray, 3)
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    return image


def sort_lines(lines: Iterable[OCRLine]) -> list[OCRLine]:
    def key_fn(item: OCRLine) -> tuple[float, float]:
        xs = [pt[0] for pt in item.box]
        ys = [pt[1] for pt in item.box]
        return (sum(ys) / len(ys), sum(xs) / len(xs))

    return sorted(lines, key=key_fn)


def read_image(path: Path) -> np.ndarray:
    return cv2.imdecode(np.fromfile(str(path), dtype=np.uint8), cv2.IMREAD_COLOR)


def ensure_bgr_image(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.ndim == 3 and image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def pixmap_to_bgr(pix: fitz.Pixmap) -> np.ndarray:
    channels = pix.n
    image = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width, channels)
    if channels == 1:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if channels == 3:
        return cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    if channels == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2BGR)
    raise ValueError(f"Unsupported pixmap channel count: {channels}")


def normalize_math_result(result: Any) -> tuple[str, float | None]:
    if isinstance(result, tuple) and result:
        text = str(result[0]).strip()
        score = float(result[1]) if len(result) > 1 else None
        return text, score
    return str(result).strip(), None


def tesseract_ocr_text(
    image_path: Path,
    psm: str,
    whitelist: str,
) -> str:
    exe = resolve_tesseract_executable()
    if not exe:
        return ""
    cmd = [
        exe,
        str(image_path),
        "stdout",
        "--psm",
        psm,
        "-l",
        "eng",
        "-c",
        f"tessedit_char_whitelist={whitelist}",
    ]
    try:
        env = os.environ.copy()
        env["PATH"] = subprocess_search_path([Path(exe).parent])
        result = subprocess.run(cmd, capture_output=True, text=True, check=False, env=env)
        return result.stdout.strip()
    except Exception:
        return ""


def normalize_simple_token(text: str) -> str:
    mapping = {
        "l": "1",
        "I": "1",
        "i": "1",
        "|": "1",
        "o": "0",
        "O": "0",
        "s": "5",
        "S": "5",
        "b": "6",
    }
    cleaned = "".join(ch for ch in text if ch.isalnum() or ch in "|")
    return "".join(mapping.get(ch, ch) for ch in cleaned)


def crop_to_content(gray: np.ndarray) -> np.ndarray:
    mask = gray < 220
    coords = np.argwhere(mask)
    if coords.size == 0:
        return gray
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return gray[max(0, y0 - 8) : y1 + 8, max(0, x0 - 8) : x1 + 8]


def simple_fraction_fallback(input_path: Path) -> str:
    image = Image.open(input_path).convert("L")
    width, height = image.size
    if not (height >= width * 1.8 and width <= 40):
        return ""

    image = image.resize((max(1, width * 30), max(1, height * 30)))
    upscaled = np.array(image)
    mask = upscaled < 180
    projection = mask.sum(axis=1)
    threshold = projection.max() * 0.7 if projection.max() > 0 else 0
    rows = [idx for idx, value in enumerate(projection) if value >= threshold]
    if not rows:
        return ""
    center = rows[len(rows) // 2]
    margin = 30
    top = Image.fromarray(crop_to_content(upscaled[: max(1, center - margin), :]))
    bottom = Image.fromarray(
        crop_to_content(upscaled[min(upscaled.shape[0], center + margin) :, :])
    )

    temp_root = input_path.parent
    top_path = temp_root / f"{input_path.stem}._ocr_top.png"
    bottom_path = temp_root / f"{input_path.stem}._ocr_bottom.png"
    top.save(top_path)
    bottom.save(bottom_path)
    try:
        numerator = normalize_simple_token(
            tesseract_ocr_text(top_path, psm="10", whitelist="0123456789lIi|oOsSb")
        )
        denominator = normalize_simple_token(
            tesseract_ocr_text(bottom_path, psm="10", whitelist="0123456789lIi|oOsSb")
        )
    finally:
        top_path.unlink(missing_ok=True)
        bottom_path.unlink(missing_ok=True)

    if numerator and denominator:
        return f"{numerator}/{denominator}"
    return ""


def run_math_ocr(input_path: Path) -> tuple[str, float | None]:
    fraction_text = simple_fraction_fallback(input_path)
    if fraction_text:
        return fraction_text, None
    raw = get_math_engine()(str(input_path))
    return normalize_math_result(raw)


def run_rapidocr_on_array(image: np.ndarray, page: int, source: str) -> list[OCRLine]:
    result, _ = get_general_engine()(image)
    if not result:
        return []

    lines: list[OCRLine] = []
    for box, text, score in result:
        cleaned = " ".join(text.split()).strip()
        if not cleaned:
            continue
        lines.append(
            OCRLine(
                text=cleaned,
                score=float(score),
                box=[[float(x), float(y)] for x, y in box],
                page=page,
                source=source,
            )
        )
    return sort_lines(lines)


def result_field(result: Any, key: str, default: Any) -> Any:
    try:
        return result[key]
    except Exception:
        return getattr(result, key, default)


def run_paddleocr_on_array(image: np.ndarray, page: int, source: str) -> list[OCRLine]:
    results = list(get_paddle_engine().predict(ensure_bgr_image(image), text_rec_score_thresh=0.0))
    if not results:
        return []

    lines: list[OCRLine] = []
    for result in results:
        texts = result_field(result, "rec_texts", []) or []
        scores = result_field(result, "rec_scores", []) or []
        polys = result_field(result, "rec_polys", []) or result_field(result, "dt_polys", []) or []
        for text, score, poly in zip(texts, scores, polys):
            cleaned = " ".join(str(text).split()).strip()
            if not cleaned:
                continue
            box = np.asarray(poly).tolist()
            lines.append(
                OCRLine(
                    text=cleaned,
                    score=float(score),
                    box=[[float(x), float(y)] for x, y in box],
                    page=page,
                    source=source,
                )
            )
    return sort_lines(lines)


def run_text_ocr_on_array(
    image: np.ndarray,
    page: int,
    source: str,
    engine: str,
) -> list[OCRLine]:
    if engine == "paddle":
        return run_paddleocr_on_array(image, page=page, source=source)
    return run_rapidocr_on_array(image, page=page, source=source)


def run_general_ocr_on_image(
    image_path: Path,
    page: int,
    profile: str,
    min_width: int,
    engine: str,
) -> list[OCRLine]:
    return run_general_ocr_on_array(
        read_image(image_path),
        page=page,
        profile=profile,
        min_width=min_width,
        engine=engine,
    )


def run_general_ocr_on_array(
    image: np.ndarray,
    page: int,
    profile: str,
    min_width: int,
    engine: str,
) -> list[OCRLine]:
    prepared = preprocess_image(ensure_bgr_image(image), profile=profile, min_width=min_width)
    return run_text_ocr_on_array(prepared, page=page, source="full", engine=engine)


def remove_structural_lines(gray: np.ndarray) -> np.ndarray:
    binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
    )[1]
    height, width = gray.shape[:2]
    horizontal_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (max(24, width // 28), 1),
    )
    vertical_kernel = cv2.getStructuringElement(
        cv2.MORPH_RECT,
        (1, max(24, height // 28)),
    )
    horizontal = cv2.morphologyEx(binary, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(binary, cv2.MORPH_OPEN, vertical_kernel)
    mask = cv2.bitwise_or(horizontal, vertical)
    cleaned = cv2.bitwise_and(binary, cv2.bitwise_not(mask))
    cleaned = cv2.medianBlur(cleaned, 3)
    output = 255 - cleaned
    return cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)


def rotate_image(image: np.ndarray, angle: int) -> np.ndarray:
    if angle == 90:
        return cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    if angle == 180:
        return cv2.rotate(image, cv2.ROTATE_180)
    if angle == 270:
        return cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return image.copy()


def average_score(lines: list[OCRLine]) -> float:
    if not lines:
        return 0.0
    return sum(line.score for line in lines) / len(lines)


def normalize_text_key(text: str) -> str:
    return "".join(ch.lower() for ch in text if not ch.isspace())


def is_rebar_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    return REBAR_TEXT_PATTERN.search(stripped) is not None


def is_engineering_detail_like(text: str) -> bool:
    stripped = text.strip()
    if not stripped:
        return False
    if ENGINEERING_DETAIL_PATTERN.search(stripped):
        return True
    normalized = normalize_text_key(stripped)
    if not normalized:
        return False
    digit_count = sum(ch.isdigit() for ch in normalized)
    alpha_count = sum(ch.isalpha() for ch in normalized)
    symbol_count = sum(ch in "@#=:/+-xX×()[]{}.%Φφ" for ch in stripped)
    if len(normalized) <= 16 and digit_count >= 2 and (alpha_count >= 1 or symbol_count >= 1):
        return True
    return any(keyword in stripped for keyword in DRAWING_DETAIL_KEYWORDS)


def has_cjk(text: str) -> bool:
    return any("\u4e00" <= ch <= "\u9fff" for ch in text)


def has_meaningful_char(text: str) -> bool:
    return any(ch.isalnum() or "\u4e00" <= ch <= "\u9fff" for ch in text)


def is_noise_text(text: str, score: float) -> bool:
    stripped = text.strip()
    if not stripped:
        return True
    if not has_meaningful_char(stripped):
        return True
    normalized = normalize_text_key(stripped)
    if normalized in {"+", "-", "_", "|", "/", "\\", ".", ",", ";", ":"}:
        return True
    if len(normalized) <= 1 and not has_cjk(stripped) and score < 0.98:
        return True
    if len(normalized) <= 2 and score < 0.55 and not has_cjk(stripped):
        return True
    return False


def unique_lines_by_text(lines: list[OCRLine]) -> list[OCRLine]:
    def source_priority(source: str) -> int:
        if source.startswith("cad.dimension"):
            return 15
        if source.startswith("cad.mleader"):
            return 14
        if source.startswith("cad.attrib"):
            return 13
        if source.startswith("cad.text"):
            return 12
        if source.startswith("hq.title_block.line_free"):
            return 11
        if source.startswith("hq.title_block"):
            return 10
        if source.startswith("hq.table.line_free"):
            return 9
        if source.startswith("hq.table"):
            return 8
        if source.startswith("hq.detail.line_free") or source.startswith("hq.rebar.line_free"):
            return 7
        if source.startswith("hq.detail") or source.startswith("hq.rebar"):
            return 6
        if source.startswith("hq.note.line_free"):
            return 5
        if source.startswith("hq.note"):
            return 4
        if source == "bottom_right.line_free":
            return 3
        if source == "bottom_right":
            return 2
        if source.endswith(".line_free"):
            return 1
        return 0

    best_by_key: dict[str, OCRLine] = {}
    key_order: list[str] = []
    for line in lines:
        key = normalize_text_key(line.text)
        if not key:
            continue
        existing = best_by_key.get(key)
        if existing is None:
            best_by_key[key] = line
            key_order.append(key)
            continue
        existing_rank = (
            1 if match_title_block_key(existing.text) else 0,
            1 if is_engineering_detail_like(existing.text) else 0,
            1 if is_rebar_like(existing.text) else 0,
            source_priority(existing.source),
            len(normalize_text_key(existing.text)),
            existing.score,
        )
        candidate_rank = (
            1 if match_title_block_key(line.text) else 0,
            1 if is_engineering_detail_like(line.text) else 0,
            1 if is_rebar_like(line.text) else 0,
            source_priority(line.source),
            len(normalize_text_key(line.text)),
            line.score,
        )
        if candidate_rank > existing_rank:
            best_by_key[key] = line
    return [best_by_key[key] for key in key_order]


def filter_drawing_lines(lines: list[OCRLine]) -> list[OCRLine]:
    filtered: list[OCRLine] = []
    for line in lines:
        if is_noise_text(line.text, line.score):
            continue
        normalized = normalize_text_key(line.text)
        if len(normalized) <= 2 and line.score < 0.60 and not has_cjk(line.text):
            continue
        filtered.append(line)
    return unique_lines_by_text(filtered)


def count_unique_section_texts(sections: dict[str, list[OCRLine]]) -> int:
    seen: set[str] = set()
    for lines in sections.values():
        for line in lines:
            key = normalize_text_key(line.text)
            if key:
                seen.add(key)
    return len(seen)


def match_title_block_key(text: str) -> str | None:
    normalized = normalize_text_key(text)
    for canonical, aliases in TITLE_BLOCK_KEYWORDS:
        for alias in aliases:
            alias_key = normalize_text_key(alias)
            if alias_key and alias_key in normalized:
                return canonical
    return None


def strip_field_prefix(text: str, canonical: str) -> str:
    value = text.strip()
    for _, aliases in TITLE_BLOCK_KEYWORDS:
        for alias in aliases:
            pattern = r"^\s*" + re.escape(alias) + r"\s*[:：]?\s*"
            value = re.sub(pattern, "", value, count=1, flags=re.IGNORECASE)
    if canonical == "图名" and value == text.strip():
        return value
    return value.strip()


def append_unique_value(mapping: dict[str, list[str]], key: str, value: str) -> None:
    cleaned = value.strip()
    if not cleaned:
        return
    bucket = mapping.setdefault(key, [])
    if cleaned not in bucket:
        bucket.append(cleaned)


def build_drawing_summary(
    page_sections: dict[int, dict[str, list[OCRLine]]],
) -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    preferred_sources = [
        "hq.title_block.raw",
        "hq.title_block.line_free",
        "hq.table.1.raw",
        "hq.table.1.line_free",
        "hq.detail.1.raw",
        "hq.detail.1.line_free",
        "hq.detail.2.raw",
        "hq.detail.2.line_free",
        "hq.note.1.raw",
        "hq.note.1.line_free",
        "hq.rebar.1.raw",
        "hq.rebar.1.line_free",
        "hq.rebar.2.raw",
        "hq.rebar.2.line_free",
        "bottom_right",
        "bottom_right.line_free",
        "bottom_left",
        "bottom_left.line_free",
        "top_right",
        "top_right.line_free",
        "full",
        "full.line_free",
    ]
    for page, sections in page_sections.items():
        fields: dict[str, list[str]] = {}
        candidates: list[str] = []
        seen_candidates: set[str] = set()
        for source in preferred_sources:
            if source not in sections:
                continue
            for line in unique_lines_by_text(sections[source]):
                text = line.text.strip()
                if not text or line.score < 0.55:
                    continue
                canonical = match_title_block_key(text)
                if canonical is None and ("：" in text or ":" in text):
                    parts = re.split(r"[:：]", text, maxsplit=1)
                    left = parts[0] if parts else ""
                    right = parts[1] if len(parts) > 1 else ""
                    canonical = match_title_block_key(left)
                    if canonical and right.strip():
                        append_unique_value(fields, canonical, right.strip())
                elif canonical:
                    stripped = strip_field_prefix(text, canonical)
                    append_unique_value(fields, canonical, stripped or text)

                if canonical or (("：" in text or ":" in text) and len(text) >= 4):
                    if text not in seen_candidates:
                        candidates.append(text)
                        seen_candidates.add(text)
        summary[str(page)] = {"fields": fields, "candidates": candidates}
    return summary


def summary_strength(page_summary: dict[str, Any]) -> int:
    fields = page_summary.get("fields", {})
    candidates = page_summary.get("candidates", [])
    return sum(len(values) for values in fields.values()) + min(len(candidates), 3)


def box_bounds(box: list[list[float]]) -> tuple[int, int, int, int]:
    xs = [int(pt[0]) for pt in box]
    ys = [int(pt[1]) for pt in box]
    return min(xs), min(ys), max(xs), max(ys)


def clip_rect(
    rect: tuple[int, int, int, int],
    width: int,
    height: int,
) -> tuple[int, int, int, int] | None:
    left, top, right, bottom = rect
    left = max(0, min(width - 1, left))
    top = max(0, min(height - 1, top))
    right = max(left + 1, min(width, right))
    bottom = max(top + 1, min(height, bottom))
    if right - left < 40 or bottom - top < 40:
        return None
    return left, top, right, bottom


def crop_rect(image: np.ndarray, rect: tuple[int, int, int, int]) -> np.ndarray:
    left, top, right, bottom = rect
    return image[top:bottom, left:right]


def upscale_image(image: np.ndarray, scale: float) -> np.ndarray:
    if scale <= 1.0:
        return image
    height, width = image.shape[:2]
    return cv2.resize(
        image,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_CUBIC,
    )


def title_block_field_count(page_summary: dict[str, Any]) -> int:
    fields = page_summary.get("fields", {})
    return sum(len(fields.get(key, [])) for key in TITLE_BLOCK_KEYWORDS_DICT)


def line_text_density_score(text: str) -> tuple[int, int, int]:
    normalized = normalize_text_key(text)
    if not normalized:
        return 0, 0, 0
    title_key = match_title_block_key(text)
    digit_count = sum(ch.isdigit() for ch in normalized)
    alpha_count = sum(ch.isalpha() for ch in normalized)
    cjk_count = sum(1 for ch in text if "\u4e00" <= ch <= "\u9fff")
    symbol_count = sum(ch in "@#=:/+-xX×()[]{}.%Φφ" for ch in text)
    detail_score = 0
    table_score = 0
    note_score = 0
    if is_rebar_like(text):
        detail_score += 5
    if is_engineering_detail_like(text):
        detail_score += 4
    if len(normalized) <= 14 and digit_count >= 2:
        detail_score += 2
    if len(normalized) <= 16 and alpha_count >= 1 and digit_count >= 1:
        detail_score += 2
    if len(normalized) <= 16 and symbol_count >= 1:
        detail_score += 2
    if any(keyword in text for keyword in DRAWING_DETAIL_KEYWORDS):
        detail_score += 2

    if title_key is not None:
        table_score += 5
    if "：" in text or ":" in text:
        table_score += 3
    if len(normalized) <= 18 and digit_count >= 1 and (cjk_count >= 1 or alpha_count >= 1):
        table_score += 2
    if len(normalized) <= 12 and symbol_count >= 1 and digit_count >= 1:
        table_score += 1

    if len(normalized) >= 14 and (has_cjk(text) or alpha_count >= 4):
        note_score += 3
    if len(normalized) >= 24:
        note_score += 2
    if any(keyword in text for keyword in DRAWING_NOTE_KEYWORDS):
        note_score += 2
    if title_key is not None:
        note_score = 0
    return detail_score, table_score, note_score


TITLE_BLOCK_KEYWORDS_DICT = {canonical for canonical, _ in TITLE_BLOCK_KEYWORDS}


def choose_dynamic_region_rect(
    lines: list[OCRLine],
    width: int,
    height: int,
    min_width_ratio: float,
    min_height_ratio: float,
) -> tuple[int, int, int, int] | None:
    if not lines:
        return None
    left = width
    top = height
    right = 0
    bottom = 0
    for line in lines:
        x0, y0, x1, y1 = box_bounds(line.box)
        left = min(left, x0)
        top = min(top, y0)
        right = max(right, x1)
        bottom = max(bottom, y1)
    margin_x = max(40, int(width * 0.03))
    margin_y = max(30, int(height * 0.03))
    min_width = int(width * min_width_ratio)
    min_height = int(height * min_height_ratio)
    rect = (
        left - margin_x,
        top - margin_y,
        max(right + margin_x, left + min_width),
        max(bottom + margin_y, top + min_height),
    )
    return clip_rect(rect, width, height)


def region_overlap(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    left = max(a[0], b[0])
    top = max(a[1], b[1])
    right = min(a[2], b[2])
    bottom = min(a[3], b[3])
    if right <= left or bottom <= top:
        return 0.0
    inter = float((right - left) * (bottom - top))
    area_a = float((a[2] - a[0]) * (a[3] - a[1]))
    area_b = float((b[2] - b[0]) * (b[3] - b[1]))
    return inter / max(1.0, min(area_a, area_b))


def detect_hq_regions(
    prepared: PreparedDrawingPage,
    sections: dict[str, list[OCRLine]],
    detail: str = "drawing",
) -> list[tuple[str, tuple[int, int, int, int], float]]:
    full_image = prepared.base_regions["full"]
    height, width = full_image.shape[:2]
    page_summary = build_drawing_summary({prepared.page: sections}).get(str(prepared.page), {})
    full_lines = unique_lines_by_text(
        sections.get("full", []) + sections.get("full.line_free", [])
    )
    tasks: list[tuple[str, tuple[int, int, int, int], float]] = []
    detail_mode = detail if detail in {"drawing", "full", "rebar"} else "drawing"
    page_aspect_ratio = max(width, height) / max(1, min(width, height))
    title_block_scale = 2.0
    table_scale = 1.5
    detail_scale = 1.5
    note_scale = 1.5

    title_fields = page_summary.get("fields", {})
    missing_required = [key for key in TITLE_BLOCK_REQUIRED_FIELDS if not title_fields.get(key)]
    bottom_right_lines = sections.get("bottom_right", []) + sections.get("bottom_right.line_free", [])
    title_keyword_hits = sum(
        1 for line in bottom_right_lines + full_lines if match_title_block_key(line.text) is not None
    )
    if prepared.base_regions.get("bottom_right") is not None and (
        len(missing_required) >= 2
        or summary_strength(page_summary) < 5
        or (len(missing_required) >= 1 and title_keyword_hits >= 2 and len(bottom_right_lines) < 6)
    ):
        left = int(width * 0.56)
        top = int(height * 0.56)
        rect = clip_rect((left, top, width, height), width, height)
        if rect is not None:
            tasks.append(("hq.title_block", rect, title_block_scale))

    grid_rows = 4
    grid_cols = 4
    detail_cells: dict[tuple[int, int], dict[str, Any]] = {}
    table_cells: dict[tuple[int, int], dict[str, Any]] = {}
    note_cells: dict[tuple[int, int], dict[str, Any]] = {}
    explicit_detail_hits = 0
    explicit_rebar_hits = 0
    table_like_hits = 0
    note_like_hits = 0
    for line in full_lines:
        detail_score, table_score, note_score = line_text_density_score(line.text)
        if detail_score <= 0 and table_score <= 0 and note_score <= 0:
            continue
        x0, y0, x1, y1 = box_bounds(line.box)
        cx = (x0 + x1) / 2
        cy = (y0 + y1) / 2
        col = min(grid_cols - 1, max(0, int(cx / max(width, 1) * grid_cols)))
        row = min(grid_rows - 1, max(0, int(cy / max(height, 1) * grid_rows)))
        cell = (row, col)
        if detail_score > 0:
            bucket = detail_cells.setdefault(
                cell,
                {"score": 0, "lines": [], "detail_hits": 0, "rebar_hits": 0},
            )
            bucket["score"] += detail_score
            bucket["lines"].append(line)
            if is_engineering_detail_like(line.text):
                bucket["detail_hits"] += 1
                explicit_detail_hits += 1
            if is_rebar_like(line.text):
                bucket["rebar_hits"] += 1
                explicit_rebar_hits += 1
        if table_score > 0:
            bucket = table_cells.setdefault(
                cell,
                {"score": 0, "lines": [], "title_hits": 0, "colon_hits": 0},
            )
            bucket["score"] += table_score
            bucket["lines"].append(line)
            if match_title_block_key(line.text) is not None:
                bucket["title_hits"] += 1
                table_like_hits += 1
            if "：" in line.text or ":" in line.text:
                bucket["colon_hits"] += 1
                table_like_hits += 1
        if note_score > 0:
            bucket = note_cells.setdefault(cell, {"score": 0, "lines": []})
            bucket["score"] += note_score
            bucket["lines"].append(line)
            if any(keyword in line.text for keyword in DRAWING_NOTE_KEYWORDS):
                note_like_hits += 1

    taken_rects = [task[1] for task in tasks]
    explicit_rebar_context = (
        any("配筋" in line.text or "钢筋" in line.text for line in full_lines)
        or explicit_rebar_hits >= 8
    )
    has_strong_detail_keyword = any(
        any(keyword in line.text for keyword in STRONG_DRAWING_DETAIL_KEYWORDS)
        for line in full_lines
    )
    page_has_general_detail = (
        explicit_rebar_context
        or has_strong_detail_keyword
        or (
            detail_mode == "full"
            and explicit_detail_hits >= 8
        )
        or (
            page_aspect_ratio >= 2.4
            and explicit_detail_hits >= 8
        )
    )

    if summary_strength(page_summary) < 6 or table_like_hits >= 4:
        table_ranked = sorted(
            table_cells.items(),
            key=lambda item: (item[1]["score"], len(item[1]["lines"])),
            reverse=True,
        )
        for _, bucket in table_ranked:
            if bucket["score"] < 10 or len(bucket["lines"]) < 3:
                continue
            rect = choose_dynamic_region_rect(bucket["lines"], width, height, 0.24, 0.18)
            if rect is None or any(region_overlap(rect, existing) > 0.45 for existing in taken_rects):
                continue
            tasks.append(("hq.table.1", rect, table_scale))
            taken_rects.append(rect)
            break

    if page_has_general_detail or (detail_mode == "rebar" and explicit_rebar_context):
        detail_ranked = sorted(
            detail_cells.items(),
            key=lambda item: (item[1]["score"], item[1]["detail_hits"], len(item[1]["lines"])),
            reverse=True,
        )
        detail_index = 1
        for _, bucket in detail_ranked:
            if detail_mode == "full":
                min_score = 13 if explicit_rebar_context else 14
            elif detail_mode == "rebar":
                min_score = 12
            else:
                min_score = 15 if not explicit_rebar_context else 13
            min_hits = 2
            if bucket["score"] < min_score or bucket["detail_hits"] < min_hits:
                continue
            if detail_index == 2 and detail_mode == "drawing" and not explicit_rebar_context and (
                page_aspect_ratio < 2.8 or bucket["score"] < 18 or bucket["detail_hits"] < 4
            ):
                break
            if detail_index == 2 and detail_mode == "full" and not explicit_rebar_context and (
                bucket["score"] < 16 or bucket["detail_hits"] < 3
            ):
                break
            rect = choose_dynamic_region_rect(bucket["lines"], width, height, 0.18, 0.16)
            if rect is None or any(region_overlap(rect, existing) > 0.45 for existing in taken_rects):
                continue
            tasks.append((f"hq.detail.{detail_index}", rect, detail_scale))
            taken_rects.append(rect)
            detail_index += 1
            if detail_index > 2:
                break

    note_context = explicit_rebar_context or any(
        any(keyword in line.text for keyword in DRAWING_NOTE_KEYWORDS) for line in full_lines
    )
    if detail_mode == "full" and note_like_hits >= 2:
        note_context = True
    if detail_mode == "drawing" and page_aspect_ratio >= 2.4 and note_like_hits >= 2:
        note_context = True
    if note_context or explicit_rebar_context:
        note_ranked = sorted(
            note_cells.items(),
            key=lambda item: (item[1]["score"], len(item[1]["lines"])),
            reverse=True,
        )
        for _, bucket in note_ranked:
            if detail_mode == "full":
                min_note_score = 7 if explicit_rebar_context else 8
                min_note_lines = 2
            else:
                min_note_score = 7 if explicit_rebar_context else 9
                min_note_lines = 2 if explicit_rebar_context else 3
            if bucket["score"] < min_note_score or len(bucket["lines"]) < min_note_lines:
                continue
            rect = choose_dynamic_region_rect(bucket["lines"], width, height, 0.28, 0.16)
            if rect is None or any(region_overlap(rect, existing) > 0.45 for existing in taken_rects):
                continue
            tasks.append(("hq.note.1", rect, note_scale))
            break

    return tasks


def run_hq_region_passes(
    prepared: PreparedDrawingPage,
    sections: dict[str, list[OCRLine]],
    engine: str,
    detail: str = "drawing",
) -> None:
    if engine != "paddle":
        return
    full_image = prepared.base_regions["full"]
    line_free_regions = ensure_line_free_regions(prepared)
    line_free_full = line_free_regions.get("full")
    if line_free_full is None:
        return

    for source_prefix, rect, scale in detect_hq_regions(prepared, sections, detail=detail):
        raw_crop = upscale_image(crop_rect(full_image, rect), scale)
        raw_source = f"{source_prefix}.raw"
        raw_lines = filter_drawing_lines(
            run_text_ocr_on_array(
                raw_crop,
                page=prepared.page,
                source=raw_source,
                engine=engine,
            )
        )
        if raw_lines:
            sections[raw_source] = raw_lines

        line_free_crop = upscale_image(crop_rect(line_free_full, rect), scale)
        line_free_source = f"{source_prefix}.line_free"
        line_free_lines = filter_drawing_lines(
            run_text_ocr_on_array(
                line_free_crop,
                page=prepared.page,
                source=line_free_source,
                engine=engine,
            )
        )
        if line_free_lines:
            sections[line_free_source] = line_free_lines


def extract_drawing_regions(image: np.ndarray) -> dict[str, np.ndarray]:
    height, width = image.shape[:2]
    regions: dict[str, np.ndarray] = {}
    for name, (x0, y0, x1, y1) in DRAWING_REGION_SPECS:
        left = int(width * x0)
        top = int(height * y0)
        right = max(left + 1, int(width * x1))
        bottom = max(top + 1, int(height * y1))
        region = image[top:bottom, left:right]
        if region.shape[0] < 80 or region.shape[1] < 80:
            continue
        regions[name] = region
    return regions


def page_render_matrix(page: fitz.Page, dpi: int, target_max_side: int | None = None) -> fitz.Matrix:
    scale = dpi / 72.0
    if target_max_side:
        max_page_side = max(float(page.rect.width), float(page.rect.height))
        if max_page_side > 0:
            capped_scale = target_max_side / max_page_side
            scale = min(scale, capped_scale)
    return fitz.Matrix(scale, scale)


def native_bbox_to_box(bbox: Iterable[float]) -> list[list[float]]:
    x0, y0, x1, y1 = [float(value) for value in bbox]
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def clean_native_pdf_text(text: str) -> str:
    if not text:
        return ""
    normalized = text.replace("\u3000", " ")
    normalized = re.sub(r"[ \t]+", " ", normalized)
    return normalized.strip()


def classify_native_pdf_page(chars: int, words: int) -> str:
    if chars >= PDF_NATIVE_VECTOR_CHAR_THRESHOLD or words >= PDF_NATIVE_VECTOR_WORD_THRESHOLD:
        return "vector_text"
    if chars > 0 or words > 0:
        return "partial_text"
    return "image_only"


def extract_native_pdf_page(page: fitz.Page, page_number: int) -> NativePDFPage:
    text_dict = page.get_text("dict", sort=True)
    lines: list[OCRLine] = []
    chars = 0
    words = 0

    for block in text_dict.get("blocks", []):
        if block.get("type") != 0:
            continue
        for raw_line in block.get("lines", []):
            text = clean_native_pdf_text(
                "".join(str(span.get("text", "")) for span in raw_line.get("spans", []))
            )
            if not text:
                continue
            chars += len(text)
            words += len(re.findall(r"\S+", text))
            bbox = raw_line.get("bbox") or block.get("bbox")
            if not bbox:
                continue
            lines.append(
                OCRLine(
                    text=text,
                    score=1.0,
                    box=native_bbox_to_box(bbox),
                    page=page_number,
                    source="full",
                )
            )

    try:
        images = len(page.get_images(full=True))
    except Exception:
        images = 0

    return NativePDFPage(
        page=page_number,
        text_class=classify_native_pdf_page(chars, words),
        chars=chars,
        words=words,
        images=images,
        lines=filter_drawing_lines(lines),
    )


def sampled_pdf_page_indexes(total_pages: int, limit: int = PDF_NATIVE_SAMPLE_LIMIT) -> list[int]:
    if total_pages <= 0:
        return []
    if total_pages <= limit:
        return list(range(total_pages))
    candidates = [0, 1, total_pages // 4, total_pages // 2, (total_pages * 3) // 4, total_pages - 1]
    return sorted({index for index in candidates if 0 <= index < total_pages})


def should_use_native_pdf_strategy(pdf_path: Path) -> bool:
    with fitz.open(pdf_path) as doc:
        indexes = sampled_pdf_page_indexes(len(doc))
        if not indexes:
            return False
        vector_hits = 0
        partial_hits = 0
        for index in indexes:
            extracted = extract_native_pdf_page(doc[index], index + 1)
            if extracted.text_class == "vector_text":
                vector_hits += 1
            elif extracted.text_class == "partial_text":
                partial_hits += 1
        if len(indexes) == 1:
            return vector_hits >= 1 or partial_hits >= 1
        min_vector_hits = max(2, (len(indexes) + 1) // 2)
        if vector_hits >= min_vector_hits:
            return True
        return vector_hits >= 1 and (vector_hits + partial_hits) >= max(2, (len(indexes) * 2 + 2) // 3)


def merge_native_lines_into_sections(
    sections: dict[str, list[OCRLine]],
    native_lines: list[OCRLine],
) -> dict[str, list[OCRLine]]:
    if not native_lines:
        return sections
    merged = {source: list(lines) for source, lines in sections.items()}
    full_lines = merged.setdefault("full", [])
    full_lines.extend(native_lines)
    merged["full"] = unique_lines_by_text(filter_drawing_lines(full_lines))
    return merged


def is_cad_input(path: Path) -> bool:
    return path.suffix.lower() in CAD_SUFFIXES


def split_cad_text(text: str) -> list[str]:
    lines: list[str] = []
    for part in re.split(r"[\r\n]+", text):
        cleaned = clean_native_pdf_text(part)
        if cleaned:
            lines.append(cleaned)
    return lines


def cad_anchor_point(entity: Any) -> tuple[float, float]:
    for attr in ("insert", "text_midpoint", "defpoint", "location"):
        try:
            point = getattr(entity.dxf, attr)
            return float(point[0]), float(point[1])
        except Exception:
            continue
    return 0.0, 0.0


def cad_text_height(entity: Any) -> float:
    for attr in ("height", "char_height", "text_height"):
        try:
            value = float(getattr(entity.dxf, attr))
            if value > 0:
                return value
        except Exception:
            continue
    return 2.5


def cad_box_from_entity(entity: Any, text: str) -> list[list[float]]:
    x, y = cad_anchor_point(entity)
    height = cad_text_height(entity)
    text_len = max(1, len(normalize_text_key(text)))
    width = max(height * 2.0, min(height * max(2.0, text_len * 0.62), height * 28.0))
    return [[x, y], [x + width, y], [x + width, y + height * 1.6], [x, y + height * 1.6]]


def safe_virtual_entities(entity: Any) -> list[Any]:
    try:
        return list(entity.virtual_entities())
    except Exception:
        return []


def direct_text_source(dxftype: str) -> str:
    if dxftype in {"ATTRIB", "ATTDEF"}:
        return "cad.attrib"
    if dxftype == "DIMENSION":
        return "cad.dimension"
    if dxftype == "MLEADER":
        return "cad.mleader"
    return "cad.text"


def direct_text_lines_from_entity(entity: Any, page: int) -> list[OCRLine]:
    dxftype = entity.dxftype()
    lines: list[OCRLine] = []

    if dxftype in {"TEXT", "MTEXT", "ATTRIB", "ATTDEF"}:
        try:
            raw_text = entity.plain_text()
        except Exception:
            raw_text = getattr(entity.dxf, "text", "")
        for text in split_cad_text(str(raw_text)):
            lines.append(
                OCRLine(
                    text=text,
                    score=1.0,
                    box=cad_box_from_entity(entity, text),
                    page=page,
                    source=direct_text_source(dxftype),
                )
            )
        if dxftype in {"ATTRIB", "ATTDEF"}:
            try:
                virtual_mtext = entity.virtual_mtext_entity()
            except Exception:
                virtual_mtext = None
            if virtual_mtext is not None:
                lines.extend(direct_text_lines_from_entity(virtual_mtext, page))
        return lines

    if dxftype == "DIMENSION":
        try:
            dimension_text = str(getattr(entity.dxf, "text", "")).strip()
        except Exception:
            dimension_text = ""
        if dimension_text and dimension_text not in {"<>", " "}:
            for text in split_cad_text(dimension_text):
                lines.append(
                    OCRLine(
                        text=text,
                        score=1.0,
                        box=cad_box_from_entity(entity, text),
                        page=page,
                        source="cad.dimension",
                    )
                )
        for virtual in safe_virtual_entities(entity):
            lines.extend(direct_text_lines_from_entity(virtual, page))
        return lines

    if dxftype == "MLEADER":
        for virtual in safe_virtual_entities(entity):
            lines.extend(direct_text_lines_from_entity(virtual, page))
        for line in lines:
            line.source = "cad.mleader"
        return lines

    if dxftype == "INSERT":
        for attrib in getattr(entity, "attribs", []):
            lines.extend(direct_text_lines_from_entity(attrib, page))
        for virtual in safe_virtual_entities(entity):
            lines.extend(direct_text_lines_from_entity(virtual, page))
        return lines

    return lines


def extract_cad_layout_sections(layout: Any, page: int) -> dict[str, list[OCRLine]]:
    sections: dict[str, list[OCRLine]] = {}
    full_lines: list[OCRLine] = []
    for entity in layout:
        entity_lines = direct_text_lines_from_entity(entity, page)
        if not entity_lines:
            continue
        for line in entity_lines:
            sections.setdefault(line.source, []).append(line)
            full_lines.append(line)
    for source, lines in list(sections.items()):
        sections[source] = unique_lines_by_text(filter_drawing_lines(lines))
    sections["full"] = unique_lines_by_text(filter_drawing_lines(full_lines))
    return sections


def merge_additional_sections(
    base_sections: dict[str, list[OCRLine]],
    extra_sections: dict[str, list[OCRLine]],
) -> dict[str, list[OCRLine]]:
    merged = {source: list(lines) for source, lines in base_sections.items()}
    for source, lines in extra_sections.items():
        bucket = merged.setdefault(source, [])
        bucket.extend(lines)
        merged[source] = unique_lines_by_text(filter_drawing_lines(bucket))

    cad_lines: list[OCRLine] = []
    for source, lines in merged.items():
        if source.startswith("cad."):
            cad_lines.extend(lines)
    full_lines = cad_lines + list(merged.get("full", []))
    if full_lines:
        merged["full"] = unique_lines_by_text(filter_drawing_lines(full_lines))
    return merged


def should_keep_cad_ocr_line(line: OCRLine) -> bool:
    text = line.text.strip()
    if not text:
        return False
    normalized = normalize_text_key(text)
    if not normalized:
        return False
    if match_title_block_key(text) is not None:
        return True
    if is_engineering_detail_like(text) or is_rebar_like(text):
        return True
    if has_cjk(text):
        return len(normalized) >= 2 and line.score >= 0.55
    digit_count = sum(ch.isdigit() for ch in normalized)
    alpha_count = sum(ch.isalpha() for ch in normalized)
    if len(normalized) >= 4 and line.score >= 0.55:
        return True
    if len(normalized) >= 3 and digit_count >= 1 and alpha_count >= 1:
        return True
    if len(normalized) >= 3 and digit_count >= 2 and line.score >= 0.72:
        return True
    return False


def filter_cad_ocr_sections(
    sections: dict[str, list[OCRLine]],
    trusted_direct_count: int,
) -> dict[str, list[OCRLine]]:
    if trusted_direct_count < 12:
        return sections
    filtered: dict[str, list[OCRLine]] = {}
    for source, lines in sections.items():
        kept = [line for line in lines if should_keep_cad_ocr_line(line)]
        if kept:
            filtered[source] = unique_lines_by_text(filter_drawing_lines(kept))
    if "full" not in filtered and sections.get("full"):
        kept = [line for line in sections["full"] if should_keep_cad_ocr_line(line)]
        if kept:
            filtered["full"] = unique_lines_by_text(filter_drawing_lines(kept))
    return filtered


def repage_sections(
    sections: dict[str, list[OCRLine]],
    page: int,
) -> dict[str, list[OCRLine]]:
    remapped: dict[str, list[OCRLine]] = {}
    for source, lines in sections.items():
        remapped[source] = [
            OCRLine(
                text=line.text,
                score=line.score,
                box=[list(point) for point in line.box],
                page=page,
                source=line.source,
            )
            for line in lines
        ]
    return remapped


def detect_oda_converter() -> Path | None:
    candidates: list[Path] = []
    env_path = os.environ.get("OCRX_ODAFC_PATH")
    if env_path:
        candidates.append(Path(env_path))
    for env_name in ("ProgramFiles", "ProgramFiles(x86)"):
        root = os.environ.get(env_name)
        if not root:
            continue
        oda_root = Path(root) / "ODA"
        if not oda_root.exists():
            continue
        for folder in sorted(oda_root.glob("ODAFileConverter*"), reverse=True):
            exe = folder / "ODAFileConverter.exe"
            if exe.exists():
                candidates.append(exe)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_cad_document(
    input_path: Path,
    temp_dir: Path,
) -> tuple[Any, Path, Path | None]:
    import ezdxf
    from ezdxf.addons import odafc

    working_path = input_path
    oda_converter = detect_oda_converter()
    if input_path.suffix.lower() == ".dwg":
        if oda_converter is None:
            raise RuntimeError(
                "ODA File Converter not found; install ODA File Converter or set OCRX_ODAFC_PATH."
            )
        ezdxf.options.set("odafc-addon", "win_exec_path", str(oda_converter))
        working_path = temp_dir / f"{input_path.stem}.converted.dxf"
        odafc.convert(str(input_path), str(working_path), replace=True, audit=True)
    doc = ezdxf.readfile(str(working_path))
    return doc, working_path, oda_converter


def cad_layout_entity_count(layout: Any) -> int:
    try:
        return len(layout)
    except Exception:
        return sum(1 for _ in layout)


def select_cad_layouts(doc: Any) -> list[Any]:
    paperspaces: list[Any] = []
    for name in doc.layout_names_in_taborder():
        layout = doc.layouts.get(name)
        if layout is None or getattr(layout, "is_modelspace", False):
            continue
        if cad_layout_entity_count(layout) > 0:
            paperspaces.append(layout)
    if paperspaces:
        return paperspaces
    model = doc.modelspace()
    return [model] if cad_layout_entity_count(model) > 0 else []


def cad_render_size_inches(layout: Any, dpi: int) -> tuple[float, float]:
    from ezdxf import bbox as ezdxf_bbox

    max_inches = CAD_RENDER_MAX_SIDE_PX / max(1, dpi)
    try:
        extents = ezdxf_bbox.extents(layout, fast=True)
        if extents.has_data:
            size = extents.size
            width = abs(float(size.x))
            height = abs(float(size.y))
            if width > 0 and height > 0:
                aspect = width / max(height, 1e-6)
                if aspect >= 1.0:
                    return max_inches, max(CAD_RENDER_MIN_SIDE_INCHES, max_inches / aspect)
                return max(CAD_RENDER_MIN_SIDE_INCHES, max_inches * aspect), max_inches
    except Exception:
        pass
    return max_inches, 0.0


def sanitize_layout_name(name: str) -> str:
    cleaned = re.sub(r"[^\w\-\.]+", "_", name.strip())
    return cleaned or "layout"


def render_cad_layout_to_png(
    layout: Any,
    out_path: Path,
    dpi: int,
) -> None:
    from ezdxf.addons.drawing import matplotlib as drawing_matplotlib

    size_inches = cad_render_size_inches(layout, dpi)
    drawing_matplotlib.qsave(
        layout,
        str(out_path),
        bg=CAD_RENDER_BG,
        fg=CAD_RENDER_FG,
        dpi=dpi,
        size_inches=size_inches,
    )


def prepare_drawing_page(
    raw: np.ndarray,
    page: int,
    min_width: int,
    engine: str,
    path: Path | None = None,
) -> PreparedDrawingPage:
    base_profile = "general" if engine == "paddle" else "drawing"
    base = preprocess_image(
        ensure_bgr_image(raw),
        profile=base_profile,
        min_width=max(min_width, 2200),
    )
    base_regions = extract_drawing_regions(base)
    return PreparedDrawingPage(
        page=page,
        base_regions=base_regions,
        path=path,
    )


def ensure_line_free_regions(prepared: PreparedDrawingPage) -> dict[str, np.ndarray]:
    if prepared.line_free_regions is None:
        line_free = remove_structural_lines(
            cv2.cvtColor(prepared.base_regions["full"], cv2.COLOR_BGR2GRAY)
        )
        prepared.line_free_regions = extract_drawing_regions(line_free)
    return prepared.line_free_regions


def ensure_rotated_fulls(prepared: PreparedDrawingPage) -> dict[str, np.ndarray]:
    if prepared.rotated_fulls is None:
        full = prepared.base_regions["full"]
        prepared.rotated_fulls = {
            "full.rot90": rotate_image(full, 90),
            "full.rot270": rotate_image(full, 270),
        }
    return prepared.rotated_fulls


def run_drawing_ocr_on_image(
    image_path: Path,
    page: int,
    min_width: int,
    engine: str,
    detail: str = "auto",
) -> dict[str, list[OCRLine]]:
    raw = read_image(image_path)
    if raw is None:
        raise ValueError(f"Unable to read drawing image: {image_path}")
    prepared = prepare_drawing_page(
        raw,
        page=page,
        min_width=min_width,
        engine=engine,
        path=image_path,
    )
    return run_drawing_ocr_on_prepared(prepared, engine=engine, detail=detail)


def run_drawing_ocr_on_array(
    raw: np.ndarray,
    page: int,
    min_width: int,
    engine: str,
    detail: str = "auto",
) -> dict[str, list[OCRLine]]:
    prepared = prepare_drawing_page(raw, page=page, min_width=min_width, engine=engine)
    return run_drawing_ocr_on_prepared(prepared, engine=engine, detail=detail)


def run_drawing_ocr_on_prepared(
    prepared: PreparedDrawingPage,
    engine: str,
    detail: str = "auto",
) -> dict[str, list[OCRLine]]:
    resolved_detail = "drawing" if detail == "auto" else detail
    sections: dict[str, list[OCRLine]] = {}
    full_lines = filter_drawing_lines(
        run_text_ocr_on_array(
            prepared.base_regions["full"],
            page=prepared.page,
            source="full",
            engine=engine,
        )
    )
    if full_lines:
        sections["full"] = full_lines

    if engine == "paddle":
        full_summary = build_drawing_summary({prepared.page: sections}).get(str(prepared.page), {})
        full_strength = summary_strength(full_summary)
        full_unique = count_unique_section_texts(sections)
        # Paddle full-page OCR is already strong on many large drawings. Skip the
        # heavier corner/rotation passes when the page is already rich, but still
        # allow the detail path to run targeted local rescans.
        if full_strength >= 24 and full_unique >= 350:
            if resolved_detail in {"drawing", "full", "rebar"}:
                run_hq_region_passes(prepared, sections, engine=engine, detail=resolved_detail)
            return sections

    for region_name in ("bottom_right", "bottom_left", "top_right"):
        region_image = prepared.base_regions.get(region_name)
        if region_image is None:
            continue
        lines = filter_drawing_lines(
            run_text_ocr_on_array(
                region_image,
                page=prepared.page,
                source=region_name,
                engine=engine,
            )
        )
        if len(lines) >= 2:
            sections[region_name] = lines

    if engine == "paddle":
        early_summary = build_drawing_summary({prepared.page: sections}).get(str(prepared.page), {})
        early_strength = summary_strength(early_summary)
        early_unique = count_unique_section_texts(sections)
        if early_strength >= 4 and early_unique >= 350:
            if resolved_detail in {"drawing", "full", "rebar"}:
                run_hq_region_passes(prepared, sections, engine=engine, detail=resolved_detail)
            return sections

    line_free_regions = ensure_line_free_regions(prepared)
    full_line_free = filter_drawing_lines(
        run_text_ocr_on_array(
            line_free_regions["full"],
            page=prepared.page,
            source="full.line_free",
            engine=engine,
        )
    )
    if full_line_free:
        sections["full.line_free"] = full_line_free

    if engine == "paddle":
        linefree_summary = build_drawing_summary({prepared.page: sections}).get(str(prepared.page), {})
        linefree_strength = summary_strength(linefree_summary)
        linefree_unique = count_unique_section_texts(sections)
        # For Paddle, a strong full + line_free result usually means later passes only add
        # page-number fragments or duplicate title-block text.
        if linefree_strength >= 14 and linefree_unique >= 70:
            if resolved_detail in {"drawing", "full", "rebar"}:
                run_hq_region_passes(prepared, sections, engine=engine, detail=resolved_detail)
            return sections

    for region_name in ("bottom_right", "bottom_left", "top_right"):
        region_image = line_free_regions.get(region_name)
        if region_image is None:
            continue
        source = f"{region_name}.line_free"
        lines = filter_drawing_lines(
            run_text_ocr_on_array(
                region_image,
                page=prepared.page,
                source=source,
                engine=engine,
            )
        )
        if len(lines) >= 2:
            sections[source] = lines

    if engine == "paddle":
        linefree_summary = build_drawing_summary({prepared.page: sections}).get(str(prepared.page), {})
        linefree_strength = summary_strength(linefree_summary)
        linefree_unique = count_unique_section_texts(sections)
        if linefree_strength >= 12 and linefree_unique >= 80:
            if resolved_detail in {"drawing", "full", "rebar"}:
                run_hq_region_passes(prepared, sections, engine=engine, detail=resolved_detail)
            return sections

    for source, rotated in ensure_rotated_fulls(prepared).items():
        lines = filter_drawing_lines(
            run_text_ocr_on_array(
                rotated,
                page=prepared.page,
                source=source,
                engine=engine,
            )
        )
        if len(lines) >= 2:
            sections[source] = lines

    if resolved_detail in {"drawing", "full", "rebar"}:
        run_hq_region_passes(prepared, sections, engine=engine, detail=resolved_detail)

    return sections


def iter_rendered_pdf_pages(
    pdf_path: Path,
    dpi: int,
    target_max_side: int | None = None,
    keep_dir: Path | None = None,
) -> Iterator[RenderedPage]:
    if keep_dir is not None:
        ensure_output_dir(keep_dir)

    data_queue: queue.Queue[Any] = queue.Queue(maxsize=2)
    sentinel = object()

    def producer() -> None:
        try:
            with fitz.open(pdf_path) as doc:
                for idx, page in enumerate(doc, start=1):
                    matrix = page_render_matrix(page, dpi=dpi, target_max_side=target_max_side)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    image = pixmap_to_bgr(pix)
                    out_path: Path | None = None
                    if keep_dir is not None:
                        out_path = keep_dir / f"{pdf_path.stem}_page_{idx:03d}.png"
                        cv2.imencode(".png", image)[1].tofile(str(out_path))
                    data_queue.put(RenderedPage(page=idx, image=image, path=out_path))
        except Exception as error:
            data_queue.put(error)
        finally:
            data_queue.put(sentinel)

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()

    while True:
        item = data_queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    worker.join()


def iter_prepared_drawing_pdf_pages(
    pdf_path: Path,
    dpi: int,
    min_width: int,
    engine: str,
    target_max_side: int | None = None,
    keep_dir: Path | None = None,
) -> Iterator[PreparedDrawingPage]:
    if keep_dir is not None:
        ensure_output_dir(keep_dir)

    prepare_workers = get_drawing_prepare_workers()
    queue_size = get_drawing_queue_size(prepare_workers)
    data_queue: queue.Queue[Any] = queue.Queue(maxsize=queue_size)
    sentinel = object()

    def producer() -> None:
        try:
            with fitz.open(pdf_path) as doc:
                with ThreadPoolExecutor(max_workers=prepare_workers) as executor:
                    in_flight: set[Any] = set()

                    def submit_page(idx: int, image: np.ndarray, out_path: Path | None) -> None:
                        future = executor.submit(
                            prepare_drawing_page,
                            image,
                            page=idx,
                            min_width=min_width,
                            engine=engine,
                            path=out_path,
                        )
                        in_flight.add(future)

                    def drain_one() -> None:
                        if not in_flight:
                            return
                        done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                        for future in done:
                            in_flight.remove(future)
                            data_queue.put(future.result())

                    for idx, page in enumerate(doc, start=1):
                        matrix = page_render_matrix(page, dpi=dpi, target_max_side=target_max_side)
                        pix = page.get_pixmap(matrix=matrix, alpha=False)
                        image = pixmap_to_bgr(pix)
                        out_path: Path | None = None
                        if keep_dir is not None:
                            out_path = keep_dir / f"{pdf_path.stem}_page_{idx:03d}.png"
                            cv2.imencode(".png", image)[1].tofile(str(out_path))
                        submit_page(idx, image, out_path)
                        while len(in_flight) >= queue_size:
                            drain_one()

                    while in_flight:
                        drain_one()
        except Exception as error:
            data_queue.put(error)
        finally:
            data_queue.put(sentinel)

    worker = threading.Thread(target=producer, daemon=True)
    worker.start()

    while True:
        item = data_queue.get()
        if item is sentinel:
            break
        if isinstance(item, Exception):
            raise item
        yield item

    worker.join()


def lines_to_text(lines: list[OCRLine]) -> str:
    pages: dict[int, list[str]] = {}
    for line in lines:
        pages.setdefault(line.page, []).append(line.text)
    chunks: list[str] = []
    for page in sorted(pages):
        if len(pages) > 1:
            chunks.append(f"[Page {page}]")
        chunks.extend(pages[page])
        chunks.append("")
    return "\n".join(chunks).strip() + "\n"


def render_drawing_text(
    page_sections: dict[int, dict[str, list[OCRLine]]],
    summary: dict[str, dict[str, Any]] | None = None,
) -> str:
    chunks: list[str] = []
    multi_page = len(page_sections) > 1

    for page in sorted(page_sections):
        sections = page_sections[page]
        if multi_page:
            chunks.append(f"[Page {page}]")

        seen: set[str] = set()
        primary_lines: list[OCRLine] = []
        for source, lines in sections.items():
            if source == "full" or source.startswith(HQ_SOURCE_PREFIXES):
                primary_lines.extend(lines)
        if not primary_lines:
            primary_lines = sections.get("full", [])
        primary = unique_lines_by_text(primary_lines)
        for line in primary:
            key = normalize_text_key(line.text)
            if key in seen:
                continue
            seen.add(key)
            chunks.append(line.text)

        if primary:
            chunks.append("")

        ordered_sources = sorted(
            (source for source in sections if source != "full"),
            key=lambda item: (
                -1
                if item.startswith(HQ_SOURCE_PREFIXES)
                else (
                    DRAWING_SOURCE_ORDER.index(item)
                    if item in DRAWING_SOURCE_ORDER
                    else len(DRAWING_SOURCE_ORDER)
                ),
                item,
            ),
        )
        for source in ordered_sources:
            fresh_texts: list[str] = []
            for line in unique_lines_by_text(sections[source]):
                key = normalize_text_key(line.text)
                if key in seen:
                    continue
                seen.add(key)
                fresh_texts.append(line.text)
            if not fresh_texts:
                continue
            chunks.append(f"[{source}]")
            chunks.extend(fresh_texts)
            chunks.append("")

        if summary:
            page_summary = summary.get(str(page), {})
            fields = page_summary.get("fields", {})
            if fields:
                chunks.append("[title_block]")
                for field_name, values in fields.items():
                    for value in values:
                        if value == field_name:
                            chunks.append(field_name)
                        else:
                            chunks.append(f"{field_name}: {value}")
                chunks.append("")

    return "\n".join(chunks).strip() + "\n"


def maybe_make_searchable_pdf(
    input_path: Path,
    output_dir: Path,
    languages: str,
) -> Path | None:
    if input_path.suffix.lower() != ".pdf":
        return None
    exe = resolve_ocrmypdf_executable()
    if not exe:
        return None
    tesseract_exe = resolve_tesseract_executable()
    if not tesseract_exe:
        print("SEARCHABLE_PDF_WARNING: tesseract not found; skipping searchable PDF generation.")
        return None

    out_pdf = output_dir / f"{input_path.stem}.searchable.pdf"
    sidecar = output_dir / f"{input_path.stem}.sidecar.txt"
    cmd = [
        exe,
        "--force-ocr",
        "--sidecar",
        str(sidecar),
        "-l",
        languages,
        str(input_path),
        str(out_pdf),
    ]
    try:
        env = os.environ.copy()
        extra_dirs: list[Path | str] = [Path(exe).parent, Path(tesseract_exe).parent]
        env["PATH"] = subprocess_search_path(extra_dirs)
        subprocess.run(cmd, check=True, env=env)
        return out_pdf
    except subprocess.CalledProcessError as error:
        print(f"SEARCHABLE_PDF_WARNING: {error}")
        return None


def save_general_outputs(
    input_path: Path,
    output_dir: Path,
    lines: list[OCRLine],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    txt_path = output_dir / f"{input_path.stem}.ocr.txt"
    json_path = output_dir / f"{input_path.stem}.ocr.json"

    payload = {
        "input": str(input_path),
        "line_count": len(lines),
        "lines": [asdict(line) for line in lines],
    }
    if extra:
        payload.update(extra)

    txt_path.write_text(lines_to_text(lines), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def save_drawing_outputs(
    input_path: Path,
    output_dir: Path,
    page_sections: dict[int, dict[str, list[OCRLine]]],
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    txt_path = output_dir / f"{input_path.stem}.ocr.txt"
    json_path = output_dir / f"{input_path.stem}.ocr.json"
    summary = build_drawing_summary(page_sections)

    payload_pages: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for page, sections in page_sections.items():
        payload_pages[str(page)] = {
            source: [asdict(line) for line in unique_lines_by_text(lines)]
            for source, lines in sections.items()
        }

    payload: dict[str, Any] = {
        "input": str(input_path),
        "mode": "drawing",
        "pages": payload_pages,
        "summary": summary,
    }
    if extra:
        payload.update(extra)

    txt_path.write_text(render_drawing_text(page_sections, summary=summary), encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def save_math_outputs(
    input_path: Path,
    output_dir: Path,
    latex_text: str,
    extra: dict[str, Any] | None = None,
) -> tuple[Path, Path]:
    txt_path = output_dir / f"{input_path.stem}.math.txt"
    json_path = output_dir / f"{input_path.stem}.math.json"
    payload = {"input": str(input_path), "latex": latex_text}
    if extra:
        payload.update(extra)
    txt_path.write_text(latex_text.strip() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return txt_path, json_path


def run_general_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)
    resolved_engine = resolve_ocr_engine(args.engine)
    render_max_side = 3800 if resolved_engine == "paddle" else None

    if input_path.suffix.lower() == ".pdf":
        lines: list[OCRLine] = []
        keep_dir = output_dir / "_pages" if getattr(args, "keep_pages", False) else None
        for rendered_page in iter_rendered_pdf_pages(
            input_path,
            dpi=args.dpi,
            target_max_side=render_max_side,
            keep_dir=keep_dir,
        ):
            lines.extend(
                run_general_ocr_on_array(
                    rendered_page.image,
                    page=rendered_page.page,
                    profile=args.profile,
                    min_width=args.min_width,
                    engine=resolved_engine,
                )
            )
        searchable_pdf = (
            maybe_make_searchable_pdf(input_path, output_dir, args.ocrmypdf_lang)
            if args.searchable_pdf
            else None
        )
        txt_path, json_path = save_general_outputs(
            input_path,
            output_dir,
            lines,
            extra={
                "mode": "pdf",
                "ocr_engine": resolved_engine,
                "searchable_pdf": str(searchable_pdf) if searchable_pdf else None,
            },
        )
    else:
        lines = run_general_ocr_on_image(
            image_path=input_path,
            page=1,
            profile=args.profile,
            min_width=args.min_width,
            engine=resolved_engine,
        )
        txt_path, json_path = save_general_outputs(
            input_path,
            output_dir,
            lines,
            extra={"mode": "general", "ocr_engine": resolved_engine},
        )

    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    return 0


def run_drawing_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)
    resolved_engine = resolve_ocr_engine(args.engine)
    requested_detail = getattr(args, "detail", "auto")
    resolved_detail = "drawing" if requested_detail == "auto" else requested_detail
    render_max_side = 3800 if resolved_engine == "paddle" else None

    page_sections: dict[int, dict[str, list[OCRLine]]] = {}
    if input_path.suffix.lower() == ".pdf":
        keep_dir = output_dir / "_pages" if getattr(args, "keep_pages", False) else None
        page_parse: dict[str, dict[str, Any]] = {}
        pdf_strategy = "ocr_only"
        if should_use_native_pdf_strategy(input_path):
            pdf_strategy = "native_first"
            with fitz.open(input_path) as doc:
                for idx, page in enumerate(doc, start=1):
                    extracted = extract_native_pdf_page(page, idx)
                    page_parse[str(idx)] = {
                        "class": extracted.text_class,
                        "chars": extracted.chars,
                        "words": extracted.words,
                        "images": extracted.images,
                        "native_line_count": len(extracted.lines),
                    }
                    if extracted.text_class == "vector_text" and extracted.lines:
                        page_sections[idx] = {"full": extracted.lines}
                        continue

                    matrix = page_render_matrix(page, dpi=args.dpi, target_max_side=render_max_side)
                    pix = page.get_pixmap(matrix=matrix, alpha=False)
                    image = pixmap_to_bgr(pix)
                    out_path: Path | None = None
                    if keep_dir is not None:
                        ensure_output_dir(keep_dir)
                        out_path = keep_dir / f"{input_path.stem}_page_{idx:03d}.png"
                        cv2.imencode(".png", image)[1].tofile(str(out_path))
                    prepared_page = prepare_drawing_page(
                        image,
                        page=idx,
                        min_width=args.min_width,
                        engine=resolved_engine,
                        path=out_path,
                    )
                    sections = run_drawing_ocr_on_prepared(
                        prepared_page,
                        engine=resolved_engine,
                        detail=resolved_detail,
                    )
                    page_sections[idx] = merge_native_lines_into_sections(sections, extracted.lines)
        else:
            for prepared_page in iter_prepared_drawing_pdf_pages(
                input_path,
                dpi=args.dpi,
                min_width=args.min_width,
                engine=resolved_engine,
                target_max_side=render_max_side,
                keep_dir=keep_dir,
            ):
                page_sections[prepared_page.page] = run_drawing_ocr_on_prepared(
                    prepared_page,
                    engine=resolved_engine,
                    detail=resolved_detail,
                )
        searchable_pdf = (
            maybe_make_searchable_pdf(input_path, output_dir, args.ocrmypdf_lang)
            if args.searchable_pdf
            else None
        )
        page_parse_summary = {
            "vector_text_pages": sum(1 for meta in page_parse.values() if meta["class"] == "vector_text"),
            "partial_text_pages": sum(1 for meta in page_parse.values() if meta["class"] == "partial_text"),
            "image_only_pages": sum(1 for meta in page_parse.values() if meta["class"] == "image_only"),
        }
        extra = {
            "searchable_pdf": str(searchable_pdf) if searchable_pdf else None,
            "profile": "drawing",
            "ocr_engine": resolved_engine,
            "detail": resolved_detail,
            "pdf_strategy": pdf_strategy,
            "page_parse": page_parse,
            "page_parse_summary": page_parse_summary,
        }
    else:
        page_sections[1] = run_drawing_ocr_on_image(
            image_path=input_path,
            page=1,
            min_width=args.min_width,
            engine=resolved_engine,
            detail=resolved_detail,
        )
        extra = {"profile": "drawing", "ocr_engine": resolved_engine, "detail": resolved_detail}

    txt_path, json_path = save_drawing_outputs(
        input_path,
        output_dir,
        page_sections,
        extra=extra,
    )
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    return 0


def run_cad_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    if not is_cad_input(input_path):
        raise ValueError("cad mode only accepts DWG or DXF input")

    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)
    resolved_engine = resolve_ocr_engine(args.engine)
    requested_detail = getattr(args, "detail", "auto")
    resolved_detail = "full" if requested_detail == "auto" else requested_detail
    render_dpi = max(getattr(args, "dpi", CAD_DEFAULT_DPI), CAD_DEFAULT_DPI)
    min_width = max(getattr(args, "min_width", CAD_DEFAULT_MIN_WIDTH), CAD_DEFAULT_MIN_WIDTH)
    keep_dir = output_dir / "_pages" if getattr(args, "keep_pages", False) else None
    if keep_dir is not None:
        ensure_output_dir(keep_dir)

    logging.getLogger("ezdxf").setLevel(logging.ERROR)
    page_sections: dict[int, dict[str, list[OCRLine]]] = {}
    cad_pages: dict[str, dict[str, Any]] = {}

    with tempfile.TemporaryDirectory(prefix="ocrx_cad_") as temp_name:
        temp_dir = Path(temp_name)
        doc, working_path, oda_converter = load_cad_document(input_path, temp_dir)
        layouts = select_cad_layouts(doc)
        if not layouts:
            raise ValueError("No non-empty CAD layouts found")
        modelspace_aux_sections: dict[str, list[OCRLine]] = {}
        if len(layouts) == 1 and not getattr(layouts[0], "is_modelspace", False):
            modelspace_aux_sections = extract_cad_layout_sections(doc.modelspace(), 0)
        modelspace_aux_count = count_unique_section_texts(modelspace_aux_sections)

        for page_number, layout in enumerate(layouts, start=1):
            direct_sections = extract_cad_layout_sections(layout, page_number)
            aux_count = 0
            if modelspace_aux_sections and modelspace_aux_count > count_unique_section_texts(direct_sections):
                direct_sections = merge_additional_sections(
                    direct_sections,
                    repage_sections(modelspace_aux_sections, page_number),
                )
                aux_count = modelspace_aux_count
            direct_count = count_unique_section_texts(direct_sections)
            layout_name = getattr(layout, "name", f"layout_{page_number}")
            page_meta: dict[str, Any] = {
                "layout_name": layout_name,
                "layout_type": "modelspace" if getattr(layout, "is_modelspace", False) else "paperspace",
                "entity_count": cad_layout_entity_count(layout),
                "direct_text_count": direct_count,
                "modelspace_aux_text_count": aux_count,
                "rendered_png": None,
                "ocr_text_count": 0,
                "merged_text_count": direct_count,
                "render_error": None,
            }

            render_dir = keep_dir if keep_dir is not None else temp_dir
            render_path = render_dir / f"{input_path.stem}_{page_number:03d}_{sanitize_layout_name(layout_name)}.png"
            ocr_sections: dict[str, list[OCRLine]] = {}
            try:
                render_cad_layout_to_png(layout, render_path, render_dpi)
                if render_path.exists():
                    page_meta["rendered_png"] = str(render_path) if keep_dir is not None else None
                    ocr_sections = run_drawing_ocr_on_image(
                        image_path=render_path,
                        page=page_number,
                        min_width=min_width,
                        engine=resolved_engine,
                        detail=resolved_detail,
                    )
                    ocr_sections = filter_cad_ocr_sections(ocr_sections, direct_count)
                    page_meta["ocr_text_count"] = count_unique_section_texts(ocr_sections)
            except Exception as error:
                page_meta["render_error"] = str(error)

            page_sections[page_number] = merge_additional_sections(ocr_sections, direct_sections)
            page_meta["merged_text_count"] = count_unique_section_texts(page_sections[page_number])
            cad_pages[str(page_number)] = page_meta

    extra = {
        "mode": "cad",
        "profile": "cad",
        "ocr_engine": resolved_engine,
        "detail": resolved_detail,
        "cad_strategy": "direct_extract_then_ocr_fallback",
        "cad_input_format": input_path.suffix.lower().lstrip("."),
        "cad_working_format": working_path.suffix.lower().lstrip("."),
        "oda_converter": str(oda_converter) if oda_converter else None,
        "cad_pages": cad_pages,
    }
    txt_path, json_path = save_drawing_outputs(
        input_path,
        output_dir,
        page_sections,
        extra=extra,
    )
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    return 0


def run_math_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    latex_text, score = run_math_ocr(input_path)
    txt_path, json_path = save_math_outputs(
        input_path,
        output_dir,
        latex_text,
        extra={"mode": "math", "score": score},
    )
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    return 0


def should_try_math(input_path: Path, general_lines: list[OCRLine]) -> bool:
    image = read_image(input_path)
    if image is None:
        return False
    height, width = image.shape[:2]
    if not general_lines:
        return True
    if max(width, height) <= 900 and len(general_lines) <= 2:
        return True
    joined = " ".join(line.text for line in general_lines)
    if len(joined.strip()) <= 10:
        return True
    return False


def run_auto_command(args: argparse.Namespace) -> int:
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_dir)

    if is_cad_input(input_path):
        return run_cad_command(args)

    if input_path.suffix.lower() == ".pdf":
        if args.profile == "drawing":
            return run_drawing_command(args)
        return run_general_command(args)

    if args.profile == "drawing":
        return run_drawing_command(args)

    resolved_engine = resolve_ocr_engine(args.engine)
    general_lines = run_general_ocr_on_image(
        image_path=input_path,
        page=1,
        profile=args.profile,
        min_width=args.min_width,
        engine=resolved_engine,
    )
    math_text = ""
    math_score: float | None = None
    try_math = should_try_math(input_path, general_lines)
    if try_math:
        try:
            math_text, math_score = run_math_ocr(input_path)
            math_text = math_text.strip()
        except Exception:
            math_text = ""
            math_score = None

    general_text = lines_to_text(general_lines).strip()
    chosen_engine = "general"
    chosen_text = general_text

    if math_text and not general_text:
        chosen_engine = "math"
        chosen_text = math_text
    elif math_text and general_text and len(general_text.replace(" ", "")) <= 8:
        chosen_engine = "math"
        chosen_text = math_text

    txt_path = output_dir / f"{input_path.stem}.ocr.txt"
    json_path = output_dir / f"{input_path.stem}.ocr.json"
    txt_path.write_text(chosen_text.strip() + "\n", encoding="utf-8")
    json_path.write_text(
        json.dumps(
            {
                "input": str(input_path),
                "mode": "auto",
                "chosen_engine": chosen_engine,
                "ocr_engine": resolved_engine,
                "general_text": general_text,
                "math_text": math_text,
                "math_score": math_score,
                "lines": [asdict(line) for line in general_lines],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"TXT: {txt_path}")
    print(f"JSON: {json_path}")
    if math_text:
        print(f"MATH: {math_text}")
    return 0


def run_pdf_command(args: argparse.Namespace) -> int:
    if Path(args.input).suffix.lower() != ".pdf":
        raise ValueError("pdf mode only accepts PDF input")
    return run_general_command(args)


def default_batch_patterns(mode: str) -> list[str]:
    image_patterns = ["*.png", "*.jpg", "*.jpeg", "*.bmp", "*.tif", "*.tiff", "*.webp"]
    cad_patterns = ["*.dwg", "*.dxf"]
    if mode == "pdf":
        return ["*.pdf"]
    if mode == "math":
        return image_patterns
    if mode == "general":
        return image_patterns
    if mode == "cad":
        return cad_patterns
    if mode == "drawing":
        return image_patterns + ["*.pdf"]
    return image_patterns + ["*.pdf"] + cad_patterns


def iter_batch_inputs(root: Path, recursive: bool, patterns: list[str]) -> list[Path]:
    iterator = root.rglob("*") if recursive else root.glob("*")
    files: list[Path] = []
    for path in iterator:
        if not path.is_file():
            continue
        if any(fnmatch.fnmatch(path.name.lower(), pattern.lower()) for pattern in patterns):
            files.append(path)
    return sorted(files)


def clone_args_for_input(
    args: argparse.Namespace,
    command: str,
    input_path: Path,
    output_dir: Path,
) -> argparse.Namespace:
    return argparse.Namespace(
        command=command,
        input=str(input_path),
        output_dir=str(output_dir),
        profile=args.profile,
        dpi=args.dpi,
        min_width=args.min_width,
        keep_pages=getattr(args, "keep_pages", False),
        searchable_pdf=args.searchable_pdf,
        ocrmypdf_lang=args.ocrmypdf_lang,
        engine=args.engine,
        detail=getattr(args, "detail", "auto"),
    )


def run_batch_command(args: argparse.Namespace) -> int:
    input_root = Path(args.input).expanduser().resolve()
    output_root = Path(args.output_dir).expanduser().resolve()
    ensure_output_dir(output_root)

    patterns = args.glob or default_batch_patterns(args.mode)
    files = iter_batch_inputs(input_root, recursive=args.recursive, patterns=patterns)
    manifest: list[dict[str, Any]] = []

    for path in files:
        rel = path.relative_to(input_root)
        file_output_dir = output_root / rel.with_suffix("")
        ensure_output_dir(file_output_dir)
        file_args = clone_args_for_input(args, args.mode, path, file_output_dir)
        try:
            if args.mode == "auto":
                run_auto_command(file_args)
            elif args.mode == "general":
                run_general_command(file_args)
            elif args.mode == "drawing":
                run_drawing_command(file_args)
            elif args.mode == "cad":
                run_cad_command(file_args)
            elif args.mode == "pdf":
                run_pdf_command(file_args)
            elif args.mode == "math":
                run_math_command(file_args)
            else:
                raise ValueError(f"Unsupported batch mode: {args.mode}")
            manifest.append(
                {
                    "input": str(path),
                    "mode": args.mode,
                    "engine": args.engine,
                    "status": "ok",
                    "output_dir": str(file_output_dir),
                }
            )
        except Exception as error:
            manifest.append(
                {
                    "input": str(path),
                    "mode": args.mode,
                    "engine": args.engine,
                    "status": "error",
                    "output_dir": str(file_output_dir),
                    "error": str(error),
                }
            )

    manifest_path = output_root / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "input_root": str(input_root),
                "mode": args.mode,
                "engine": args.engine,
                "patterns": patterns,
                "file_count": len(files),
                "results": manifest,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"MANIFEST: {manifest_path}")
    print(f"FILES: {len(files)}")
    return 0


def run_list_command() -> int:
    print("ocrx modes:")
    print("  auto     auto route between general OCR and formula OCR")
    print("  batch    batch OCR a directory")
    print("  general  images and screenshots")
    print("  pdf      scanned PDFs and searchable PDFs")
    print("  drawing  engineering drawings and drawing-heavy PDFs")
    print("  cad      DWG/DXF direct extraction with OCR fallback")
    print("  math     formula images to LaTeX")
    print("  doctor   environment health")
    print("engines:")
    print("  auto     prefer Paddle when its GPU backend is ready, otherwise RapidOCR")
    print("  rapid    force RapidOCR")
    print("  paddle   force PaddleOCR")
    return 0


def version_or_missing(module_name: str) -> str:
    try:
        module = __import__(module_name)
        return getattr(module, "__version__", "installed")
    except Exception:
        return "missing"


def run_doctor_command() -> int:
    prepare_onnxruntime()
    oda_converter = detect_oda_converter()
    tesseract_exe = resolve_tesseract_executable()
    ocrmypdf_exe = resolve_ocrmypdf_executable()
    print("ocrx doctor")
    print(f"python: {sys.executable}")
    print(f"rapidocr_onnxruntime: {version_or_missing('rapidocr_onnxruntime')}")
    print(f"rapid_latex_ocr: {version_or_missing('rapid_latex_ocr')}")
    print(f"paddleocr: {version_or_missing('paddleocr')}")
    print(f"paddlex: {version_or_missing('paddlex')}")
    print(f"paddlepaddle: {version_or_missing('paddle')}")
    print(f"ezdxf: {version_or_missing('ezdxf')}")
    print(f"matplotlib: {version_or_missing('matplotlib')}")
    print(f"onnxruntime: {version_or_missing('onnxruntime')}")
    print(f"onnx_providers: {ort.get_available_providers()}")
    print(f"gpu_ready: {ORT_GPU_READY}")
    print(f"paddle_available: {paddle_available()}")
    print(f"paddle_gpu_ready: {paddle_gpu_ready()}")
    print(f"auto_engine: {resolve_ocr_engine('auto')}")
    print(f"paddle_rec_batch_size: {get_paddle_rec_batch_size()}")
    print(f"drawing_prepare_workers: {get_drawing_prepare_workers()}")
    print(f"drawing_queue_size: {get_drawing_queue_size()}")
    print(f"cv2: {cv2.__version__}")
    print(f"pymupdf: {fitz.__doc__.split()[1]}")
    print(f"pdfplumber: {version_or_missing('pdfplumber')}")
    print(f"pypdf: {version_or_missing('pypdf')}")
    print(f"odafc: {str(oda_converter) if oda_converter else 'missing'}")
    print(f"tesseract: {tesseract_exe or 'missing'}")
    print(f"ocrmypdf: {ocrmypdf_exe or 'missing'}")
    print(f"pdftoppm: {shutil.which('pdftoppm') or 'missing'}")
    print(f"pdftotext: {shutil.which('pdftotext') or 'missing'}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "list":
        return run_list_command()
    if args.command == "doctor":
        return run_doctor_command()
    if args.command == "batch":
        return run_batch_command(args)
    if args.command == "general":
        return run_general_command(args)
    if args.command == "drawing":
        return run_drawing_command(args)
    if args.command == "cad":
        return run_cad_command(args)
    if args.command == "math":
        return run_math_command(args)
    if args.command == "pdf":
        return run_pdf_command(args)
    if args.command == "auto":
        return run_auto_command(args)
    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
