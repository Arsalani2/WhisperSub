# -*- coding: utf-8 -*-
# ==== WhisperSub Release Info (auto-generated) ====
# WhisperSub — نسخه v1.0.5
# تاریخ ساخت این ریلیز: 2026-10-26 | کامیت: 87ae20a
# ریلیز: https://github.com/Personalmo/WhisperSub/releases/tag/v1.0.5
# این نوت‌بوک منبع اصلی پروژه است؛ فایل src/whispersub.py به‌صورت خودکار از روی همین فایل ساخته می‌شود.
# ==== /WhisperSub Release Info ====

"""WhisperSub — رونویسی دسته‌ای ویدیو/صوت با OpenAI Whisper.

این نوت‌بوک منبع اصلی پروژه است؛ فایل whispersub.py به‌صورت خودکار از روی همین
فایل در ورکفلو ریلیز ساخته می‌شود. برای اجرا از Google Colab (ترجیحاً GPU)
یا Jupyter استفاده کنید و بخش‌ها را به ترتیب اجرا کنید.
"""

# @title 🛠️ ۱. نصب پیش‌نیازها و کتابخانه‌های مورد نیاز
# نکته (فیکس P3-8): نسخه‌ها پین شدند تا نصب بین اجراهای مختلف یکسان بماند
# (large-v3 از نسخه 20231117 این پکیج اضافه شده است؛ نسخهٔ پین‌شده جدیدترین نسخهٔ پایدار است)
!pip install -q "openai-whisper==20250625" "ffmpeg-python==0.2.0"


# @title ⚙️ ۲. پیکربندی محیط و فراخوانی مدل Whisper
import os
import torch
import whisper
from google.colab import drive

# --- اتصال به Google Drive ---
try:
    drive.mount('/content/drive')
except Exception as e:
    raise RuntimeError(f"اتصال به Google Drive ناموفق بود: {e}\nلطفاً دوباره تلاش کنید و دسترسی لازم را تأیید نمایید.") from e

# --- تشخیص دستگاه پردازشی ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[Info] دستگاه پردازشی: {device}")

gpu_mem_gb = None
if device == "cuda":
    try:
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        print(f"[Info] GPU: {gpu_name} | حافظه کل GPU: {gpu_mem_gb:.1f} GB")
    except Exception:
        pass
else:
    print("[Warning] GPU فعال نیست؛ پردازش روی CPU کندتر خواهد بود. (Runtime > Change runtime type > GPU)")

WHISPER_MODEL_NAME = "large-v3"

# --- زبان رونویسی (فیکس P3-2: به‌جای هاردکد 'en'؛ در transcribe و متادیتا از همین ثابت استفاده می‌شود) ---
TRANSCRIBE_LANGUAGE = "en"

# --- تخمین تقریبی VRAM لازم برای هر مدل (fp16) — فقط جهت هشدار پیشگیرانه، غیرقطعی ---
MODEL_VRAM_REQUIREMENTS_GB = {
    "tiny": 1.5, "base": 1.5, "small": 3.0, "medium": 6.0,
    "large": 11.0, "large-v2": 11.0, "large-v3": 11.0, "turbo": 7.0,
}
required_vram = MODEL_VRAM_REQUIREMENTS_GB.get(WHISPER_MODEL_NAME)
if device == "cuda" and gpu_mem_gb is not None and required_vram is not None:
    if gpu_mem_gb < required_vram:
        print(f"[Warning] حافظه GPU ({gpu_mem_gb:.1f} GB) ممکن است برای مدل «{WHISPER_MODEL_NAME}» (نیاز تقریبی {required_vram:.1f} GB) کافی نباشد.")
        print("[Warning] در صورت بروز خطای Out of Memory حین پردازش، همان فایل رد می‌شود و پردازش با فایل بعدی ادامه می‌یابد (جزئیات بیشتر در لاگ).")
        print("[Warning] راه‌حل جایگزین در صورت تکرار خطا: مدل 'turbo' یا 'medium'، یا GPU با حافظه بیشتر (Colab Pro / A100 / L4).")

# --- فیکس P3-6: هشدار اجرای CPU-only با مدل سنگین (فقط هشدار؛ مدل و رفتار برنامه تغییر نمی‌کند) ---
if device == "cpu" and WHISPER_MODEL_NAME in ("large", "large-v2", "large-v3", "turbo"):
    print(f"[Warning] مدل «{WHISPER_MODEL_NAME}» روی CPU بسیار کند است و حدود ~۱۰ گیگابایت رم می‌خواهد؛ رانتایم کولب ممکن است کند یا بی‌پاسخ شود.")
    print("[Warning] در صورت مشکل: Runtime > Change runtime type > GPU، یا مدل را به 'medium'/'small' کاهش دهید. در حال حاضر پردازش ادامه می‌یابد.")


# --- بارگذاری مدل Whisper ---
try:
    model = whisper.load_model(WHISPER_MODEL_NAME).to(device)
except torch.cuda.OutOfMemoryError as e:
    raise RuntimeError(
        f"بارگذاری مدل «{WHISPER_MODEL_NAME}» به دلیل کمبود حافظه GPU ناموفق بود.\n"
        f"پیشنهاد: مدل را به 'turbo' یا 'medium' تغییر دهید یا Runtime را ریست کرده و دوباره امتحان کنید.\n"
        f"جزئیات خطا: {e}"
    ) from e
except Exception as e:
    raise RuntimeError(f"بارگذاری مدل Whisper «{WHISPER_MODEL_NAME}» ناموفق بود: {e}") from e

MODEL_LABEL = WHISPER_MODEL_NAME.capitalize()
print(f"[Info] مدل Whisper «{MODEL_LABEL}» با موفقیت بارگذاری شد.")

# --- مسیر محلی سشن برای ذخیره فایل‌های WAV استخراج‌شده ---
# این مسیر فقط داخل سشن کولب است و هرگز به گوگل درایو منتقل نمی‌شود.
SESSION_AUDIO_DIR = "/content/session_audio_cache"
os.makedirs(SESSION_AUDIO_DIR, exist_ok=True)
print(f"[Info] فایل‌های WAV هر پروژه در همین مسیر سشن کولب ذخیره می‌شوند (نه گوگل درایو): {SESSION_AUDIO_DIR}")

# --- فیکس P3-4: WAV سشنِ فایلِ موفق پس از پایان همان فایل حذف می‌شود تا دیسک سشن پر نشود ---
# برای نگه‌داشتن WAVها در طول سشن این سوییچ را True کنید.
KEEP_SESSION_WAVS = False

# --- مسیر فایل لاگ کامل اجرای برنامه (در سشن کولب) ---
LOG_FILE_PATH = "/content/whisper_processing_log.txt"
print(f"[Info] گزارش کامل اجرای برنامه در این فایل ذخیره می‌شود: {LOG_FILE_PATH}")

# --- فیکس P3-9: نسخهٔ پشتیبان لاگ در گوگل درایو (فایل لاگ سشن با پایان سشن برای همیشه پاک می‌شود) ---
DRIVE_LOG_PATH = "/content/drive/MyDrive/WhisperSub_Logs/whisper_processing_log.txt"
print(f"[Info] در پایان هر پردازش دسته‌ای، نسخهٔ پشتیبان لاگ در گوگل درایو ذخیره می‌شود: {DRIVE_LOG_PATH}")

# --- پسوندهای رسانهٔ پشتیبانی‌شده (فیکس P4-2: ثابت واحد در همین بخش؛ build_pairs_from_folder و extract_audio از همین استفاده می‌کنند) ---
VIDEO_EXT = ('.mp4', '.mkv', '.avi', '.mov', '.webm', '.wmv', '.flv', '.ts')
AUDIO_EXT = ('.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.amr', '.aiff', '.wma', '.alac')

# --- فایل‌های ورودی و پوشهٔ خروجی هرکدام (فیکس P2-2: مسیرهای شخصی از ریپوی عمومی حذف شدند) ---
# هر جفت به شکل (مسیر فایل ورودی، پوشهٔ خروجی در گوگل درایو) است. مثال:
# input_output_pairs = [
#     ('/content/drive/MyDrive/Videos/lesson1.mp4', '/content/drive/MyDrive/Subs/lesson1'),
# ]
# یا ساخت خودکار از روی یک پوشه (تابع build_pairs_from_folder در بخش ۳ تعریف می‌شود):
# input_output_pairs = build_pairs_from_folder('/content/drive/MyDrive/Videos', '/content/drive/MyDrive/Subs')
input_output_pairs = []  # مسیرهای خودتان را اینجا (یا با build_pairs_from_folder) تنظیم کنید


# @title 📧 ۳. تنظیمات سامانه اطلاع‌رسانی ایمیل و اعتبارسنجی ورودی‌ها

from google.colab import userdata
import smtplib

def get_secret(name):
    """خواندن امن Secret از Colab؛ اگر وجود نداشت یا در دسترس نبود None برمی‌گرداند (فیکس P1-1)."""
    try:
        return userdata.get(name)
    except Exception:
        return None

SENDER_EMAIL = get_secret('SENDER_EMAIL')
SENDER_APP_PASSWORD = get_secret('SENDER_APP_PASSWORD')
RECEIVER_EMAIL = get_secret('RECEIVER_EMAIL')

def build_pairs_from_folder(source_folder, destination_root):
    """برای هر فایل رسانه‌ای داخل source_folder یک زوج (ورودی، خروجی) در destination_root می‌سازد."""
    # فیکس P4-2: از ثابت واحد بخش ۲ به‌جای media_ext محلیِ تکراری استفاده می‌شود
    pairs = []
    # فیکس P4-7: لیست خالی دیگر بی‌صدا برنمی‌گردد؛ علت (پوشهٔ ناموجود / نبود فایل رسانه‌ای) هشدار داده می‌شود
    if not os.path.isdir(source_folder):
        print(f"[Warning] پوشهٔ منبع یافت نشد یا پوشه نیست: {source_folder} — لیست فایل‌ها خالی خواهد بود.")
        return pairs
    for f in sorted(os.listdir(source_folder)):
        full_path = os.path.join(source_folder, f)
        if os.path.isfile(full_path) and f.lower().endswith(VIDEO_EXT + AUDIO_EXT):
            out_folder = os.path.join(destination_root, os.path.splitext(f)[0])
            pairs.append((full_path, out_folder))
    if not pairs:
        print(f"[Warning] هیچ فایل رسانه‌ای با پسوندهای پشتیبانی‌شده در «{source_folder}» پیدا نشد (فقط خود پوشه بررسی می‌شود، نه زیرپوشه‌ها).")
    return pairs

def validate_pairs(pairs):
    """فیکس P4-7: اعتبارسنجی ساختار pairs؛ لیست جفت‌های معتبر را برمی‌گرداند و موارد نامعتبر را هشدار می‌دهد."""
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(f"input_output_pairs باید لیستی از جفت‌های (مسیر ورودی، پوشهٔ خروجی) باشد؛ نوع دریافتی: {type(pairs).__name__}")
    valid = []
    seen_outputs = {}
    for i, item in enumerate(pairs, start=1):
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, str) and x.strip() for x in item)):
            print(f"[Warning] آیتم شمارهٔ {i} در input_output_pairs معتبر نیست (باید جفت دو رشتهٔ غیرخالی باشد) و رد می‌شود: {item!r}")
            continue
        in_path, out_folder = item
        if not in_path.lower().endswith(VIDEO_EXT + AUDIO_EXT):
            print(f"[Warning] پسوند فایل «{os.path.basename(in_path)}» جزو پسوندهای پشتیبانی‌شده نیست؛ در صورت ناموفق بودن ffmpeg، فایل رد می‌شود.")
        if out_folder in seen_outputs:
            print(f"[Warning] پوشهٔ خروجی «{out_folder}» برای چند فایل تکراری است؛ خروجی‌ها یکدیگر را بازنویسی می‌کنند (فایل‌های {seen_outputs[out_folder]} و {i}).")
        else:
            seen_outputs[out_folder] = i
        valid.append((in_path, out_folder))
    return valid

if SENDER_EMAIL and SENDER_APP_PASSWORD and RECEIVER_EMAIL:
    print(f"[Info] ایمیل فرستنده تنظیم شد: {SENDER_EMAIL}")
    print(f"[Info] ایمیل گیرنده: {RECEIVER_EMAIL}")
    print("[Info] در حال آزمایش اتصال به سرور SMTP جیمیل...")
    try:
        with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=20) as server:
            server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
        print("[Info] اتصال و احراز هویت ایمیل با موفقیت تأیید شد. ✅")
    except smtplib.SMTPAuthenticationError as e:
        print(f"[Warning] احراز هویت ایمیل ناموفق بود؛ App Password را بررسی کنید. جزئیات: {e}")
    except Exception as e:
        print(f"[Warning] آزمایش اتصال ایمیل ناموفق بود (پردازش ادامه می‌یابد اما ایمیل‌ها ممکن است ارسال نشوند): {e}")
else:
    print("[Warning] تنظیمات ایمیل در Secrets یافت نشد؛ برنامه بدون اعلان ایمیلی ادامه می‌یابد. برای فعال‌سازی، SENDER_EMAIL و SENDER_APP_PASSWORD و RECEIVER_EMAIL را در Secrets کولب اضافه کنید.")

print(f"\n[Info] تعداد فایل‌های ثبت‌شده برای پردازش: {len(input_output_pairs)}")
missing = [p for p, _ in input_output_pairs if not os.path.exists(p)]
if missing:
    print(f"[Warning] {len(missing)} فایل از لیست یافت نشدند (در زمان پردازش رد خواهند شد):")
    for m in missing:
        print(f"   - {m}")
else:
    print("[Info] همه فایل‌های ورودی موجود هستند. ✅")


# @title 🚀 ۴. اجرای موتور پردازشگر Whisper (تبدیل گفتار به متن و تولید خروجی‌های چندگانه)

import shutil
import tempfile
import time
import zipfile
import traceback
import contextlib
import types
import tqdm
import ffmpeg
from email.mime.text import MIMEText

# ============================================================
# سیستم لاگینگ پیشرفته
# ============================================================
class ColabLogger:
    ICONS = {"INFO": "ℹ️ ", "SUCCESS": "✅ ", "WARNING": "⚠️ ", "ERROR": "❌ ", "STEP": "🔹 "}

    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.start_time = time.time()
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(f"\n{'='*70}\nشروع اجرای جدید در {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*70}\n")
        except Exception as e:
            print(f"[Warning] امکان نوشتن فایل لاگ نبود: {e}")

    def _elapsed(self):
        e = time.time() - self.start_time
        h, rem = divmod(e, 3600)
        m, s = divmod(rem, 60)
        return f"{int(h):02}:{int(m):02}:{int(s):02}"

    def log(self, msg, level="INFO"):
        icon = self.ICONS.get(level, "")
        line = f"[{time.strftime('%H:%M:%S')}] [+{self._elapsed()}] {icon}{msg}"
        print(line)
        try:
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except Exception:
            pass

    def info(self, msg): self.log(msg, "INFO")
    def success(self, msg): self.log(msg, "SUCCESS")
    def warning(self, msg): self.log(msg, "WARNING")
    def error(self, msg): self.log(msg, "ERROR")
    def step(self, msg): self.log(msg, "STEP")

logger = ColabLogger(LOG_FILE_PATH)

# ============================================================
# توابع کمکی
# ============================================================
def format_srt_time(seconds):
    # فیکس P3-1: گردکردن صحیح (به‌جای truncate) برای حذف خطای تا ۱ms در تایم‌استمپ
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem_ms = divmod(total_ms, 3600000)
    m, rem_ms = divmod(rem_ms, 60000)
    s, ms = divmod(rem_ms, 1000)
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

def probe_media_duration(media_path):
    """مدت زمان دقیق فایل رسانه را با ffprobe برمی‌گرداند (ثانیه) یا None در صورت خطا."""
    try:
        info = ffmpeg.probe(media_path)
        return float(info['format']['duration'])
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logger.warning(f"عدم موفقیت ffprobe در تشخیص مدت‌زمان: {stderr.strip().splitlines()[-1] if stderr.strip() else e}")
        return None
    except (KeyError, ValueError) as e:
        logger.warning(f"اطلاعات مدت‌زمان در خروجی ffprobe یافت نشد: {e}")
        return None
    except Exception as e:
        logger.warning(f"خطای غیرمنتظره هنگام بررسی مدت‌زمان فایل: {e}")
        return None

PROGRESS_LOG_INTERVAL_SEC = 30  # فاصلهٔ ثبت پیشرفت رونویسی در لاگ (ثانیه)

class _LoggingTqdm(tqdm.tqdm):
    """فیکس P4-6: نوار پیشرفت داخلی Whisper را حفظ می‌کند و هر چند ثانیه درصد پیشرفت را در لاگ هم می‌نویسد."""
    _last_log = 0.0

    def update(self, n=1):
        result = super().update(n)
        now = time.time()
        if self.total and now - self._last_log >= PROGRESS_LOG_INTERVAL_SEC:
            self._last_log = now
            logger.info(f"پیشرفت رونویسی: {min(100.0, self.n / self.total * 100):.0f}٪")
        return result

@contextlib.contextmanager
def transcribe_progress_logging():
    """جایگزینی موقت tqdm داخل whisper.transcribe با نسخهٔ لاگ‌کننده؛ پس از اتمام همیشه به حالت اولیه برمی‌گردد."""
    import whisper.transcribe as _wt
    original = getattr(_wt, "tqdm", None)
    patched = False
    try:
        _wt.tqdm = types.SimpleNamespace(tqdm=_LoggingTqdm)
        patched = True
    except Exception as e:
        logger.warning(f"فعال‌سازی لاگ پیشرفت رونویسی ممکن نشد (رونویسی بدون آن ادامه می‌یابد): {e}")
    try:
        yield
    finally:
        if patched:
            _wt.tqdm = original

def extract_audio(media_path, base_filename):
    """صدا را استخراج و در مسیر سشن کولب (نه درایو) ذخیره می‌کند."""
    if not os.path.exists(media_path):
        raise FileNotFoundError(f"فایل ورودی یافت نشد: {media_path}")
    if not media_path.lower().endswith(VIDEO_EXT + AUDIO_EXT):
        raise ValueError(f"فرمت فایل پشتیبانی نمی‌شود: {media_path}")

    safe_name = "".join(c for c in base_filename if c not in '\\/:*?"<>|').strip() or "audio"
    audio_path = os.path.join(SESSION_AUDIO_DIR, f"{safe_name}.wav")
    if os.path.exists(audio_path):
        logger.warning("فایل صوتی هم‌نام از قبل در سشن کولب وجود دارد؛ نام جدید تولید می‌شود.")
        counter = 1
        while os.path.exists(audio_path):
            audio_path = os.path.join(SESSION_AUDIO_DIR, f"{safe_name}_{counter}.wav")
            counter += 1

    try:
        (
            ffmpeg
            .input(media_path)
            # فیکس P3-3: خروجی مونو 16kHz — Whisper خودش صدای ورودی را به 16kHz مونو
            # تبدیل می‌کند؛ نرخ منبع/استریو حدود ۱۰ برابر حجیم‌تر از نیاز است و فقط
            # دیسک سشن کولب را پر می‌کند (نزدیک به ۱۱۵ مگابایت در ساعت کافی است).
            .output(audio_path, ar=16000, ac=1)
            .overwrite_output()
            .run(quiet=True, capture_stdout=True, capture_stderr=True)
        )
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        raise RuntimeError(f"خطای ffmpeg در استخراج صدا: {stderr.strip().splitlines()[-1] if stderr.strip() else e}")

    if not os.path.exists(audio_path) or os.path.getsize(audio_path) == 0:
        raise RuntimeError("فایل صوتی استخراج‌شده خالی یا نامعتبر است.")

    return audio_path

def save_all_formats(segments, base_filename, work_dir, media_path, model_label, media_duration, audio_path, processing_seconds):
    suffix = f"- Whisper [{model_label} Model]"
    names = {
        'srt': f"{base_filename}{suffix}.srt",
        'txt': f"{base_filename}{suffix}.txt",
        'txt_tl': f"{base_filename}{suffix}_timeline.txt",
        'meta': f"{base_filename}_Metadata.txt",
        'zip': f"{base_filename}_Whisper_Transcription.zip",
    }
    paths = {k: os.path.join(work_dir, v) for k, v in names.items()}

    total_chars = 0
    line_count = 0  # فیکس P4-5: فقط سگمنت‌های دارای متن شمرده می‌شوند
    with open(paths['srt'], 'w', encoding='utf-8') as srt_f, \
          open(paths['txt'], 'w', encoding='utf-8') as txt_f, \
          open(paths['txt_tl'], 'w', encoding='utf-8') as txt_tl_f:
        for i, seg in enumerate(segments):
            text = seg['text'].strip()
            # فیکس P4-5: سگمنت با متن خالی کیوی خالی در SRT تولید می‌کرد؛ رد می‌شود
            # (شماره‌گذاری از enumerate حفظ می‌شود)
            if not text:
                continue
            start_s, end_s = seg['start'], seg['end']
            line_no = i + 1
            start_srt, end_srt = format_srt_time(start_s), format_srt_time(end_s)
            srt_f.write(f"{line_no}\n{start_srt} --> {end_srt}\n{text}\n\n")
            txt_f.write(f"{text}\n")
            txt_tl_f.write(f"{line_no}\n{start_srt} --> {end_srt}\n{text}\n\n")
            total_chars += len(text)
            line_count += 1

    timeline_start = segments[0]['start'] if segments else 0.0
    timeline_end = segments[-1]['end'] if segments else 0.0
    total_words = sum(len(seg['text'].strip().split()) for seg in segments)
    avg_chars_per_line = (total_chars / line_count) if line_count else 0
    duration_for_report = media_duration if media_duration is not None else timeline_end

    source_size_mb = os.path.getsize(media_path) / (1024 * 1024) if os.path.exists(media_path) else None
    audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024) if audio_path and os.path.exists(audio_path) else None

    with open(paths['meta'], 'w', encoding='utf-8') as meta_f:
        meta_f.write("========================================\n")
        meta_f.write("        گزارش کامل متادیتای رونویسی\n")
        meta_f.write("========================================\n\n")
        meta_f.write(f"نام فایل ورودی: {os.path.basename(media_path)}\n")
        meta_f.write(f"مسیر کامل فایل ورودی: {media_path}\n")
        meta_f.write(f"حجم فایل ورودی: {(f'{source_size_mb:.2f} مگابایت' if source_size_mb is not None else 'نامشخص')}\n")
        meta_f.write(f"حجم فایل صوتی استخراج‌شده: {(f'{audio_size_mb:.2f} مگابایت' if audio_size_mb is not None else 'نامشخص')}\n")
        meta_f.write(f"مسیر فایل صوتی (فقط داخل سشن کولب - در درایو ذخیره نمی‌شود): {audio_path}\n\n")

        meta_f.write("--- اطلاعات تایم‌لاین ---\n")
        meta_f.write(f"شروع تایم‌لاین: {format_srt_time(timeline_start)}\n")
        meta_f.write(f"پایان تایم‌لاین: {format_srt_time(timeline_end)}\n")
        meta_f.write(f"کل مدت‌زمان ویدیو/رسانه: {format_srt_time(duration_for_report)}\n")
        if media_duration is not None and abs(media_duration - timeline_end) > 1.0:
            meta_f.write(f"توجه: {abs(media_duration - timeline_end):.1f} ثانیه اختلاف بین پایان تایم‌لاین و مدت واقعی رسانه وجود دارد (احتمالاً به دلیل حذف سکوت یا نویز).\n")
        meta_f.write("\n")

        meta_f.write("--- آمار متن ---\n")
        meta_f.write(f"تعداد کل لاین‌ها (Segments): {line_count}\n")
        meta_f.write(f"تعداد کل کاراکترها: {total_chars}\n")
        meta_f.write(f"تعداد کل کلمات: {total_words}\n")
        meta_f.write(f"میانگین کاراکتر در هر لاین: {avg_chars_per_line:.1f}\n\n")

        meta_f.write("--- اطلاعات پردازش ---\n")
        meta_f.write(f"مدل Whisper استفاده‌شده: {model_label}\n")
        meta_f.write(f"دستگاه پردازشی: {device}\n")
        meta_f.write(f"زبان رونویسی: {TRANSCRIBE_LANGUAGE}\n")
        meta_f.write(f"زمان صرف‌شده برای پردازش این فایل: {processing_seconds:.1f} ثانیه\n")
        meta_f.write(f"تاریخ و ساعت پردازش: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

        meta_f.write("--- فایل‌های تولیدشده در این بسته ---\n")
        # فیکس P4-3: نام ZIP نیز در فهرست «فایل‌های تولیدشده» ثبت می‌شود
        for key in ['srt', 'txt', 'txt_tl', 'zip']:
            meta_f.write(f"- {names[key]}\n")

    logger.step(f"متادیتای کامل تولید شد: {line_count} لاین | تایم‌لاین {format_srt_time(timeline_start)} تا {format_srt_time(timeline_end)} | کل مدت‌زمان {format_srt_time(duration_for_report)}")

    # فیکس P4-3: فشرده‌سازی واقعی به‌جای ذخیرهٔ بدون فشرده‌سازی (ZIP_STORED) پیش‌فرض
    with zipfile.ZipFile(paths['zip'], 'w', compression=zipfile.ZIP_DEFLATED) as zf:
        for key, p in paths.items():
            if key != 'zip' and os.path.exists(p):
                zf.write(p, os.path.basename(p))

    return {
        'line_count': line_count,
        'timeline_start': timeline_start,
        'timeline_end': timeline_end,
        'media_duration': duration_for_report,
        'total_chars': total_chars,
        'total_words': total_words,
    }

def send_email(subject, body, max_retries=3):
    if not (SENDER_EMAIL and SENDER_APP_PASSWORD and RECEIVER_EMAIL):
        logger.warning("اطلاعات ایمیل کامل نیست؛ این ایمیل ارسال نشد.")
        return False

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['Subject'] = subject
    msg['From'] = SENDER_EMAIL
    msg['To'] = RECEIVER_EMAIL

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP_SSL('smtp.gmail.com', 465, timeout=30) as server:
                server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
                server.sendmail(SENDER_EMAIL, [RECEIVER_EMAIL], msg.as_string())
            logger.success(f"ایمیل «{subject}» ارسال شد.")
            return True
        except smtplib.SMTPAuthenticationError as e:
            # فیکس P3-7: تلاش مجدد روی خطای احراز هویت بی‌فایده است؛ فوراً رد می‌شود
            logger.error(f"خطای احراز هویت ایمیل: App Password یا SENDER_EMAIL را بررسی کنید. {e}")
            logger.error("تلاش مجدد برای خطای احراز هویت بی‌فایده است؛ ارسال این ایمیل رد شد.")
            return False
        except smtplib.SMTPException as e:
            logger.error(f"خطای SMTP هنگام ارسال ایمیل (تلاش {attempt}/{max_retries}): {e}")
        except (OSError, TimeoutError) as e:
            logger.error(f"خطای شبکه هنگام ارسال ایمیل (تلاش {attempt}/{max_retries}): {e}")
        except Exception as e:
            logger.error(f"خطای غیرمنتظره هنگام ارسال ایمیل (تلاش {attempt}/{max_retries}): {e}")

        if attempt < max_retries:
            wait_s = 5 * attempt
            logger.info(f"تلاش مجدد برای ارسال ایمیل ظرف {wait_s} ثانیه...")
            time.sleep(wait_s)

    logger.error(f"ارسال ایمیل «{subject}» پس از {max_retries} تلاش ناموفق بود.")
    return False

def backup_log_to_drive():
    """کپی فایل لاگ سشن به گوگل درایو تا با پایان سشن از بین نرود (فیکس P3-9)."""
    try:
        if os.path.exists(LOG_FILE_PATH):
            os.makedirs(os.path.dirname(DRIVE_LOG_PATH), exist_ok=True)
            shutil.copy2(LOG_FILE_PATH, DRIVE_LOG_PATH)
            return True
    except Exception as e:
        logger.warning(f"کپی لاگ به گوگل درایو ناموفق بود: {e}")
    return False

def read_log_tail(max_lines=40):
    """آخرین خط‌های فایل لاگ را برای الصاق به ایمیل برمی‌گرداند (فیکس P3-9)."""
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            lines = f.readlines()
        return "".join(lines[-max_lines:]).strip()
    except Exception:
        return ""

# ============================================================
# پردازش دسته‌ای اصلی
# ============================================================
def estimate_eta_seconds(remaining_media_durations, done_records, done_proc_times):
    """فیکس P4-10: تخمین زمان باقی‌مانده.
    اگر مدت همهٔ فایل‌های باقی‌مانده معلوم باشد: نرخ پردازش (ثانیهٔ پردازش به‌ازای ثانیهٔ رسانه) × مجموع مدت باقی‌مانده.
    در غیر این صورت: median زمان پردازش فایل‌های انجام‌شده × تعداد باقی‌مانده (median در برابر یک فایل طولانی مقاوم است).
    اگر هنوز هیچ فایل موفقی نیست None برمی‌گرداند."""
    if not done_proc_times:
        return None
    total_media = sum(m for _, m in done_records)
    if done_records and total_media > 0 and all(d is not None for d in remaining_media_durations):
        rate = sum(p for p, _ in done_records) / total_media
        return rate * sum(remaining_media_durations)
    ordered = sorted(done_proc_times)
    mid = len(ordered) // 2
    median_time = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    return median_time * len(remaining_media_durations)

def process_all(pairs):
    # فیکس P4-7: اعتبارسنجی ساختار pairs و هشدار لیست خالی پیش از شروع
    pairs = validate_pairs(pairs)
    total = len(pairs)
    results = []
    file_durations = []
    done_records = []  # فیکس P4-10: (زمان پردازش، مدت رسانه) برای فایل‌های موفق با مدت معلوم
    success_count = 0
    fail_count = 0
    batch_start = time.time()

    if total == 0:
        logger.warning("لیست فایل‌های ورودی خالی است؛ چیزی برای پردازش وجود ندارد. input_output_pairs را تنظیم کنید (یا از build_pairs_from_folder استفاده کنید).")
        return

    logger.info(f"====== شروع پردازش {total} فایل ======")

    missing_files = [p for p, _ in pairs if not os.path.exists(p)]
    if missing_files:
        logger.warning(f"{len(missing_files)} فایل از {total} فایل ورودی یافت نشد و هنگام پردازش رد خواهند شد.")

    # فیکس P4-10: مدت رسانهٔ همهٔ فایل‌ها یک‌بار از قبل خوانده می‌شود (ffprobe سریع است) تا ETA بر پایهٔ مدت واقعی باشد
    media_durations = {}
    for p, _ in pairs:
        if p not in media_durations and os.path.exists(p):
            media_durations[p] = probe_media_duration(p)

    for idx, (input_path, output_folder) in enumerate(pairs, start=1):
        base_filename = os.path.splitext(os.path.basename(input_path))[0]
        logger.info(f"------ فایل {idx}/{total} ({(idx-1)/total*100:.0f}٪ انجام‌شده) : {base_filename} ------")

        eta_seconds = estimate_eta_seconds(
            [media_durations.get(p) for p, _ in pairs[idx - 1:]], done_records, file_durations)
        if eta_seconds is not None:
            eta_h, eta_rem = divmod(eta_seconds, 3600)
            eta_m, _ = divmod(eta_rem, 60)
            logger.info(f"زمان تقریبی باقی‌مانده تا پایان کل عملیات: {int(eta_h)} ساعت و {int(eta_m)} دقیقه")

        file_start = time.time()
        work_dir = tempfile.mkdtemp()
        audio_path = None
        status = "نامشخص"
        error_detail = ""
        stats = {}
        duration = 0.0

        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"فایل ورودی یافت نشد: {input_path}")

            logger.step("در حال بررسی مدت‌زمان دقیق فایل رسانه...")
            media_duration = media_durations.get(input_path)  # فیکس P4-10: از نتیجهٔ ffprobe پیش از حلقه استفاده می‌شود
            if media_duration is not None:
                logger.info(f"مدت رسانه: {format_srt_time(media_duration)[:8]}")

            # فیکس P3-4: گارد فضای دیسک سشن قبل از استخراج صدا
            free_gb = shutil.disk_usage('/').free / (1024 ** 3)
            if free_gb < 2.0:
                raise RuntimeError(f"فضای خالی دیسک سشن کولب کمتر از 2 گیگابایت است ({free_gb:.1f} GB)؛ ابتدا فضا آزاد کنید و دوباره تلاش کنید.")
            logger.step("در حال استخراج صدا (WAV در سشن کولب ذخیره می‌شود)...")
            audio_path = extract_audio(input_path, base_filename)
            logger.success(f"صدا استخراج شد و در سشن کولب ذخیره ماند (نه درایو): {audio_path}")

            logger.step("در حال رونویسی با Whisper (بسته به طول فایل ممکن است چند دقیقه طول بکشد)...")
            # فیکس P4-6: پیشرفت رونویسی به‌صورت دوره‌ای در لاگ (و فایل لاگ) ثبت می‌شود
            with transcribe_progress_logging():
                result = model.transcribe(
                    audio_path,
                    language=TRANSCRIBE_LANGUAGE,
                    verbose=False,
                    fp16=(device == "cuda"),
                    condition_on_previous_text=False,
                )
            segments = result['segments']
            if not segments:
                raise ValueError("هیچ گفتاری در فایل شناسایی نشد.")
            logger.success(f"رونویسی کامل شد: {len(segments)} لاین شناسایی شد.")

            logger.step("در حال تولید فرمت‌های خروجی (SRT/TXT) و متادیتا...")
            stats = save_all_formats(segments, base_filename, work_dir, input_path,
                                      MODEL_LABEL, media_duration, audio_path,
                                      time.time() - file_start)

            logger.step(f"در حال انتقال فایل‌های خروجی به گوگل درایو: {output_folder}")
            os.makedirs(output_folder, exist_ok=True)
            for fname in os.listdir(work_dir):
                dest_path = os.path.join(output_folder, fname)
                if os.path.exists(dest_path):  # فیکس P2-1: اجرای مجدد روی خروجی موجود خطا نمی‌دهد
                    os.remove(dest_path)
                    logger.info(f"خروجی قبلی بازنویسی شد: {fname}")
                shutil.move(os.path.join(work_dir, fname), dest_path)

            duration = time.time() - file_start
            file_durations.append(duration)
            if media_duration:
                done_records.append((duration, media_duration))  # فیکس P4-10
            status = "موفق"
            success_count += 1
            logger.success(f"فایل «{base_filename}» با موفقیت پردازش شد. ({duration:.1f} ثانیه)")
            results.append((input_path, f"موفق ({duration:.1f}s, {stats.get('line_count', '?')} لاین)"))

        except FileNotFoundError as e:
            duration = time.time() - file_start
            status, error_detail = "ناموفق", f"{type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
            fail_count += 1
            logger.error(f"فایل یافت نشد: {e}")
            results.append((input_path, f"ناموفق - فایل یافت نشد ({str(e)[:120]})"))
        except torch.cuda.OutOfMemoryError as e:
            duration = time.time() - file_start
            status, error_detail = "ناموفق", f"کمبود حافظه GPU (Out of Memory) — {type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
            fail_count += 1
            logger.error(f"خطای کمبود حافظه GPU حین پردازش «{base_filename}»؛ این فایل رد شد و پردازش با فایل بعدی ادامه می‌یابد.")
            try:
                torch.cuda.empty_cache()
                logger.info("حافظه CUDA پاک‌سازی شد.")
            except Exception:
                pass
            results.append((input_path, "ناموفق - کمبود حافظه GPU (OOM)"))
        except (ValueError, RuntimeError) as e:
            duration = time.time() - file_start
            status, error_detail = "ناموفق", f"{type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
            fail_count += 1
            logger.error(f"خطا در پردازش فایل: {e}")
            results.append((input_path, f"ناموفق - {type(e).__name__}: {str(e)[:120]}"))
        except Exception as e:
            duration = time.time() - file_start
            status, error_detail = "ناموفق", f"{type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
            fail_count += 1
            logger.error(f"خطای غیرمنتظره: {type(e).__name__}: {e} (جزئیات کامل در فایل لاگ ذخیره شد)")
            try:
                with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
                    f.write(traceback.format_exc() + "\n")
            except Exception:
                pass
            results.append((input_path, f"ناموفق - {type(e).__name__}: {str(e)[:120]}"))
        finally:
            shutil.rmtree(work_dir, ignore_errors=True)
            # فیکس P3-4: WAV سشنِ فایل موفق حذف می‌شود تا در بچ‌های بزرگ دیسک سشن پر نشود
            if audio_path and status == "موفق" and not KEEP_SESSION_WAVS:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass
            if device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass

        logger.info(f"خلاصه تا این لحظه: {success_count} موفق، {fail_count} ناموفق از {idx} فایل بررسی‌شده")

        per_file_body = (
            f"گزارش پردازش فایل {idx} از {total}\n"
            f"{'=' * 40}\n"
            f"نام فایل: {base_filename}\n"
            f"وضعیت: {status}\n"
        )
        if status == "موفق":
            per_file_body += (
                f"مدت‌زمان پردازش: {duration:.1f} ثانیه\n"
                f"تعداد لاین‌ها: {stats.get('line_count', '-')}\n"
                f"شروع تایم‌لاین: {format_srt_time(stats.get('timeline_start', 0))}\n"
                f"پایان تایم‌لاین: {format_srt_time(stats.get('timeline_end', 0))}\n"
                f"کل مدت‌زمان رسانه: {format_srt_time(stats.get('media_duration', 0))}\n"
                f"مسیر خروجی در گوگل درایو: {output_folder}\n"
                f"مسیر فایل WAV (فقط سشن کولب): {audio_path}\n"
            )
        else:
            per_file_body += f"جزئیات خطا: {error_detail}\n"
        per_file_body += f"\nپیشرفت کلی پروژه: {idx}/{total} فایل ({idx/total*100:.0f}٪)\n"

        send_email(f"{'✅' if status == 'موفق' else '❌'} گزارش فایل {idx}/{total}: {base_filename}", per_file_body)

    total_time = time.time() - batch_start
    th, trem = divmod(total_time, 3600)
    tm, ts = divmod(trem, 60)

    summary_header = (
        f"گزارش نهایی پردازش دسته‌ای\n"
        f"{'=' * 40}\n"
        f"تعداد کل فایل‌ها: {total}\n"
        f"موفق: {success_count}   |   ناموفق: {fail_count}\n"
        f"زمان کل پردازش: {int(th)} ساعت {int(tm)} دقیقه {int(ts)} ثانیه\n"
    )
    if file_durations:
        avg_all = sum(file_durations) / len(file_durations)
        summary_header += f"میانگین زمان پردازش هر فایل: {avg_all:.1f} ثانیه\n"
    summary_header += f"\n{'-' * 40}\nجزئیات هر فایل:\n{'-' * 40}\n"
    summary = summary_header + "\n".join(f"{os.path.basename(p)}: {s}" for p, s in results)

    logger.info("====== پردازش تمام فایل‌ها به پایان رسید ======")
    logger.info(f"موفق: {success_count} | ناموفق: {fail_count} | زمان کل: {int(th):02}:{int(tm):02}:{int(ts):02}")

    try:
        cache_files = os.listdir(SESSION_AUDIO_DIR)
        cache_size_mb = sum(os.path.getsize(os.path.join(SESSION_AUDIO_DIR, f)) for f in cache_files) / (1024 * 1024)
        logger.info(f"فایل‌های WAV این سشن ({len(cache_files)} فایل، مجموعاً {cache_size_mb:.1f} مگابایت) در مسیر زیر باقی ماندند و به درایو منتقل نمی‌شوند:")
        logger.info(f"  {SESSION_AUDIO_DIR}")
        logger.warning("این فایل‌ها با پایان یا ریست‌شدن سشن کولب برای همیشه پاک خواهند شد.")
    except Exception:
        pass

    # فیکس P3-9: لاگ سشن با پایان سشن نابود می‌شود؛ نسخهٔ درایو + دنبالهٔ لاگ در ایمیل نهایی
    if backup_log_to_drive():
        logger.info(f"نسخهٔ پشتیبان لاگ در گوگل درایو ذخیره شد: {DRIVE_LOG_PATH}")
        summary += f"\n\nنسخهٔ پشتیبان لاگ کامل در گوگل درایو: {DRIVE_LOG_PATH}"
    else:
        logger.warning("نسخهٔ پشتیبان لاگ در گوگل درایو ساخته نشد؛ متن لاگ فقط داخل سشن باقی می‌ماند.")
    log_tail = read_log_tail()
    if log_tail:
        summary += f"\n\n{'=' * 40}\nآخرین بخش فایل لاگ:\n{'=' * 40}\n{log_tail}"

    send_email(f"📊 گزارش نهایی پردازش دسته‌ای ({success_count}/{total} موفق)", summary)

    return results


# @title 🏁 ۵. اجرای پردازش دسته‌ای (برای تلاش مجدد فقط همین سلول کافی است)

try:
    process_all(input_output_pairs)
except KeyboardInterrupt:
    logger.error("پردازش توسط کاربر متوقف شد.")
    send_email("⛔ پردازش متوقف شد", "پردازش دسته‌ای توسط کاربر متوقف شد (KeyboardInterrupt).")
except Exception as e:
    logger.error(f"خطای بحرانی و غیرمنتظره در سطح اصلی برنامه: {e}")
    try:
        with open(LOG_FILE_PATH, 'a', encoding='utf-8') as f:
            f.write(traceback.format_exc() + "\n")
    except Exception:
        pass
    # فیکس P3-9 و P3-10: نسخهٔ پشتیبان لاگ + نوع خطا + آخرین خطوط لاگ در خود ایمیل بحرانی
    backup_log_to_drive()
    critical_tail = read_log_tail(30)
    send_email("🚨 خطای بحرانی در اجرای برنامه", f"برنامه با خطای غیرمنتظره متوقف شد:\n{type(e).__name__}: {e}\n\nجزئیات کامل خطا در فایل لاگ ذخیره شد؛ نسخهٔ پشتیبان: {DRIVE_LOG_PATH}\n\n{'=' * 40}\n{critical_tail}")
    raise
