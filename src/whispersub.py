# -*- coding: utf-8 -*-
# ==== WhisperSub Release Info (auto-generated) ====
# WhisperSub — نسخه v1.0.6
# تاریخ ساخت این ریلیز: 2026-10-26 | کامیت: 19dde50
# ریلیز: https://github.com/Personalmo/WhisperSub/releases/tag/v1.0.6
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

# فیکس FB-21: قبلاً شکست !pip بی‌صدا رد می‌شد و اولین نشانه، ModuleNotFoundError
# گمراه‌کننده در سلول بعدی بود؛ حالا نصب بلافاصله پس از pip بررسی می‌شود
import importlib
for _pkg in ("whisper", "ffmpeg"):
    try:
        importlib.import_module(_pkg)
    except ImportError as e:
        raise RuntimeError(
            f"نصب پیش‌نیازها ناموفق بود (بستهٔ {_pkg}): {e}\n"
            "خروجی pip در همین سلول را بررسی کنید (شبکه یا تضاد وابستگی‌ها) و سلول را دوباره اجرا کنید."
        ) from e
print("[Info] نصب پیش‌نیازها با موفقیت بررسی شد (whisper, ffmpeg).")


# @title ⚙️ ۲. پیکربندی محیط و فراخوانی مدل Whisper
import os
import torch
import whisper

# فیکس FB-24: در نسخه‌های قدیمی torch، torch.cuda.OutOfMemoryError وجود ندارد و
# خودِ except با AttributeError کرش می‌کرد؛ getattr با عقب‌افتادگی به RuntimeError
# (در نسخه‌های جدید، پدرِ همین OOM است) این شکاف را می‌بندد
OOM_ERROR_TYPE = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)

# --- فیکس FB-03: وابستگی سخت به google.colab حذف شد؛ تشخیص محیط اجرا ---
# در Colab رفتار دقیقاً مثل قبل است؛ خارج از Colab (Jupyter محلی و…) برنامه بدون درایو و
# Secrets کولب هم راه می‌افتد (مسیرها محلی و Secret از متغیرهای محیطی خوانده می‌شود).
try:
    from google.colab import drive
    IN_COLAB = True
except ImportError:
    drive = None
    IN_COLAB = False
print(f"[Info] محیط اجرا: {'Google Colab' if IN_COLAB else 'Jupyter محلی (بدون قابلیت‌های کولب؛ مسیرها را محلی تنظیم کنید)'}")

# --- اتصال به Google Drive (فقط در Colab؛ فیکس FB-03) ---
if IN_COLAB:
    try:
        drive.mount('/content/drive')
    except Exception as e:
        raise RuntimeError(f"اتصال به Google Drive ناموفق بود: {e}\nلطفاً دوباره تلاش کنید و دسترسی لازم را تأیید نمایید.") from e
else:
    print("[Info] محیط Colab نیست؛ مرحلهٔ اتصال به Google Drive رد شد. مسیرهای ورودی/خروجی را محلی تنظیم کنید.")

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
except OOM_ERROR_TYPE as e:  # فیکس FB-24
    raise RuntimeError(
        f"بارگذاری مدل «{WHISPER_MODEL_NAME}» به دلیل کمبود حافظه GPU ناموفق بود.\n"
        f"پیشنهاد: مدل را به 'turbo' یا 'medium' تغییر دهید یا Runtime را ریست کرده و دوباره امتحان کنید.\n"
        f"جزئیات خطا: {e}"
    ) from e
except Exception as e:
    raise RuntimeError(f"بارگذاری مدل Whisper «{WHISPER_MODEL_NAME}» ناموفق بود: {e}") from e

# فیکس FB-22: capitalize() نام مدل را به «Large-v3» تغییر می‌داد در حالی که نام
# واقعی مدل در پروژهٔ Whisper همان «large-v3» است؛ رشتهٔ اصلی حفظ می‌شود
MODEL_LABEL = WHISPER_MODEL_NAME
print(f"[Info] مدل Whisper «{MODEL_LABEL}» با موفقیت بارگذاری شد.")

# --- مسیر محلی سشن برای ذخیره فایل‌های WAV استخراج‌شده ---
# این مسیر فقط داخل سشن است و هرگز به گوگل درایو منتقل نمی‌شود.
# فیکس FB-03: خارج از Colab به‌جای /content از پوشهٔ جاری استفاده می‌شود.
SESSION_AUDIO_DIR = "/content/session_audio_cache" if IN_COLAB else os.path.join(os.getcwd(), "session_audio_cache")
os.makedirs(SESSION_AUDIO_DIR, exist_ok=True)
print(f"[Info] فایل‌های WAV هر پروژه در همین مسیر سشن ذخیره می‌شوند (نه گوگل درایو): {SESSION_AUDIO_DIR}")

# --- فیکس P3-4: WAV سشنِ فایلِ موفق پس از پایان همان فایل حذف می‌شود تا دیسک سشن پر نشود ---
# برای نگه‌داشتن WAVها در طول سشن این سوییچ را True کنید.
KEEP_SESSION_WAVS = False

# --- فیکس FB-04: WAV سشنِ فایل‌های ناموفق (OOM و…) نیز حذف می‌شود تا با جمع‌شدنشان دیسک سشن پر نشود ---
# فقط برای دیباگ (بررسی صوت فایل خطاخورده) این سوییچ را True کنید.
KEEP_FAILED_WAVS = False

# --- فیکس FB-10: ایمیل به‌ازای هر فایل ریسک rate-limit جیمیل و در بچ‌های بزرگ ---
# تأخیر ناشی از retry دارد؛ پیش‌فرض فقط فایل‌های ناموفق ایمیل می‌شوند و گزارش نهایی
# همیشه ارسال می‌شود. برای ایمیل موفق‌ها هم (رفتار قبلی) این سوییچ را True کنید.
EMAIL_PER_FILE = False

# --- فیکس FB-05: پوشهٔ بازیابی سشن — نسخهٔ امن خروجی‌ها قبل از انتقال به مقصد ---
# اگر انتقال وسط کار شکست بخورد، خروجی‌های آماده از این پوشه قابل دسترسی‌اند و با
# پاک‌شدن پوشهٔ کاری موقت در finally از بین نمی‌روند (خودش با پایان سشن پاک می‌شود).
SESSION_RECOVERY_DIR = "/content/session_output_recovery" if IN_COLAB else os.path.join(os.getcwd(), "session_output_recovery")
os.makedirs(SESSION_RECOVERY_DIR, exist_ok=True)
print(f"[Info] نسخهٔ بازیابی خروجی‌ها هنگام انتقال، در این مسیر سشن نگه داشته می‌شود: {SESSION_RECOVERY_DIR}")

# --- مسیر فایل لاگ کامل اجرای برنامه (در سشن؛ فیکس FB-03: خارج از Colab پوشهٔ جاری) ---
LOG_FILE_PATH = "/content/whisper_processing_log.txt" if IN_COLAB else os.path.join(os.getcwd(), "whisper_processing_log.txt")
print(f"[Info] گزارش کامل اجرای برنامه در این فایل ذخیره می‌شود: {LOG_FILE_PATH}")

# --- فیکس P3-9: نسخهٔ پشتیبان لاگ در گوگل درایو (فایل لاگ سشن با پایان سشن برای همیشه پاک می‌شود) ---
DRIVE_LOG_PATH = "/content/drive/MyDrive/WhisperSub_Logs/whisper_processing_log.txt" if IN_COLAB else os.path.join(os.getcwd(), "WhisperSub_Logs", "whisper_processing_log.txt")
print(f"[Info] در پایان هر پردازش دسته‌ای، نسخهٔ پشتیبان لاگ با نام زمان‌دار در این پوشهٔ گوگل درایو ذخیره می‌شود: {os.path.dirname(DRIVE_LOG_PATH)}")

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

import smtplib

# --- فیکس FB-03: در Colab Secret از Colab Secrets خوانده می‌شود؛ خارج از Colab از متغیرهای محیطی ---
try:
    from google.colab import userdata
except ImportError:
    userdata = None

def get_secret(name):
    """خواندن امن Secret؛ در Colab از Secrets و در محیط محلی از متغیرهای محیطی (None اگر موجود نبود) — فیکس P1-1 + FB-03."""
    if userdata is None:
        return os.environ.get(name)
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
    seen_outputs = set()  # فیکس FB-06: نام‌های پوشهٔ خروجی یکتا ساخته می‌شوند
    # فیکس P4-7: لیست خالی دیگر بی‌صدا برنمی‌گردد؛ علت (پوشهٔ ناموجود / نبود فایل رسانه‌ای) هشدار داده می‌شود
    if not os.path.isdir(source_folder):
        print(f"[Warning] پوشهٔ منبع یافت نشد یا پوشه نیست: {source_folder} — لیست فایل‌ها خالی خواهد بود.")
        return pairs
    for f in sorted(os.listdir(source_folder)):
        full_path = os.path.join(source_folder, f)
        if os.path.isfile(full_path) and f.lower().endswith(VIDEO_EXT + AUDIO_EXT):
            stem = os.path.splitext(f)[0]
            out_folder = os.path.join(destination_root, stem)
            # فیکس FB-06: فایل‌های هم‌نام با پسوند متفاوت (مثل a.mp4 و a.mkv) قبلاً به یک
            # پوشهٔ خروجی می‌رفتند و خروجی یکی بی‌صدا بازنویسی می‌شد؛ حالا نام یکتا می‌سازیم (_2، _3، …)
            unique_n = 2
            while out_folder in seen_outputs:
                out_folder = os.path.join(destination_root, f"{stem}_{unique_n}")
                unique_n += 1
            seen_outputs.add(out_folder)
            pairs.append((full_path, out_folder))
    if not pairs:
        print(f"[Warning] هیچ فایل رسانه‌ای با پسوندهای پشتیبانی‌شده در «{source_folder}» پیدا نشد (فقط خود پوشه بررسی می‌شود، نه زیرپوشه‌ها).")
    return pairs

def validate_pairs(pairs):
    """فیکس P4-7: اعتبارسنجی ساختار pairs؛ لیست جفت‌های معتبر را برمی‌گرداند و موارد نامعتبر را هشدار می‌دهد."""
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(f"input_output_pairs باید لیستی از جفت‌های (مسیر ورودی، پوشهٔ خروجی) باشد؛ نوع دریافتی: {type(pairs).__name__}")
    valid = []
    seen_outputs = {}  # فیکس FB-06: پوشهٔ خروجی -> (شمارهٔ آیتم، نام فایل ورودی)
    for i, item in enumerate(pairs, start=1):
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, str) and x.strip() for x in item)):
            print(f"[Warning] آیتم شمارهٔ {i} در input_output_pairs معتبر نیست (باید جفت دو رشتهٔ غیرخالی باشد) و رد می‌شود: {item!r}")
            continue
        in_path, out_folder = item
        if not in_path.lower().endswith(VIDEO_EXT + AUDIO_EXT):
            print(f"[Warning] پسوند فایل «{os.path.basename(in_path)}» جزو پسوندهای پشتیبانی‌شده نیست؛ در صورت ناموفق بودن ffmpeg، فایل رد می‌شود.")
        if out_folder in seen_outputs:
            # فیکس FB-06: قبلاً فقط هشدار داده می‌شد و خروجی فایل قبلی بی‌صدا بازنویسی می‌شد؛ حالا آیتم تکراری رد می‌شود
            prev_i, prev_name = seen_outputs[out_folder]
            print(f"[Warning] پوشهٔ خروجی «{out_folder}» تکراری است (آیتم‌های {prev_i} و {i}؛ فایل‌های «{prev_name}» و «{os.path.basename(in_path)}»)؛ این آیتم رد می‌شود تا خروجی فایل قبلی بازنویسی نشود.")
            continue
        seen_outputs[out_folder] = (i, os.path.basename(in_path))
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
    print("[Warning] تنظیمات ایمیل یافت نشد؛ برنامه بدون اعلان ایمیلی ادامه می‌یابد. برای فعال‌سازی، SENDER_EMAIL و SENDER_APP_PASSWORD و RECEIVER_EMAIL را در Secrets کولب (یا در محیط محلی به‌صورت متغیرهای محیطی هم‌نام) اضافه کنید.")

# --- فیکس FB-07: precheck «یافت‌نشد»ها قبلاً روی لیست خام اجرا می‌شد؛ آیتم بدشکل ---
# (رشته یا جفت نامعتبر) در unpackingِ «p, _» کرش می‌کرد و آمار شامل مواردی بود که
# validate_pairs رد می‌کند. حالا اول validate_pairs اجرا می‌شود و precheck فقط روی
# لیست معتبر انجام می‌شود؛ نتیجهٔ معتبر جای لیست خام را می‌گیرد.
input_output_pairs = validate_pairs(input_output_pairs)
if not input_output_pairs:
    print("[Warning] هیچ جفت معتبری برای پردازش باقی نمانده است؛ در صورت نیاز input_output_pairs را تنظیم کنید (یا از build_pairs_from_folder استفاده کنید).")
else:
    print(f"\n[Info] تعداد جفت‌های معتبر برای پردازش: {len(input_output_pairs)}")
    missing = [p for p, _ in input_output_pairs if not os.path.exists(p)]
    if missing:
        print(f"[Warning] {len(missing)} فایل از لیست یافت نشدند (در زمان پردازش رد خواهند شد):")
        for m in missing:
            print(f"   - {m}")
    else:
        print("[Info] همه فایل‌های ورودی موجود هستند. ✅")


# @title 🚀 ۴. اجرای موتور پردازشگر Whisper (تبدیل گفتار به متن و تولید خروجی‌های چندگانه)

import collections  # فیکس FB-19: deque برای read_log_tail بدون بارگذاری کل لاگ
import shutil
import tempfile
import time
import zipfile
import traceback
import contextlib
import types
import tqdm
import ffmpeg
from email.header import Header
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

    LOG_ROTATE_MAX_BYTES = 10 * 1024 * 1024  # فیکس FB-13: سقف رشد لاگ سشن (۱۰ مگابایت)

    def _rotate_if_needed(self):
        """فیکس FB-13: اگر فایل لاگ سشن از ۱۰ مگابایت بزرگ‌تر شد، به «.1» چرخش می‌کند
        تا در بچ‌های طولانی/اجراهای مکررِ یک سشن، فایل بی‌نهایت رشد نکند
        (نسخهٔ قبلیِ .1 جایگزین می‌شود؛ سقف مصرف دیسک سشن ~۲۰ مگابایت می‌ماند)."""
        try:
            if os.path.exists(self.log_file_path) and os.path.getsize(self.log_file_path) > self.LOG_ROTATE_MAX_BYTES:
                rotated = self.log_file_path + ".1"
                if os.path.exists(rotated):
                    os.remove(rotated)
                os.rename(self.log_file_path, rotated)
        except OSError:
            pass

    def log(self, msg, level="INFO"):
        icon = self.ICONS.get(level, "")
        line = f"[{time.strftime('%H:%M:%S')}] [+{self._elapsed()}] {icon}{msg}"
        print(line)
        try:
            self._rotate_if_needed()  # فیکس FB-13: چرخش ساده پیش از هر نوشتن
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

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # فیکس FB-12: _last_log قبلاً attribute سطح کلاس بود → state بین همهٔ instanceها
        # (فایل‌های مختلف بچ و نوارهای داخلی) مشترک می‌شد و پیشرفت فایل بعدی بلافاصله
        # log می‌خورد؛ حالا هر instance شمارندهٔ زمانی خودش را دارد
        self._last_log = 0.0

    def update(self, n=1):
        result = super().update(n)
        now = time.time()
        if self.total and now - self._last_log >= PROGRESS_LOG_INTERVAL_SEC:
            self._last_log = now
            logger.info(f"پیشرفت رونویسی: {min(100.0, self.n / self.total * 100):.0f}٪")
        return result

@contextlib.contextmanager
def transcribe_progress_logging():
    """جایگزینی موقت tqdm داخل whisper.transcribe با نسخهٔ لاگ‌کننده؛ پس از اتمام همیشه به حالت اولیه برمی‌گردد.

    فیکس FB-09: پچ فقط با سبک module-style اعمال می‌شود (نسخهٔ پین‌شده در
    whisper/transcribe.py ماژولِ tqdm را import می‌کند و «tqdm.tqdm(...)» صدا می‌زند)؛
    اگر نسخه‌ای از whisper سبک import را عوض کند، به‌جای کرشِ رونویسی فقط هشدار داده
    می‌شود و رونویسی بدون لاگ پیشرفت ادامه می‌یابد. سنتل هم تضمین می‌کند اگر attribute
    از ابتدا وجود نداشته باشد، در finally چیزی مثل «tqdm=None» در ماژول باقی نماند.
    """
    import whisper.transcribe as _wt
    _MISSING = object()  # فیکس FB-09: سنتل به‌جای None — تمایز «موجود ولی None» از «غیرموجود»
    original = getattr(_wt, "tqdm", _MISSING)
    patched = False
    # فقط وقتی پچ کن که ساختار فعلی همان سبک موردانتظار است؛ در غیر این صورت رونویسی
    # دست‌نخورده و سالم ادامه می‌یابد (رفتار پیشین: پچ کورکورانه = کرش در سبک دیگر)
    if original is _MISSING or not callable(getattr(original, "tqdm", None)):
        logger.warning("ساختار tqdm در این نسخهٔ whisper موردانتظار (module-style) نیست؛ لاگ پیشرفت رونویسی غیرفعال شد اما خود رونویسی بدون مشکل ادامه می‌یابد.")
    else:
        try:
            _wt.tqdm = types.SimpleNamespace(tqdm=_LoggingTqdm)
            patched = True
        except Exception as e:
            logger.warning(f"فعال‌سازی لاگ پیشرفت رونویسی ممکن نشد (رونویسی بدون آن ادامه می‌یابد): {e}")
    try:
        yield
    finally:
        if patched:
            if original is _MISSING:
                try:
                    delattr(_wt, "tqdm")  # فیکس FB-09: قبلاً «tqdm=None» در ماژول باقی می‌ماند
                except (AttributeError, TypeError):
                    pass
            else:
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

def save_all_formats(segments, base_filename, work_dir, media_path, model_label, media_duration, audio_path, file_start):
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
    # فیکس FB-15: کلمات هم مانند line_count و total_chars فقط برای سگمنت‌های
    # نوشته‌شده شمرده می‌شوند تا آمار متادیتا با فیلتر متن‌خالی سازگار بماند
    total_words = 0
    line_count = 0  # فیکس P4-5: فقط سگمنت‌های دارای متن شمرده می‌شوند
    with open(paths['srt'], 'w', encoding='utf-8') as srt_f, \
          open(paths['txt'], 'w', encoding='utf-8') as txt_f, \
          open(paths['txt_tl'], 'w', encoding='utf-8') as txt_tl_f:
        for seg in segments:
            text = seg['text'].strip()
            # فیکس P4-5: سگمنت با متن خالی کیوی خالی در SRT تولید می‌کرد؛ رد می‌شود
            if not text:
                continue
            start_s, end_s = seg['start'], seg['end']
            # فیکس FB-02: شماره‌گذاری از enumerate بود و با ردشدن سگمنت خالی، شماره‌ها گسسته
            # و خارج از استاندارد SRT و ناسازگار با line_count متادیتا می‌شد؛ شمارندهٔ مستقل
            # فقط موارد نوشته‌شده را پیوسته شماره‌گذاری می‌کند
            line_count += 1
            line_no = line_count
            start_srt, end_srt = format_srt_time(start_s), format_srt_time(end_s)
            srt_f.write(f"{line_no}\n{start_srt} --> {end_srt}\n{text}\n\n")
            txt_f.write(f"{text}\n")
            txt_tl_f.write(f"{line_no}\n{start_srt} --> {end_srt}\n{text}\n\n")
            total_chars += len(text)
            total_words += len(text.split())  # فیکس FB-15: شمارش داخل حلقهٔ فیلترشده

    timeline_start = segments[0]['start'] if segments else 0.0
    timeline_end = segments[-1]['end'] if segments else 0.0
    avg_chars_per_line = (total_chars / line_count) if line_count else 0
    duration_for_report = media_duration if media_duration is not None else timeline_end

    source_size_mb = os.path.getsize(media_path) / (1024 * 1024) if os.path.exists(media_path) else None
    audio_size_mb = os.path.getsize(audio_path) / (1024 * 1024) if audio_path and os.path.exists(audio_path) else None

    # فیکس FB-17: زمان صرف‌شده در لحظۀ نوشتن متادیتا نهایی می‌شود و شامل رونویسی و
    # ساخت خروجی‌هاست. انتقال به درایو عمداً بیرون از این عدد است چون متادیتا باید
    # پیش از انتقال نهایی (و داخل ZIP) آماده باشد.
    processing_seconds = time.time() - file_start
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

_email_config_warned = False  # فیکس FB-18: هشدار «اطلاعات ایمیل کامل نیست» فقط یک‌بار برای کل سشن چاپ می‌شود

def send_email(subject, body, max_retries=3):
    if not (SENDER_EMAIL and SENDER_APP_PASSWORD and RECEIVER_EMAIL):
        # فیکس FB-18: قبلاً این هشدار به‌ازای هر ایمیل (هر فایل + گزارش‌ها) تکرار می‌شد؛
        # فقط بار اول چاپ می‌شود و بقیهٔ ارسال‌ها بی‌هشدار رد می‌شوند
        global _email_config_warned
        if not _email_config_warned:
            logger.warning("اطلاعات ایمیل کامل نیست؛ تا تنظیم‌شدن ایمیل هیچ ایمیلی ارسال نمی‌شود. (این هشدار یک‌بار چاپ می‌شود)")
            _email_config_warned = True
        return False

    msg = MIMEText(body, 'plain', 'utf-8')
    # فیکس FB-01: Subjectهای فارسی/ایموجی با policy پیش‌فرض compat32 به UnicodeEncodeError
    # در sendmail می‌انجامید و هیچ ایمیلی ارسال نمی‌شد؛ Header با انکود utf-8 موضوع را
    # مطابق RFC 2047 امن انکود می‌کند
    msg['Subject'] = Header(subject, 'utf-8')
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
        except UnicodeEncodeError as e:
            # فیکس FB-01: خطای انکود یونیکد با تلاش مجدد اصلاح نمی‌شود؛ فوراً رد می‌شود (الگوی P3-7)
            logger.error(f"خطای انکود یونیکد هنگام ارسال ایمیل؛ تلاش مجدد بی‌فایده است و ارسال رد شد: {e}")
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
    """کپی فایل لاگ سشن به گوگل درایو تا با پایان سشن از بین نرود (فیکس P3-9).

    فیکس FB-13: قبلاً همیشه روی یک فایل ثابت درایو copy2 می‌شد و لاگ هر اجرا
    نسخهٔ قبلی را بی‌صدا از بین می‌برد؛ حالا نام مقصد با timestamp ساخته می‌شود و
    مسیر فایلِ نوشته‌شده (نه bool) برگردانده می‌شود تا گزارش‌ها مسیر واقعی را بگویند."""
    try:
        if os.path.exists(LOG_FILE_PATH):
            dest_dir = os.path.dirname(DRIVE_LOG_PATH)
            os.makedirs(dest_dir, exist_ok=True)
            dest = os.path.join(dest_dir, f"whisper_processing_log_{time.strftime('%Y%m%d_%H%M%S')}.txt")
            shutil.copy2(LOG_FILE_PATH, dest)
            return dest
    except Exception as e:
        logger.warning(f"کپی لاگ به گوگل درایو ناموفق بود: {e}")
    return None

def read_log_tail(max_lines=40):
    """آخرین خط‌های فایل لاگ را برای الصاق به ایمیل برمی‌گرداند (فیکس P3-9).
    فیکس FB-19: فقط آخرین max_lines خط در حافظه نگه داشته می‌شود (collections.deque)، نه کل فایل."""
    try:
        with open(LOG_FILE_PATH, "r", encoding="utf-8") as f:
            tail = collections.deque(f, maxlen=max_lines)
        return "".join(tail).strip()
    except Exception:
        return ""

# ============================================================
# پردازش دسته‌ای اصلی
# ============================================================
def estimate_eta_seconds(remaining_media_durations, done_records, done_proc_times):
    """فیکس P4-10 + FB-11: تخمین زمان باقی‌مانده.
    پایه: median زمان پردازش فایل‌های انجام‌شده × تعداد باقی‌مانده (median در برابر یک فایل طولانی مقاوم است).
    اگر مدت همهٔ فایل‌های باقی‌مانده معلوم باشد: median نرخِ هر فایل (ثانیهٔ پردازش به‌ازای ثانیهٔ رسانه —
    به‌جای نرخ تجمعی sum/sum که بین فایل‌های نامساوی اریب است) × مجموع مدت باقی‌مانده؛
    فیکس FB-11: نتیجهٔ نرخ‌محور در باند [median÷5 ، median×5] cap می‌شود تا ETAهای وهم‌آلود گزارش نشوند.
    اگر هنوز هیچ فایل موفقی نیست None برمی‌گرداند."""
    if not done_proc_times:
        return None
    ordered = sorted(done_proc_times)
    mid = len(ordered) // 2
    median_time = ordered[mid] if len(ordered) % 2 else (ordered[mid - 1] + ordered[mid]) / 2
    median_based = median_time * len(remaining_media_durations)
    # فیکس FB-11: نرخ per-file (نه تجمعی) → median نرخ‌ها؛ بین فایل‌های نامساوی کمتر اریب است
    per_file_rates = sorted(p / m for p, m in done_records if m and m > 0)
    if per_file_rates and all(d is not None and d > 0 for d in remaining_media_durations):
        mid_r = len(per_file_rates) // 2
        median_rate = per_file_rates[mid_r] if len(per_file_rates) % 2 else (per_file_rates[mid_r - 1] + per_file_rates[mid_r]) / 2
        rate_based = median_rate * sum(remaining_media_durations)
        return min(max(rate_based, median_based / 5.0), median_based * 5.0)  # فیکس FB-11: cap در باند ۵ برابر
    return median_based

def process_all(pairs):
    # فیکس P4-7: اعتبارسنجی ساختار pairs و هشدار لیست خالی پیش از شروع
    pairs = validate_pairs(pairs)
    total = len(pairs)
    results = []
    file_durations = []  # فقط موفق‌ها (مبنای median برای ETA)
    all_file_durations = []  # فیکس FB-11: همهٔ فایل‌ها (موفق + ناموفق) برای میانگین جدا در گزارش نهایی
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

        # فیکس FB-11: فایل‌های ناموجود (شکست فوری) از «باقی‌مانده» حذف می‌شوند تا ETA
        # واقعی بماند؛ media_durations فقط فایل‌های موجود در شروع بچ را در خود دارد
        eta_seconds = estimate_eta_seconds(
            [media_durations[p] for p, _ in pairs[idx - 1:] if p in media_durations], done_records, file_durations)
        if eta_seconds:
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
            free_gb = shutil.disk_usage(SESSION_AUDIO_DIR).free / (1024 ** 3)  # فیکس FB-03: دیسکِ محل ذخیرهٔ WAV (خارج از Colab هم کار می‌کند)
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
            # فیکس FB-08: فایل سکوت/نویز گاهی سگمنت‌هایی با متن خالی تولید می‌کند؛ خروجی
            # صفرلاینی نباید «موفق» ثبت و ایمیل شود. اگر متن همهٔ سگمنت‌ها خالی باشد،
            # همانند نبودِ سگمنت خطا صادر می‌شود تا فایل ناموفق شمرده شود و پردازش
            # با فایل بعدی ادامه یابد (ValueError در except مربوطه مدیریت می‌شود).
            if not any(seg.get('text', '').strip() for seg in segments):
                raise ValueError("هیچ گفتاری شناسایی نشد — متن همهٔ سگمنت‌ها خالی است (احتمالاً فایل سکوت/نویز است).")
            logger.success(f"رونویسی کامل شد: {len(segments)} لاین شناسایی شد.")

            logger.step("در حال تولید فرمت‌های خروجی (SRT/TXT) و متادیتا...")
            stats = save_all_formats(segments, base_filename, work_dir, input_path,
                                      MODEL_LABEL, media_duration, audio_path,
                                      file_start)  # فیکس FB-17: زمان داخل تابع و در لحظۀ نوشتن متادیتا نهایی می‌شود

            logger.step(f"در حال انتقال فایل‌های خروجی به گوگل درایو: {output_folder}")
            os.makedirs(output_folder, exist_ok=True)
            # فیکس FB-05: انتقال قبلاً ناامن بود — (۱) با شکست انتقال، خروجی‌های آماده در
            # rmtreeِ finally نابود می‌شدند؛ (۲) حذف مقصد قبل از move با خطای move هر دو نسخه را
            # از بین می‌برد؛ (۳) os.remove روی مقصد پوشه‌ای کرش می‌کرد. الگوی امن: اول نسخهٔ
            # بازیابی در پوشهٔ سشن، سپس بازنویسی مقصد با rename موقت .bak.
            transfer_recovery_dir = os.path.join(SESSION_RECOVERY_DIR, f"{base_filename}_{int(time.time())}")
            os.makedirs(transfer_recovery_dir, exist_ok=True)
            for fname in os.listdir(work_dir):
                src_path = os.path.join(work_dir, fname)
                dest_path = os.path.join(output_folder, fname)
                shutil.copy2(src_path, os.path.join(transfer_recovery_dir, fname))  # فیکس FB-05: نسخهٔ امن در سشن
                if os.path.isdir(dest_path):  # فیکس FB-05: گارد isdir — os.remove روی پوشه کرش می‌کرد
                    logger.warning(f"مقصد «{fname}» پوشه است نه فایل؛ این مورد رد می‌شود و نسخهٔ امن آن در پوشهٔ بازیابی باقی می‌ماند.")
                    continue
                if os.path.exists(dest_path):  # فیکس P2-1: اجرای مجدد روی خروجی موجود خطا نمی‌دهد
                    bak_path = dest_path + ".bak"
                    if os.path.exists(bak_path):
                        try:
                            os.remove(bak_path)
                        except OSError:
                            pass
                    os.rename(dest_path, bak_path)  # فیکس FB-05: حذف مستقیم مقصد در صورت خطای move هر دو نسخه را از بین می‌برد
                    try:
                        shutil.move(src_path, dest_path)
                        os.remove(bak_path)  # انتقال موفق؛ نسخهٔ قدیمی حذف می‌شود
                        logger.info(f"خروجی قبلی بازنویسی شد: {fname}")
                    except Exception:
                        os.rename(bak_path, dest_path)  # فیکس FB-05: بازگرداندن نسخهٔ قبلی هنگام خطای انتقال
                        raise
                else:
                    shutil.move(src_path, dest_path)
            shutil.rmtree(transfer_recovery_dir, ignore_errors=True)  # فیکس FB-05: انتقال کامل شد؛ نسخهٔ بازیابی این فایل لازم نیست

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
        except OOM_ERROR_TYPE as e:  # فیکس FB-24
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
            # فیکس P3-4 + FB-04: WAV سشن پس از پایان هر فایل — چه موفق چه ناموفق — حذف می‌شود تا
            # در بچ‌های بزرگ دیسک سشن پر نشود؛ قبلاً فقط فایل‌های موفق پاک می‌شدند و WAV خطاخورده‌ها
            # (مثل OOM) جمع می‌شدند و دیسک را پر می‌کردند (آبشار شکست). KEEP_FAILED_WAVS فقط برای دیباگ است.
            keep_wav = KEEP_SESSION_WAVS or (status != "موفق" and KEEP_FAILED_WAVS)
            if audio_path and not keep_wav:
                try:
                    os.remove(audio_path)
                except OSError:
                    pass

        all_file_durations.append(duration)  # فیکس FB-11: زمان همهٔ فایل‌ها (موفق + ناموفق) ثبت می‌شود
        logger.info(f"خلاصه تا این لحظه: {success_count} موفق، {fail_count} ناموفق از {idx} فایل بررسی‌شده")

        # فیکس FB-10: به‌جای ایمیل به‌ازای هر فایل (ریسک rate-limit جیمیل + تأخیر retry
        # در بچ‌های بزرگ)، پیش‌فرض فقط ناموفق‌ها ایمیل می‌شوند؛ گزارش نهایی همیشه ارسال
        # می‌شود. برای ایمیل موفق‌ها هم EMAIL_PER_FILE را در بخش ۲ True کنید.
        if EMAIL_PER_FILE or status != "موفق":
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
                )
                # فیکس FB-14: WAV پس از فایل موفق حذف می‌شود؛ مسیر حذف‌شده «موجود» جلوه نمی‌کند
                per_file_body += (f"مسیر فایل WAV (فقط سشن کولب): {audio_path}\n" if KEEP_SESSION_WAVS else "فایل WAV سشن پس از پردازش موفق حذف شد (برای نگه‌داشتن، KEEP_SESSION_WAVS را در بخش ۲ True کنید)\n")
            else:
                per_file_body += f"جزئیات خطا: {error_detail}\n"
            per_file_body += f"\nپیشرفت کلی پروژه: {idx}/{total} فایل ({idx/total*100:.0f}٪)\n"

            send_email(f"{'✅' if status == 'موفق' else '❌'} گزارش فایل {idx}/{total}: {base_filename}", per_file_body)

    # فیکس FB-23: empty_cache از پایان هر فایل به پایان بچ منتقل شد — پاک‌سازی
    # به‌ازای هر فایل سربار sync و allocator دارد؛ پس از OOM همچنان بلافاصله در
    # همان هندلر پاک‌سازی می‌شود
    if device == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

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
    # فیکس FB-11: میانگین‌های جدا — قبلاً میانگین فقط موفق‌ها را نشان می‌داد و شکست‌های
    # سریع/کشده (ناموجود، بدون گفتار، OOM) از آمار نهایی مخفی می‌ماندند
    if file_durations:
        avg_ok = sum(file_durations) / len(file_durations)
        summary_header += f"میانگین زمان پردازش هر فایل موفق: {avg_ok:.1f} ثانیه\n"
    if all_file_durations:
        avg_all = sum(all_file_durations) / len(all_file_durations)
        summary_header += f"میانگین زمان پردازش همهٔ فایل‌ها (موفق + ناموفق): {avg_all:.1f} ثانیه\n"
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

    # فیکس FB-05: اطلاع‌رسانی دربارهٔ نسخه‌های بازیابی مانده در سشن (خروجی فایل‌های با انتقال ناموفق)
    try:
        recovery_files = [os.path.join(root, f) for root, _, files in os.walk(SESSION_RECOVERY_DIR) for f in files]
        if recovery_files:
            logger.info(f"{len(recovery_files)} نسخهٔ بازیابی از خروجی فایل‌های ناموفق در پوشهٔ سشن باقی مانده است: {SESSION_RECOVERY_DIR}")
            logger.warning("این نسخه‌ها با پایان یا ریست‌شدن سشن برای همیشه پاک می‌شوند؛ در صورت نیاز همین حالا به جای امنی منتقلشان کنید.")
    except Exception:
        pass

    # فیکس P3-9: لاگ سشن با پایان سشن نابود می‌شود؛ نسخهٔ درایو + دنبالهٔ لاگ در ایمیل نهایی
    drive_backup_path = backup_log_to_drive()  # فیکس FB-13: مسیر زمان‌دار واقعی برمی‌گردد
    if drive_backup_path:
        logger.info(f"نسخهٔ پشتیبان لاگ در گوگل درایو ذخیره شد: {drive_backup_path}")
        summary += f"\n\nنسخهٔ پشتیبان لاگ کامل در گوگل درایو: {drive_backup_path}"
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
    critical_backup_path = backup_log_to_drive()  # فیکس FB-13: مسیر زمان‌دار واقعی
    critical_tail = read_log_tail(30)
    send_email("🚨 خطای بحرانی در اجرای برنامه", f"برنامه با خطای غیرمنتظره متوقف شد:\n{type(e).__name__}: {e}\n\nجزئیات کامل خطا در فایل لاگ ذخیره شد؛ نسخهٔ پشتیبان: {critical_backup_path if critical_backup_path else 'ساخته نشد (متن لاگ فقط داخل سشن باقی است)'}\n\n{'=' * 40}\n{critical_tail}")
    raise
