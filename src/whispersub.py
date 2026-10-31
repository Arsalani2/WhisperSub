# -*- coding: utf-8 -*-
# ==== WhisperSub Release Info (auto-generated) ====
# WhisperSub — نسخه v1.0.7
# تاریخ ساخت این ریلیز: 2026-10-31 | کامیت: 8804b8b
# ریلیز: https://github.com/Personalmo/WhisperSub/releases/tag/v1.0.7
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
# فیکس FB-27: tqdm صریحاً نصب می‌شود — پچ پیشرفت P4-6 مستقیماً به آن وابسته است و
# بیرون از Colab ممکن است از قبل نصب نباشد (فقط به وابستگی انتقالی whisper اتکا نکنید)
# فیکس NB-07: tqdm هم مثل سایر پکیج‌ها پین شد (آخرین نسخهٔ پایدار PyPI) — مانکی‌پچ
# پیشرفت (P4-6/FB-09) به رفتار tqdm وابسته است و پیشرفتِ بستهٔ پین‌نشده بین اجراها
# ممکن است تغییر کند؛ requirements.txt در ریلیز بعدی خودکار sync می‌شود
# !pip install -q "openai-whisper==20250625" "ffmpeg-python==0.2.0" "tqdm==4.70.1"  # Colab shell magic

# فیکس FB-21: قبلاً شکست !pip بی‌صدا رد می‌شد و اولین نشانه، ModuleNotFoundError
# گمراه‌کننده در سلول بعدی بود؛ حالا نصب بلافاصله پس از pip بررسی می‌شود
import importlib
for _pkg in ("whisper", "ffmpeg", "tqdm", "torch"):  # فیکس NB-13: tqdm (FB-27) و torch هم مستقیم import می‌شوند
    try:
        importlib.import_module(_pkg)
    except ImportError as e:
        raise RuntimeError(
            f"نصب پیش‌نیازها ناموفق بود (بستهٔ {_pkg}): {e}\n"
            "خروجی pip در همین سلول را بررسی کنید (شبکه یا تضاد وابستگی‌ها) و سلول را دوباره اجرا کنید."
        ) from e
print("[Info] نصب پیش‌نیازها با موفقیت بررسی شد (whisper, ffmpeg, tqdm, torch).")


# @title ⚙️ ۲. پیکربندی محیط و فراخوانی مدل Whisper
import os
import shutil  # فیکس FB-26: چک وجود باینری ffmpeg/ffprobe در PATH
import torch
import whisper

# فیکس FB-24: در نسخه‌های قدیمی torch، torch.cuda.OutOfMemoryError وجود ندارد و
# خودِ except با AttributeError کرش می‌کرد؛ getattr با عقب‌افتادگی به RuntimeError
# (در نسخه‌های جدید، پدرِ همین OOM است) این شکاف را می‌بندد
OOM_ERROR_TYPE = getattr(torch.cuda, "OutOfMemoryError", RuntimeError)
OOM_FALLBACK_ACTIVE = OOM_ERROR_TYPE is RuntimeError  # فیکس NB-14: روی torch قدیمی OOM کلاس ندارد

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

# --- فیکس FB-26: پکیج ffmpeg-python فقط wrapper است؛ خودِ باینری جدا نصب می‌شود ---
# (در Colab از قبل موجود است؛ در محیط محلی اگر نباشد، اولین خطا گمراه‌کننده است)
_missing_bins = [b for b in ("ffmpeg", "ffprobe") if shutil.which(b) is None]
if _missing_bins:
    print(f"[Warning] باینری {', '.join(_missing_bins)} در PATH یافت نشد؛ استخراج/بررسی صدا شکست خواهد خورد.")
    print("[Warning] راهنمای نصب: اوبونتو: sudo apt install ffmpeg | مک: brew install ffmpeg | ویندوز: بستهٔ static-build و افزودن به PATH.")
else:
    print("[Info] باینری‌های ffmpeg و ffprobe موجود هستند. ✅")

WHISPER_MODEL_NAME = "large-v3"  # @param ["tiny", "base", "small", "medium", "large", "large-v2", "large-v3", "turbo"]

# --- زبان رونویسی (فیکس P3-2: به‌جای هاردکد 'en'؛ در transcribe و متادیتا از همین ثابت استفاده می‌شود) ---
TRANSCRIBE_LANGUAGE = "en"  # @param {type:"string"}

# فیکس FB-28: اعتبارسنجی زبان در شروع (نه وسط بچ بعد از لود مدل) — کد اشتباه مثل
# «farsi» یا «EN» همین‌جا با پیام روشن و فهرست کدهای معتبر متوقف می‌شود
TRANSCRIBE_LANGUAGE = str(TRANSCRIBE_LANGUAGE).lower()
if TRANSCRIBE_LANGUAGE not in whisper.tokenizer.LANGUAGES:
    raise ValueError(
        f"TRANSCRIBE_LANGUAGE «{TRANSCRIBE_LANGUAGE}» در کدهای زبان Whisper نیست؛ "
        f"کدهای معتبر: {', '.join(sorted(whisper.tokenizer.LANGUAGES))}"
    )

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
    # فیکس FB-38: پارامتر device به خودِ load_model داده می‌شود؛ انتقال دوم (.to)
    # فقط سربار و ابهام بود — whisper.load_model خودش روی همان device لود می‌کند
    model = whisper.load_model(WHISPER_MODEL_NAME, device=device)
except OOM_ERROR_TYPE as e:  # فیکس FB-24
    # فیکس NB-14: روی torch قدیمی (fallback = RuntimeError) هر RuntimeError این شاخه را می‌گیرد؛
    # بدون نشانهٔ OOM در پیام، ادعای «کمبود حافظه GPU» غلط است و پیام generic می‌آید
    if OOM_FALLBACK_ACTIVE and not ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
        raise RuntimeError(f"بارگذاری مدل Whisper «{WHISPER_MODEL_NAME}» ناموفق بود: {e}") from e
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
KEEP_SESSION_WAVS = False  # @param {type:"boolean"}

# --- فیکس FB-04: WAV سشنِ فایل‌های ناموفق (OOM و…) نیز حذف می‌شود تا با جمع‌شدنشان دیسک سشن پر نشود ---
# فقط برای دیباگ (بررسی صوت فایل خطاخورده) این سوییچ را True کنید.
KEEP_FAILED_WAVS = False  # @param {type:"boolean"}

# --- حالت UI/UX موبایلی (طبق الگوی پروژهٔ TeleScrape): خطوط خروجی کنسول به ---
# --- عرض صفحهٔ موبایل شکسته می‌شوند (اسلایدر زیر) و تنظیمات کلیدی به فرم Colab ---
# --- (اسلایدر/دراپ‌داونِ قابل استفاده روی موبایل) تبدیل شده‌اند ---
UI_WIDTH = 34  # @param {type:"slider", min:24, max:60, step:1}

# --- لاگ کم‌جزئیات: پیش‌فرض فقط رویدادهای مهم در کنسول چاپ می‌شود (شروع/پایان هر ---
# --- فایل، ETA، هشدارها و خطاها)؛ پیام‌های گام‌به‌گام (🔹) فقط در فایل لاگ ---
# --- ذخیره می‌شوند و هیچ اطلاعاتی از دست نمی‌رود. True = لاگ کامل مثل قبل ---
VERBOSE_LOGS = False  # @param {type:"boolean"}

# --- فیکس NB-12: توقف بچ بعد از N شکست سیستمی متوالی (قطع درایو / دیسک پر) ---
# خطاهای زیرساختیِ پشت‌سرهم نشانهٔ مشکل محیط است؛ ادامهٔ بچ فقط GPU را هدر می‌دهد
# و به‌ازای هر فایل یک ایمیل خطا می‌فرستد. 0 = خاموش (رفتار قبلی)
ABORT_AFTER_SYSTEMIC_FAILURES = 3  # @param {type:"integer"}

# --- فیکس FB-10: ایمیل به‌ازای هر فایل ریسک rate-limit جیمیل و در بچ‌های بزرگ ---
# تأخیر ناشی از retry دارد؛ پیش‌فرض فقط فایل‌های ناموفق ایمیل می‌شوند و گزارش نهایی
# همیشه ارسال می‌شود. برای ایمیل موفق‌ها هم (رفتار قبلی) این سوییچ را True کنید.
EMAIL_PER_FILE = False  # @param {type:"boolean"}

# --- فیکس FB-29: هاست/پورت SMTP هاردکد بود؛ برای سرویس ایمیل غیر Gmail قابل تنظیم ---
SMTP_HOST = "smtp.gmail.com"  # مثلاً smtp.mail.yahoo.com یا smtp.office365.com
SMTP_PORT = 465  # SSL

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


# --- فیکس NB-24: تابع استاندارد برای حذف کاراکترهای غیرمجاز فایل‌سیستم ---
def sanitize_filename(name, fallback="media"):
    """حذف کاراکترهای ممنوعه در فایل‌سیستم‌های ویندوز و Google Drive FUSE (\/:*?"<>|)
    تا نام فایل‌ها و پوشه‌ها بدون خطا ساخته شوند."""
    cleaned = "".join(c for c in name if c not in r'\/:*?"<>|').strip().rstrip(". ")
    return cleaned or fallback

# --- ساخت خودکار لیست فایل‌ها از روی یک پوشه (فیکس NB-22: تعریف در بخش ۲) ---
def build_pairs_from_folder(source_folder, destination_root):
    """برای هر فایل رسانه‌ای داخل source_folder یک زوج (ورودی، خروجی) در destination_root می‌سازد.
    فیکس NB-22: تعریف در بخش ۲ تا قبل از مقداردهی input_output_pairs در دسترس باشد (رفع NameError).
    فیکس NB-24: پاک‌سازی کاراکترهای غیرمجاز در نام پوشهٔ خروجی.
    فیکس NB-25: نرمال‌سازی مسیرها و حذف فاصله‌ها/کوتیشن‌های احتمالی."""
    pairs = []
    seen_outputs = set()
    source_folder = str(source_folder).strip().strip("'\"")
    destination_root = str(destination_root).strip().strip("'\"")
    if not os.path.isdir(source_folder):
        print(f"[Warning] پوشهٔ منبع یافت نشد یا پوشه نیست: {source_folder} — لیست فایل‌ها خالی خواهد بود.")
        return pairs
    for f in sorted(os.listdir(source_folder)):
        full_path = os.path.join(source_folder, f)
        if os.path.isfile(full_path) and f.lower().endswith(VIDEO_EXT + AUDIO_EXT):
            stem = sanitize_filename(os.path.splitext(f)[0], fallback="media")  # فیکس NB-24
            out_folder = os.path.normpath(os.path.join(destination_root, stem))  # فیکس NB-25
            unique_n = 2
            while out_folder in seen_outputs:
                out_folder = os.path.normpath(os.path.join(destination_root, f"{stem}_{unique_n}"))
                unique_n += 1
            seen_outputs.add(out_folder)
            pairs.append((full_path, out_folder))
    if not pairs:
        print(f"[Warning] هیچ فایل رسانه‌ای با پسوندهای پشتیبانی‌شده در «{source_folder}» پیدا نشد (فقط خود پوشه بررسی می‌شود، نه زیرپوشه‌ها).")
    return pairs

# --- فایل‌های ورودی و پوشهٔ خروجی هرکدام (فیکس P2-2: مسیرهای شخصی از ریپوی عمومی حذف شدند) ---
# هر جفت به شکل (مسیر فایل ورودی، پوشهٔ خروجی در گوگل درایو) است. مثال:
# input_output_pairs = [
#     ('/content/drive/MyDrive/Videos/lesson1.mp4', '/content/drive/MyDrive/Subs/lesson1'),
# ]
# یا ساخت خودکار از روی یک پوشه (با تابع build_pairs_from_folder تعریف‌شده در بالا):
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
    """خواندن امن Secret؛ در Colab از Secrets و در محیط محلی از متغیرهای محیطی (None اگر موجود نبود) — فیکس P1-1 + FB-03.

    فیکس FB-25: «نبودِ Secret» (SecretNotFoundError) حالت عادیِ تنظیم‌نشدن است و مثل
    قبل None برمی‌گردد؛ اما خطاهای واقعی (مثل خطای دسترسی) دیگر بی‌صدا بلعیده نمی‌شوند."""
    if userdata is None:
        return os.environ.get(name)
    try:
        return userdata.get(name)
    except Exception as e:
        # فیکس FB-25: فقط «Secret یافت نشد» حالت عادی است؛ سایر خطاها هشدار می‌گیرند.
        # (SecretNotFoundError در همهٔ نسخه‌های colab به‌صورت عمومی export نشده و
        # چک با نامِ رده انجام می‌شود)
        if type(e).__name__ == "SecretNotFoundError":
            return None
        print(f"[Warning] خطا در خواندن Secret «{name}»: {type(e).__name__}: {e}")
        return None

SENDER_EMAIL = get_secret('SENDER_EMAIL')
SENDER_APP_PASSWORD = get_secret('SENDER_APP_PASSWORD')
RECEIVER_EMAIL = get_secret('RECEIVER_EMAIL')

# فیکس NB-22: تابع build_pairs_from_folder جهت دسترسی آسان هنگام مقداردهی ورودی‌ها به بخش ۲ منتقل شد.
# برای حفظ سازگاری معکوس، در صورتی که نیاز باشد تابع از بخش ۲ در دسترس است.

def validate_pairs(pairs, quiet=False):
    """فیکس P4-7: اعتبارسنجی ساختار pairs؛ جفت‌های معتبر + علت‌های ردشدگی را برمی‌گرداند.

    فیکس NB-06: پارامتر quiet — فراخوان دومِ اعتبارسنجی داخل process_all (تلاش
    مجدد سلول ۵) دیگر هشدارهای تکراری چاپ نمی‌کند؛ پیام‌ها یک‌بار در precheck سلول ۳ کافی است.
    فیکس FB-32: خروجی (valid, rejects) — علت‌های کوتاه ردشدگی برای گزارش نهایی.
    فیکس NB-08: پسوند ناشناخته دیگر با هشدارِ گمراه‌کننده در بچ نمی‌ماند؛ رد قطعی می‌شود.
    """
    if not isinstance(pairs, (list, tuple)):
        raise TypeError(f"input_output_pairs باید لیستی از جفت‌های (مسیر ورودی، پوشهٔ خروجی) باشد؛ نوع دریافتی: {type(pairs).__name__}")
    valid = []
    rejects = []  # فیکس FB-32: علت‌های کوتاه ردشدگی برای گزارش نهایی
    seen_outputs = {}  # فیکس FB-06: پوشهٔ خروجی -> (شمارهٔ آیتم، نام فایل ورودی)
    for i, item in enumerate(pairs, start=1):
        if not (isinstance(item, (list, tuple)) and len(item) == 2 and all(isinstance(x, str) and x.strip() for x in item)):
            if not quiet:  # فیکس NB-06: پیام فقط در precheck سلول ۳ چاپ می‌شود
                print(f"[Warning] آیتم شمارهٔ {i} در input_output_pairs معتبر نیست (باید جفت دو رشتهٔ غیرخالی باشد) و رد می‌شود: {item!r}")
            rejects.append(f"آیتم {i}: ساختار نامعتبر")  # فیکس FB-32
            continue
        in_path, out_folder = item
        # فیکس NB-25: پاک‌سازی فاصله‌های اضافی و کوتیشن‌های احتمالی اطراف مسیرها و نرمال‌سازی مسیر خروجی
        in_path = in_path.strip().strip("'\"")
        out_folder = os.path.normpath(out_folder.strip().strip("'\""))
        if not in_path.lower().endswith(VIDEO_EXT + AUDIO_EXT):
            # فیکس NB-08: قبلاً فقط هشدار داده می‌شد و جفت در بچ می‌ماند؛ پیام هم به ffmpeg
            # نسبت داده می‌شد درحالی‌که extract_audio خودش در چک پسوند ValueError می‌دهد و
            # تنها نتیجه‌اش slot بچ + ایمیل خطا برای چیزی بود که از قبل معلوم بود. رد قطعی:
            if not quiet:  # فیکس NB-06
                print(f"[Warning] پسوند فایل «{os.path.basename(in_path)}» پشتیبانی نمی‌شود؛ این جفت رد شد.")
            rejects.append(f"آیتم {i}: پسوند پشتیبانی نمی‌شود ({os.path.basename(in_path)})")  # فیکس FB-32 + NB-08
            continue
        if out_folder in seen_outputs:
            # فیکس FB-06: قبلاً فقط هشدار داده می‌شد و خروجی فایل قبلی بی‌صدا بازنویسی می‌شد؛ حالا آیتم تکراری رد می‌شود
            prev_i, prev_name = seen_outputs[out_folder]
            if not quiet:  # فیکس NB-06
                print(f"[Warning] پوشهٔ خروجی «{out_folder}» تکراری است (آیتم‌های {prev_i} و {i}؛ فایل‌های «{prev_name}» و «{os.path.basename(in_path)}»)؛ این آیتم رد می‌شود تا خروجی فایل قبلی بازنویسی نشود.")
            rejects.append(f"آیتم {i}: پوشهٔ خروجی تکراری «{out_folder}»")  # فیکس FB-32
            continue
        seen_outputs[out_folder] = (i, os.path.basename(in_path))
        valid.append((in_path, out_folder))
    return valid, rejects  # فیکس FB-32: علت‌های ردشدگی برای گزارش نهایی

def ensure_output_folder(out_folder):
    """فیکس NB-02: پوشهٔ خروجی را می‌سازد و قابل‌نوشتن بودن آن را با write-test واقعی بررسی می‌کند (False = نامعتبر)."""
    test_file = os.path.join(out_folder, ".whispersub_write_test")
    try:
        os.makedirs(out_folder, exist_ok=True)
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        return True
    except OSError:
        return False

if SENDER_EMAIL and SENDER_APP_PASSWORD and RECEIVER_EMAIL:
    print(f"[Info] ایمیل فرستنده تنظیم شد: {SENDER_EMAIL}")
    print(f"[Info] ایمیل گیرنده: {RECEIVER_EMAIL}")
    print(f"[Info] در حال آزمایش اتصال به سرور SMTP ({SMTP_HOST})...")
    try:
        with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=20) as server:  # فیکس FB-29
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
# فیکس FB-32: علت‌های ردشدگی در متغیر ماژول‌سطح نگه داشته می‌شود تا در گزارش نهایی
# ایمیل هم بیاید؛ کاربری که فقط ایمیل نهایی را می‌خواند از کم‌بودنِ تعداد بی‌خبر نماند
input_output_pairs, PAIRS_REJECTS = validate_pairs(input_output_pairs)
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
    # فیکس NB-02: پوشهٔ خروجی نامعتبر/غیرقابل‌نوشتن پیش از هر پردازشی گزارش می‌شود (fail-fast)
    bad_outputs = [out for _, out in input_output_pairs if not ensure_output_folder(out)]
    if bad_outputs:
        print(f"[Warning] {len(bad_outputs)} پوشهٔ خروجی قابل ساخت/نوشتن نیست (جفت‌های مربوطه در شروع پردازش رد می‌شوند):")
        for b in bad_outputs:
            print(f"   - {b}")


# @title 🚀 ۴. اجرای موتور پردازشگر Whisper (تبدیل گفتار به متن و تولید خروجی‌های چندگانه)

import math  # فیکس NB-29: بررسی امن NaN در محاسبات زمانی
import collections  # فیکس FB-19: deque برای read_log_tail بدون بارگذاری کل لاگ
import textwrap  # حالت موبایلی: شکستن خطوط لاگ کنسول به عرض UI_WIDTH (الگوی say در TeleScrape)
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

    def __init__(self, log_file_path, verbose=False, wrap_width=0):
        self.log_file_path = log_file_path
        self.start_time = time.time()
        # حالت موبایلی/کم‌جزئیات: wrap_width>0 خطوط کنسول را به عرض موبایل می‌شکند و
        # verbose=True لاگ کامل قدیم را برمی‌گرداند (فایل لاگ در هر حالتی کامل نوشته می‌شود)
        self.verbose = verbose
        self.wrap_width = int(wrap_width) if wrap_width else 0
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
                # فیکس FB-34: exists/remove/rename سه‌مرحله‌ای race داشت — اگر remove
                # خطا بخورد، except بیرونی کل بلوک را رد می‌کند و rename بی‌صدا انجام
                # نمی‌شود (چرخش از کار می‌افتد بدون هیچ نشانه)؛ os.replace بازنویسی
                # اتمیک است (لینوکس + ویندوز) و جفت exists/remove کلاً حذف می‌شود
                os.replace(self.log_file_path, rotated)
        except OSError:
            pass

    def _console(self, line):
        """چاپ کنسول با شکستن خط به عرض موبایل (الگوی say در TeleScrape):
        توکن‌های بی‌قاعده (مسیرهای طولانی) هارد‌وریز می‌شوند و بعد textwrap.fill —
        در دسکتاپ (wrap_width خاموش) رفتار قبلی یعنی print ساده حفظ می‌شود."""
        if not (self.wrap_width and self.wrap_width >= 20):
            print(line)
            return
        w = self.wrap_width
        for ln in line.split("\n"):
            if not ln:
                print()
                continue
            if len(ln) > w:
                pieces = []
                for tok in ln.split(" "):
                    while len(tok) > w:
                        pieces.append(tok[:w])
                        tok = tok[w:]
                    pieces.append(tok)
                ln = " ".join(pieces)
            print(textwrap.fill(ln, w, break_long_words=False, break_on_hyphens=False))

    def log(self, msg, level="INFO", force=False):
        icon = self.ICONS.get(level, "")
        # فایل لاگ همیشه با فرمت کامل نوشته می‌شود (حتی در حالت کم‌جزئیات) تا
        # بکاپ درایو و دنبالهٔ ایمیل نهایی (فیکس P3-9) کامل بمانند
        full_line = f"[{time.strftime('%H:%M:%S')}] [+{self._elapsed()}] {icon}{msg}"
        try:
            self._rotate_if_needed()  # فیکس FB-13: چرخش ساده پیش از هر نوشتن
            with open(self.log_file_path, "a", encoding="utf-8") as f:
                f.write(full_line + "\n")
        except Exception:
            pass
        # حالت کم‌جزئیات (پیش‌فرض): پیام‌های گام‌به‌گام (🔹) فقط در فایل لاگ
        # ذخیره می‌شوند و در کنسول چاپ نمی‌شوند؛ پیشوند کنسول هم کوتاه می‌شود
        # تا روی موبایل جا برای خود پیام باقی بماند. force=True پیام گامیِ مهم
        # (مثل شروع رونویسی) را در هر حالتی نشان می‌دهد
        if level == "STEP" and not self.verbose and not force:
            return
        if self.verbose:
            self._console(full_line)
        else:
            self._console(f"[{time.strftime('%H:%M')}] {icon}{msg}")

    def info(self, msg): self.log(msg, "INFO")
    def success(self, msg): self.log(msg, "SUCCESS")
    def warning(self, msg): self.log(msg, "WARNING")
    def error(self, msg): self.log(msg, "ERROR")
    def step(self, msg, force=False): self.log(msg, "STEP", force=force)

logger = ColabLogger(LOG_FILE_PATH, verbose=VERBOSE_LOGS, wrap_width=UI_WIDTH)

# ============================================================
# توابع کمکی
# ============================================================
def format_srt_time(seconds):
    # فیکس P3-1 + NB-29: گردکردن صحیح + محافظت در برابر None و NaN
    if seconds is None or (isinstance(seconds, float) and math.isnan(seconds)):
        seconds = 0.0
    total_ms = max(0, int(round(seconds * 1000)))
    h, rem_ms = divmod(total_ms, 3600000)
    m, rem_ms = divmod(rem_ms, 60000)
    s, ms = divmod(rem_ms, 1000)
    # فیکس FB-35: استاندارد SRT ساعتِ دو رقمی است؛ برای مدت‌های ۱۰۰+ ساعته سه‌رقمی
    # شدنِ ساعت خروجی را غیراستاندارد می‌کند — در سقف 99:59:59,999 محدود می‌شود
    if h > 99:
        h, m, s, ms = 99, 59, 59, 999
        logger.warning("مدت رسانه از ۹۹ ساعت بیشتر است؛ تایم‌استمپ SRT در سقف استاندارد cap شد")
    return f"{int(h):02}:{int(m):02}:{int(s):02},{ms:03}"

def probe_media_duration(media_path):
    """مدت زمان دقیق فایل رسانه را با ffprobe برمی‌گرداند (ثانیه) یا None در صورت خطا.

    فیکس FB-36: برخی فرمت‌های صوتی مدت را فقط در streams[0].duration یا
    streams[0].tags['DURATION'] می‌دهند؛ زنجیرهٔ fallback: format.duration →
    streams[0].duration → tags['DURATION'] (فرمت HH:MM:SS.mmm) — اولین مقدار معتبر برنده است."""
    try:
        info = ffmpeg.probe(media_path)
    except ffmpeg.Error as e:
        stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else str(e)
        logger.warning(f"عدم موفقیت ffprobe در تشخیص مدت‌زمان: {stderr.strip().splitlines()[-1] if stderr.strip() else e}")
        return None
    except Exception as e:
        logger.warning(f"خطای غیرمنتظره هنگام بررسی مدت‌زمان فایل: {e}")
        return None
    # فیکس FB-36: زنجیرهٔ fallback با try/exceptهای جدا؛ اولین مقدار معتبر return می‌شود
    try:
        return float(info['format']['duration'])
    except (KeyError, TypeError, ValueError):
        pass
    # فیکس NB-26: پیمایش تمام استریم‌ها به‌جای اتکا به streams[0] — استریم اول ممکن است ویدیو/کانتینر بدون مدت باشد
    for stream in info.get("streams", []):
        try:
            if "duration" in stream and stream["duration"] is not None:
                return float(stream["duration"])
        except (KeyError, TypeError, ValueError):
            pass
        try:
            if "tags" in stream and "DURATION" in stream["tags"]:
                _dur_tag = str(stream["tags"]["DURATION"])  # فیکس FB-36: فرمت HH:MM:SS.mmm
                _hh, _mm, _ss = _dur_tag.split(":")
                return int(_hh) * 3600 + int(_mm) * 60 + float(_ss)
        except (KeyError, TypeError, ValueError, IndexError):
            pass
    logger.warning(f"اطلاعات مدت‌زمان در هیچ‌یک از format.duration / streams.duration / tags.DURATION یافت نشد: {os.path.basename(media_path)}")
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

_tqdm_struct_warned = False  # هشدار ساختار tqdm در کل سشن فقط یک‌بار چاپ می‌شود (الگوی FB-18)

@contextlib.contextmanager
def transcribe_progress_logging():
    """جایگزینی موقت tqdm داخل whisper.transcribe با نسخهٔ لاگ‌کننده؛ پس از اتمام همیشه به حالت اولیه برمی‌گردد.

    فیکس FB-09: پچ فقط با سبک module-style اعمال می‌شود (نسخهٔ پین‌شده در
    whisper/transcribe.py ماژولِ tqdm را import می‌کند و «tqdm.tqdm(...)» صدا می‌زند)؛
    اگر نسخه‌ای از whisper سبک import را عوض کند، به‌جای کرشِ رونویسی فقط هشدار داده
    می‌شود و رونویسی بدون لاگ پیشرفت ادامه می‌یابد. سنتل هم تضمین می‌کند اگر attribute
    از ابتدا وجود نداشته باشد، در finally چیزی مثل «tqdm=None» در ماژول باقی نماند.
    """
    global _tqdm_struct_warned  # هشدار ساختار tqdm در هر سشن فقط یک‌بار (الگوی FB-18)
    import whisper.transcribe as _wt
    _MISSING = object()  # فیکس FB-09: سنتل به‌جای None — تمایز «موجود ولی None» از «غیرموجود»
    original = getattr(_wt, "tqdm", _MISSING)
    patched = False
    # فقط وقتی پچ کن که ساختار فعلی همان سبک موردانتظار است؛ در غیر این صورت رونویسی
    # دست‌نخورده و سالم ادامه می‌یابد (رفتار پیشین: پچ کورکورانه = کرش در سبک دیگر)
    if original is _MISSING or not callable(getattr(original, "tqdm", None)):
        if not _tqdm_struct_warned:
            _tqdm_struct_warned = True
            logger.warning("ساختار tqdm در این نسخهٔ whisper موردانتظار (module-style) نیست؛ لاگ پیشرفت رونویسی غیرفعال شد اما خود رونویسی بدون مشکل ادامه می‌یابد. (این هشدار یک‌بار چاپ می‌شود)")
    else:
        try:
            _wt.tqdm = types.SimpleNamespace(tqdm=_LoggingTqdm)
            patched = True
        except Exception as e:
            if not _tqdm_struct_warned:
                _tqdm_struct_warned = True
                logger.warning(f"فعال‌سازی لاگ پیشرفت رونویسی ممکن نشد (رونویسی بدون آن ادامه می‌یابد): {e} (این هشدار یک‌بار چاپ می‌شود)")
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

    safe_name = "".join(c for c in base_filename if c not in r'\/:*?"<>|').strip() or "audio"
    audio_path = os.path.join(SESSION_AUDIO_DIR, f"{safe_name}.wav")
    if os.path.exists(audio_path):
        logger.warning("فایل صوتی هم‌نام از قبل در سشن کولب وجود دارد؛ نام جدید تولید می‌شود.")
        counter = 1
        while os.path.exists(audio_path):
            audio_path = os.path.join(SESSION_AUDIO_DIR, f"{safe_name}_{counter}.wav")
            counter += 1

    # فیکس FB-33: اگر ورودی از قبل WAV 16kHz مونوی pcm_s16le باشد (دقیقاً همان خروجی
    # استاندارد این تابع)، transcode تکراری فقط زمان و دیسک سشن هدر می‌دهد؛ فقط کپی محلی.
    # ⚠️ هرگز media_path اصلی return نمی‌شود — finally آن را حذف می‌کند و فایل ورودی
    # کاربر از بین می‌رود؛ همیشه کپی در SESSION_AUDIO_DIR ساخته می‌شود.
    try:
        if media_path.lower().endswith(".wav"):
            _probe_info = ffmpeg.probe(media_path)
            _stream = _probe_info["streams"][0] if _probe_info.get("streams") else {}
            if (
                _stream.get("codec_name") == "pcm_s16le"
                and str(_stream.get("sample_rate")) == "16000"
                and _stream.get("channels") == 1
            ):
                shutil.copy2(media_path, audio_path)
                logger.step("ورودی از قبل WAV 16kHz مونوی سازگار بود؛ فقط کپی شد (بدون transcode).")  # جزئیات گام‌به‌گام — فقط فایل لاگ
                return audio_path
    except Exception:
        pass  # فیکس FB-33: خطای probe → مسیر عادی transcode ادامه می‌یابد

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
    # فیکس NB-24: پاک‌سازی کاراکترهای غیرمجاز فایل‌سیستم در نام خروجی‌ها
    safe_base = sanitize_filename(base_filename, fallback="media")
    suffix = f"- Whisper [{model_label} Model]"
    names = {
        'srt': f"{safe_base}{suffix}.srt",
        'txt': f"{safe_base}{suffix}.txt",
        'txt_tl': f"{safe_base}{suffix}_timeline.txt",
        'meta': f"{safe_base}_Metadata.txt",
        'zip': f"{safe_base}_Whisper_Transcription.zip",
    }
    paths = {k: os.path.join(work_dir, v) for k, v in names.items()}

    total_chars = 0
    # فیکس FB-15: کلمات هم مانند line_count و total_chars فقط برای سگمنت‌های
    # نوشته‌شده شمرده می‌شوند تا آمار متادیتا با فیلتر متن‌خالی سازگار بماند
    total_words = 0
    line_count = 0  # فیکس P4-5: فقط سگمنت‌های دارای متن شمرده می‌شوند
    timeline_start = None  # فیکس NB-28: شروع اولین سگمنت دارای متن
    timeline_end = None    # فیکس NB-28: پایان آخرین سگمنت دارای متن
    with open(paths['srt'], 'w', encoding='utf-8') as srt_f, \
          open(paths['txt'], 'w', encoding='utf-8') as txt_f, \
          open(paths['txt_tl'], 'w', encoding='utf-8') as txt_tl_f:
        for seg in segments:
            text = seg['text'].strip()
            # فیکس P4-5: سگمنت با متن خالی کیوی خالی در SRT تولید می‌کرد؛ رد می‌شود
            if not text:
                continue
            start_s, end_s = seg['start'], seg['end']
            if timeline_start is None:
                timeline_start = start_s  # فیکس NB-28
            timeline_end = end_s          # فیکس NB-28
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

    if timeline_start is None:
        timeline_start = 0.0
    if timeline_end is None:
        timeline_end = 0.0
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
        # فیکس NB-04: فیکس FB-14 فقط ایمیل را درست کرد؛ متادیتای ماندگار (داخل ZIP و درایو)
        # هم نباید مسیر WAV حذف‌شده را «موجود» جلوه دهد — همان الگوی FB-14 اینجا هم اعمال شد
        # (statement شرطیِ جدا؛ درج عبارت شرطی داخل بلوک پرانتزی الحاق رشته در پایتون
        # به‌صورت call روی رشته parse می‌شود و کرش می‌آورد — تجربهٔ فیکس FB-14)
        if KEEP_SESSION_WAVS:
            meta_f.write(f"مسیر فایل صوتی (فقط داخل سشن کولب - در درایو ذخیره نمی‌شود): {audio_path}\n\n")
        else:
            meta_f.write("فایل WAV سشن پس از پردازش موفق حذف شد (برای ثبت مسیر، KEEP_SESSION_WAVS را در بخش ۲ True کنید).\n\n")

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
        # فیکس FB-31: پارامترهای کیفی رونویسی برای بازتولیدپذیری ثبت می‌شوند
        meta_f.write(f"پارامترهای رونویسی: language={TRANSCRIBE_LANGUAGE}, fp16={device == 'cuda'}, condition_on_previous_text=False, verbose=False\n")
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

import re  # فیکس NB-27: پشتیبانی از کامای فارسی و سمی‌کالن در جداکننده‌ها

def _recipients():
    """فیکس NB-05 + NB-27: RECEIVER_EMAIL می‌تواند چند گیرنده با کاما (انگلیسی یا فارسی) یا سمی‌کالن باشد.
    پارس به لیست آدرس‌های معتبر جهت جلوگیری از خطای SMTP."""
    if not RECEIVER_EMAIL:
        return []
    return [r.strip() for r in re.split(r"[,;،]", RECEIVER_EMAIL) if r.strip()]

def send_email(subject, body, max_retries=3):
    # فیکس NB-17: Secretها فقط یک‌بار در سلول ۳ خوانده می‌شدند؛ افزودن Secret در میانهٔ سشن
    # (تلاش مجدد سلول ۵ بدون اجرای دوبارهٔ سلول ۳) بی‌اثر بود. مقدارهای غایب فقط وقتی پیکربندی
    # ناقص است دوباره خوانده می‌شوند (هزینهٔ userdata.get ناچیز است)
    global SENDER_EMAIL, SENDER_APP_PASSWORD, RECEIVER_EMAIL
    if not (SENDER_EMAIL and SENDER_APP_PASSWORD and RECEIVER_EMAIL):
        if not SENDER_EMAIL:
            SENDER_EMAIL = get_secret('SENDER_EMAIL')
        if not SENDER_APP_PASSWORD:
            SENDER_APP_PASSWORD = get_secret('SENDER_APP_PASSWORD')
        if not RECEIVER_EMAIL:
            RECEIVER_EMAIL = get_secret('RECEIVER_EMAIL')
        if SENDER_EMAIL and SENDER_APP_PASSWORD and RECEIVER_EMAIL:
            logger.info("تنظیمات ایمیل در میانهٔ اجرا فعال شد.")
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
    # فیکس NB-27: استفاده از لیست گیرندگان معتبر در هدر To با کامای استاندارد
    msg['To'] = ", ".join(_recipients())

    for attempt in range(1, max_retries + 1):
        try:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=30) as server:  # فیکس FB-29
                server.login(SENDER_EMAIL, SENDER_APP_PASSWORD)
                server.sendmail(SENDER_EMAIL, _recipients(), msg.as_string())  # فیکس NB-05: لیست پارس‌شدهٔ گیرنده‌ها
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
            _ts = time.strftime('%Y%m%d_%H%M%S')
            dest = os.path.join(dest_dir, f"whisper_processing_log_{_ts}.txt")
            shutil.copy2(LOG_FILE_PATH, dest)
            # فیکس NB-03: نسخهٔ چرخیدهٔ لاگ (.1 — فیکس FB-13) محتوای قدیمیِ سشن‌های
            # طولانی است و هرگز به درایو نمی‌رسید؛ با نام part2 جداگانه بکاپ می‌شود
            # (مقدار برگشتی همان مسیر فایل اصلی می‌ماند تا فراخوان‌ها تغییر نکنند)
            try:
                _rotated_log = LOG_FILE_PATH + ".1"
                if os.path.exists(_rotated_log):
                    shutil.copy2(_rotated_log, os.path.join(dest_dir, f"whisper_processing_log_{_ts}_part2.txt"))
            except Exception as e:
                logger.warning(f"کپی نسخهٔ چرخیدهٔ لاگ (.1) به درایو ناموفق بود: {e}")
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

# فیکس NB-01: وضعیت جاری بچ در سطح ماژول نگه داشته می‌شود تا هنگام توقف کاربر
# (KeyboardInterrupt) گزارش نتایج تا لحظهٔ توقف و بکاپ لاگ از بیرونِ process_all ممکن باشد
_batch_state = {"total": 0, "done": 0, "success": 0, "fail": 0, "results": []}

# فیکس NB-12: خطای زیرساختی (دیسک/درایو) — نه خطای محتوایی فایل؛ زیرکلاس RuntimeError
# تا در هندلر فعلی (ValueError, RuntimeError) بیفتد و re-indent لازم نشود
class SystemicBatchError(RuntimeError):
    """خطای زیرساختی (دیسک/درایو) که ادامهٔ بچ را بی‌معنا می‌کند."""


def process_all(pairs):
    # فیکس P4-7: اعتبارسنجی ساختار pairs و هشدار لیست خالی پیش از شروع
    # فیکس NB-06: فراخوان دومِ اعتبارسنجی بی‌پیام است (quiet=True) تا در هر تلاش
    # مجددِ سلول ۵ هشدارهای تکراری دوباره چاپ نشوند؛ شمارش ردشدگی‌ها (FB-32) انجام می‌شود
    pairs, _run_rejects = validate_pairs(pairs, quiet=True)
    # فیکس NB-02: پوشهٔ خروجی نامعتبر/غیرقابل‌نوشتن قبلاً فقط بعد از رونویسی کشف می‌شد
    # (os.makedirs داخل حلقه) و با خطای آن، خروجی کاملِ آماده در finally با rmtree نابود
    # می‌شد — قبل از ساختن نسخهٔ بازیابی FB-05. حالا همهٔ پوشه‌های خروجی در شروع بچ
    # ساخته و write-test می‌شوند و جفت نامعتبر پیش از مصرف GPU رد می‌شود (fail-fast).
    usable_pairs = []
    output_rejects = []  # فیکس NB-11: علت‌های ردشدگی پوشهٔ خروجی برای گزارش نهایی
    for in_path, out_folder in pairs:
        if not ensure_output_folder(out_folder):
            logger.warning(f"پوشهٔ خروجی «{out_folder}» قابل ساخت/نوشتن نیست؛ جفت «{os.path.basename(in_path)}» پیش از پردازش رد شد.")
            output_rejects.append(f"{os.path.basename(in_path)} → پوشهٔ خروجی قابل ساخت/نوشتن نیست: {out_folder}")  # فیکس NB-11
            continue
        usable_pairs.append((in_path, out_folder))
    pairs = usable_pairs
    total = len(pairs)
    results = []
    file_durations = []  # فقط موفق‌ها (مبنای median برای ETA)
    all_file_durations = []  # فیکس FB-11: همهٔ فایل‌ها (موفق + ناموفق) برای میانگین جدا در گزارش نهایی
    done_records = []  # فیکس P4-10: (زمان پردازش، مدت رسانه) برای فایل‌های موفق با مدت معلوم
    success_count = 0
    fail_count = 0
    batch_start = time.time()
    _batch_state.update(total=total, done=0, success=0, fail=0, results=[])  # فیکس NB-01: ریست وضعیت بچ در شروع هر اجرا
    consecutive_systemic = 0  # فیکس NB-12: شمارندهٔ شکست سیستمی متوالی (دیسک/درایو)
    aborted_systemic = False  # فیکس NB-12: توقف زودهنگام بچ به‌علت شکست سیستمی متوالی

    if total == 0:
        logger.warning("لیست فایل‌های ورودی خالی است؛ چیزی برای پردازش وجود ندارد. input_output_pairs را تنظیم کنید (یا از build_pairs_from_folder استفاده کنید).")
        # فیکس NB-15: قبلاً هیچ ایمیلی ارسال نمی‌شد و کاربرِ فقط-ایمیل بی‌خبر می‌ماند؛
        # حالا یک ایمیل کوتاه با علت‌های ردشدگی ارسال می‌شود (بدون پیکربندی ایمیل، بی‌صدا رد می‌شود)
        _zero_body = "هیچ جفت معتبری برای پردازش باقی نمانده است.\n\n"
        _zero_rejects = _run_rejects if _run_rejects else PAIRS_REJECTS
        # فیکس NB-23: علت‌های ردشدگی شامل پوشهٔ خروجی نامعتبر هم می‌شود تا کاربر گمراه نشود
        if _zero_rejects:
            _zero_body += f"علت‌های ردشدگی ساختار/پسوند ({len(_zero_rejects)} مورد؛ حداکثر ۵ مورد):\n"
            _zero_body += "\n".join(f"  - {_r}" for _r in _zero_rejects[:5]) + "\n"
        if output_rejects:
            _zero_body += f"علت‌های ردشدگی پوشهٔ خروجی ({len(output_rejects)} مورد؛ حداکثر ۵ مورد):\n"
            _zero_body += "\n".join(f"  - {_orj}" for _orj in output_rejects[:5]) + "\n"
        if not _zero_rejects and not output_rejects:
            _zero_body += "input_output_pairs را تنظیم کنید (یا از build_pairs_from_folder استفاده کنید).\n"
        send_email("⚠️ هیچ جفت معتبری برای پردازش نبود", _zero_body)
        return

    logger.info(f"====== شروع پردازش {total} فایل ======")

    # فیکس FB-37: چکِ وجودِ ورودی که اینجا تکراری بود حذف شد — precheck سلول ۳ قبل از
    # شروع اطلاع‌رسانی می‌کند و raise داخل حلقه مرجعِ خطاست (چک extract_audio هم دفاعی می‌ماند)

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
        systemic = False  # فیکس NB-12: آیا خطای این فایل زیرساختی بود؟
        _in_transfer_phase = False  # فیکس NB-12: تا ورود به بخش انتقال، خطاها سیستمی حساب نمی‌شوند

        try:
            if not os.path.exists(input_path):
                raise FileNotFoundError(f"فایل ورودی یافت نشد: {input_path}")

            logger.step("در حال بررسی مدت‌زمان دقیق فایل رسانه...")
            media_duration = media_durations.get(input_path)  # فیکس P4-10: از نتیجهٔ ffprobe پیش از حلقه استفاده می‌شود
            if media_duration is not None:
                logger.step(f"مدت رسانه: {format_srt_time(media_duration)[:8]}")  # جزئیات گام‌به‌گام — فقط فایل لاگ

            # فیکس P3-4: گارد فضای دیسک سشن قبل از استخراج صدا
            free_gb = shutil.disk_usage(SESSION_AUDIO_DIR).free / (1024 ** 3)  # فیکس FB-03: دیسکِ محل ذخیرهٔ WAV (خارج از Colab هم کار می‌کند)
            if free_gb < 2.0:
                # فیکس NB-12: کمبود دیسک خطای زیرساختی است، نه خطای محتوای فایل
                raise SystemicBatchError(f"فضای خالی دیسک سشن کولب کمتر از 2 گیگابایت است ({free_gb:.1f} GB)؛ ابتدا فضا آزاد کنید و دوباره تلاش کنید.")
            logger.step("در حال استخراج صدا (WAV در سشن کولب ذخیره می‌شود)...")
            audio_path = extract_audio(input_path, base_filename)
            logger.step(f"صدا استخراج شد و در سشن کولب ذخیره ماند (نه درایو): {audio_path}")  # فیکس NB-19: جزئیات گام‌به‌گام — فقط فایل لاگ

            logger.step("در حال رونویسی با Whisper (بسته به طول فایل ممکن است چند دقیقه طول بکشد)...", force=True)  # پیام گامیِ مهم — حتی در حالت کم‌جزئیات چاپ می‌شود
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
            logger.step(f"رونویسی کامل شد: {len(segments)} لاین شناسایی شد.")  # فیکس NB-19: جزئیات گام‌به‌گام — فقط فایل لاگ

            logger.step("در حال تولید فرمت‌های خروجی (SRT/TXT) و متادیتا...")
            stats = save_all_formats(segments, base_filename, work_dir, input_path,
                                      MODEL_LABEL, media_duration, audio_path,
                                      file_start)  # فیکس FB-17: زمان داخل تابع و در لحظۀ نوشتن متادیتا نهایی می‌شود

            _in_transfer_phase = True  # فیکس NB-12: خطای OSError در بخش انتقال «سیستمی» است (قطع درایو)
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
                    # فیکس FB-34: os.replace اتمیک به‌جای exists/remove/rename — همان race چرخش لاگ
                    # اینجا هم بود (حذف ناکامل .bak → rename بی‌صدا رد می‌شد)؛ بازنویسی .bak موجود هم امن است
                    os.replace(dest_path, bak_path)  # فیکس FB-05: حذف مستقیم مقصد در صورت خطای move هر دو نسخه را از بین می‌برد
                    try:
                        shutil.move(src_path, dest_path)
                        os.remove(bak_path)  # انتقال موفق؛ نسخهٔ قدیمی حذف می‌شود
                        logger.step(f"خروجی قبلی بازنویسی شد: {fname}")  # جزئیات گام‌به‌گام — فقط فایل لاگ
                    except Exception:
                        # فیکس NB-21: اگر shutil.move فایل ناقصی در مقصد ساخته باشد، os.rename با FileExistsError کرش می‌کند؛
                        # os.replace نسخهٔ پشتیبان .bak را بازمی‌گرداند و خطای اصلی پنهان نمی‌شود
                        try:
                            os.replace(bak_path, dest_path)
                        except OSError:
                            pass
                        raise
                else:
                    # فیکس NB-09: انتقال بین‌فایل‌سیستم‌ی (سشن → درایو) در واقع
                    # copy+unlink است؛ اگر copy وسط کار بخورد، فایل نیمه‌کاره با نام
                    # درست در مقصدِ تازه می‌ماند. فایل ناقص پاک و خطا دوباره raise می‌شود
                    # (نسخهٔ امن از قبل در پوشهٔ بازیابی FB-05 کپی شده است)
                    try:
                        shutil.move(src_path, dest_path)
                    except Exception:
                        try:
                            os.remove(dest_path)
                        except OSError:
                            pass
                        raise
            shutil.rmtree(transfer_recovery_dir, ignore_errors=True)  # فیکس FB-05: انتقال کامل شد؛ نسخهٔ بازیابی این فایل لازم نیست
            _in_transfer_phase = False  # فیکس NB-12: انتقال تمام شد؛ خطاهای بعدیِ این فایل سیستمی نیست

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
            # فیکس NB-14: روی torch قدیمی (fallback = RuntimeError) این هندلر همهٔ RuntimeErrorها
            # (خطای ffmpeg، گارد دیسک NB-12 و…) را می‌گرفت و به‌عنوان «کمبود حافظه GPU» گزارش می‌کرد؛
            # در حالت fallback فقط خطای با نشانهٔ OOM در پیام مسیر OOM را طی می‌کند و بقیه به همان
            # مسیر عادی هندلر (ValueError, RuntimeError) می‌روند (if/else — نه except جدید)
            if OOM_FALLBACK_ACTIVE and not ("out of memory" in str(e).lower() or "cuda" in str(e).lower()):
                duration = time.time() - file_start
                systemic = isinstance(e, SystemicBatchError)  # فیکس NB-12: خطای زیرساختی (گارد دیسک)
                status, error_detail = "ناموفق", f"{type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
                fail_count += 1
                logger.error(f"خطا در پردازش فایل: {e}")
                results.append((input_path, f"ناموفق - {type(e).__name__}: {str(e)[:120]}"))
            else:
                duration = time.time() - file_start
                status, error_detail = "ناموفق", f"کمبود حافظه GPU (Out of Memory) — {type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
                fail_count += 1
                logger.error(f"خطای کمبود حافظه GPU حین پردازش «{base_filename}»؛ این فایل رد شد و پردازش با فایل بعدی ادامه می‌یابد.")
                try:
                    torch.cuda.empty_cache()
                    logger.step("حافظه CUDA پاک‌سازی شد.")  # جزئیات گام‌به‌گام — فقط فایل لاگ
                except Exception:
                    pass
                results.append((input_path, "ناموفق - کمبود حافظه GPU (OOM)"))
        except (ValueError, RuntimeError) as e:
            duration = time.time() - file_start
            systemic = isinstance(e, SystemicBatchError)  # فیکس NB-12: خطای زیرساختی (گارد دیسک)
            status, error_detail = "ناموفق", f"{type(e).__name__}: {str(e)[:200]}"  # فیکس P3-10
            fail_count += 1
            logger.error(f"خطا در پردازش فایل: {e}")
            results.append((input_path, f"ناموفق - {type(e).__name__}: {str(e)[:120]}"))
        except Exception as e:
            duration = time.time() - file_start
            systemic = _in_transfer_phase and isinstance(e, OSError)  # فیکس NB-12: شکست انتقال به درایو خطای سیستمی است
            # فیکس FB-30: سه خط آخر traceback به ایمیل افزوده می‌شود تا علت خطای
            # غیرمنتظره بدون مراجعه به فایل لاگ قابل تشخیص باشد
            _tb_tail = "\n".join(traceback.format_exc().strip().splitlines()[-3:])
            status, error_detail = "ناموفق", f"{type(e).__name__}: {str(e)[:200]}\n{_tb_tail}"  # فیکس P3-10 + FB-30
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
        logger.step(f"خلاصه تا این لحظه: {success_count} موفق، {fail_count} ناموفق از {idx} فایل بررسی‌شده")  # جزئیات گام‌به‌گام — فقط فایل لاگ

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

        # فیکس NB-01: وضعیت بچ پس از پایان هر فایل به‌روز می‌شود تا گزارش هنگام توقف کاربر دقیق باشد
        _batch_state.update(success=success_count, fail=fail_count, done=idx, results=list(results))
        # فیکس NB-12: شمارش شکست سیستمی متوالی — فقط موفقیت ریست می‌کند؛ شکست معمولی (OOM و…) بی‌اثر است
        if status == "موفق":
            consecutive_systemic = 0
        elif systemic:
            consecutive_systemic += 1
        # فیکس NB-12: توقف بچ پس از شکست سیستمی متوالی — اینجا (بعد از ایمیل و ثبت وضعیت) می‌شکند
        if ABORT_AFTER_SYSTEMIC_FAILURES and consecutive_systemic >= ABORT_AFTER_SYSTEMIC_FAILURES:
            logger.error(f"{consecutive_systemic} شکست سیستمی متوالی (دیسک/درایو)؛ ادامهٔ بچ فقط GPU را هدر می‌دهد — بچ متوقف می‌شود.")
            aborted_systemic = True
            break

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
    # فیکس FB-32: جفت‌های ردشدهٔ اعتبارسنجی در گزارش نهایی هم می‌آیند تا کاربری که فقط
    # ایمیل نهایی را می‌خواند از کم‌بودنِ تعداد فایل‌ها بی‌خبر نماند. در جریان عادی نوت‌بوک،
    # ردشدگی‌ها مربوط به precheck سلول ۳ هستند (PAIRS_REJECTS) چون ورودیِ process_all
    # از قبل فیلتر شده؛ در فراخوان مستقیم با لیست خام، ردشدنی‌های همین اجرا (_run_rejects)
    # معتبرترند و مقدم می‌شوند
    _summary_rejects = _run_rejects if _run_rejects else PAIRS_REJECTS
    if _summary_rejects:
        summary_header += f"جفت‌های ردشده در اعتبارسنجی: {len(_summary_rejects)}\n"
        for _rj in _summary_rejects[:5]:
            summary_header += f"  - {_rj}\n"
        if len(_summary_rejects) > 5:
            summary_header += f"  - و {len(_summary_rejects) - 5} مورد دیگر (جزئیات در خروجی سلول ۳)\n"
    # فیکس NB-12: اگر بچ به‌علت شکست سیستمی متوالی متوقف شد، در گزارش نهایی ذکر می‌شود
    if aborted_systemic:
        summary_header += f"⛔ بچ به‌علت {consecutive_systemic} شکست سیستمی متوالی (دیسک/درایو) زودتر از پایان لیست متوقف شد.\n"
    # فیکس NB-11: ردشدگی‌های پوشهٔ خروجی هم مثل ردشدگی‌های اعتبارسنجی (FB-32) به ایمیل نهایی می‌آیند
    if output_rejects:
        summary_header += f"جفت‌های ردشده به‌علت پوشهٔ خروجی نامعتبر: {len(output_rejects)}\n"
        for _orj in output_rejects[:5]:
            summary_header += f"  - {_orj}\n"
        if len(output_rejects) > 5:
            summary_header += f"  - و {len(output_rejects) - 5} مورد دیگر\n"
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
    # فیکس NB-01: قبلاً فقط یک ایمیل ساده فرستاده می‌شد؛ نه بکاپ لاگ به درایو اجرا می‌شد
    # نه گزارش نتایج تا لحظهٔ توقف — و با پایان سشن همه‌چیز از بین می‌رفت. حالا گزارش
    # کامل نتایج + بکاپ لاگ + دنبالهٔ لاگ در همان ایمیل توقف ارسال می‌شود
    _interrupt_backup = backup_log_to_drive()
    _interrupt_tail = read_log_tail(30)
    _st = _batch_state
    _interrupt_body = (
        f"پردازش دسته‌ای توسط کاربر متوقف شد (KeyboardInterrupt).\n"
        f"تا لحظهٔ توقف: {_st['success']} موفق | {_st['fail']} ناموفق از {_st['done']} فایل بررسی‌شده (کل لیست: {_st['total']})\n\n"
    )
    if _st['results']:
        _interrupt_body += "جزئیات فایل‌های بررسی‌شده:\n" + "\n".join(f"{os.path.basename(p)}: {s}" for p, s in _st['results']) + "\n\n"
    _interrupt_body += f"نسخهٔ پشتیبان لاگ: {_interrupt_backup if _interrupt_backup else 'ساخته نشد (متن لاگ فقط داخل سشن است)'}\n\n{'=' * 40}\n{_interrupt_tail}"
    send_email("⛔ پردازش متوقف شد", _interrupt_body)
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
