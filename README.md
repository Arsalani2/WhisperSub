<div dir="rtl">

# 🎙️ WhisperSub

**رونویسی دسته‌ای ویدیو و صوت به متن با OpenAI Whisper (large-v3)** — نوت‌بوکی آماده برای اجرا در **Google Colab**

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/Arsalani2/WhisperSub/blob/main/notebook/whispersub.ipynb)
[![نسخه](https://img.shields.io/github/v/release/Arsalani2/WhisperSub.svg?style=flat-square&label=%D9%86%D8%B3%D8%AE%D9%87)](https://github.com/Arsalani2/WhisperSub/releases/latest)
[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![مجوز](https://img.shields.io/badge/%D9%85%D8%AC%D9%88%D8%B2-MIT-yellow?style=flat-square)](LICENSE)

> یک پوشهٔ ویدیو را به زیرنویس SRT و متن ساده تبدیل کنید؛ اعلان ایمیلی پیشرفت بگیرید و گزارش نهایی را دریافت کنید. تمام پردازش روی GPU رایگان Colab انجام می‌شود.

---

## 🧭 معرفی

WhisperSub یک نوت‌بوک Jupyter است که چندین فایل ویدیویی/صوتی را با مدل **Whisper large-v3** به‌صورت دسته‌ای رونویسی می‌کند، برای هر فایل زیرنویس و متن تولید می‌کند، خروجی‌ها را در Google Drive ذخیره می‌کند و روند پیشرفت را از طریق ایمیل گزارش می‌دهد. کد برای اجرا در Colab (ترجیحاً GPU) نوشته شده و در صورت خطا (مثلاً کمبود حافظهٔ GPU) پردازش با فایل بعدی ادامه می‌یابد.

## ✨ امکانات

| | قابلیت |
|:---:|---|
| 🎯 | رونویسی دسته‌ای چند فایل با Whisper `large-v3` (زبان پیش‌فرض: انگلیسی) |
| 🔊 | استخراج خودکار صدا با ffmpeg — کش WAV فقط در سشن Colab می‌ماند و به درایو نمی‌رود |
| 📄 | خروجی SRT + TXT (با و بدون تایم‌لاین) + متادیتا + بستهٔ ZIP برای هر فایل |
| ☁️ | ذخیرهٔ خروجی‌ها در Google Drive |
| 📧 | اعلان ایمیلی (Gmail SMTP) بعد از هر فایل + گزارش نهایی |
| 🛡️ | ادامهٔ پردازش در صورت خطا؛ تخمین زمان باقی‌مانده؛ لاگ کامل اجرا |

## 📂 ساختار مخزن

```
WhisperSub/
├── notebook/
│   └── whispersub.ipynb      ← 🎯 منبع اصلی پروژه — فقط همین فایل ادیت شود
├── src/
│   └── whispersub.py         ← ⚙️ تولیدشدهٔ خودکار از نوت‌بوک — ادیت دستی نشود
├── docs/
│   └── bug.md                ← 🐞 گزارش باگ‌ها و کارهای باقی‌مانده
├── .github/
│   ├── workflows/
│   │   └── release.yml       ← 🏷️ ورک‌فلو ریلیز خودکار
│   └── scripts/
│       └── build_release.py ← 📝 درج اطلاعات نسخه + تولید py
├── VERSION                   ← نسخهٔ فعلی پروژه (فرمت 1.0.x)
├── requirements.txt
├── README.md
└── LICENSE                   ← مجوز MIT
```

| فایل | نقش |
|---|---|
| [`notebook/whispersub.ipynb`](notebook/whispersub.ipynb) | **منبع اصلی** — نوت‌بوک اجرایی پروژه |
| [`src/whispersub.py`](src/whispersub.py) | نسخهٔ اسکریپتی، تولیدشدهٔ خودکار از نوت‌بوک |
| [`docs/bug.md`](docs/bug.md) | باگ‌های مستندشده (P1 تا P4) |
| [`VERSION`](VERSION) | نسخهٔ فعلی پروژه (`1.0.x`) |
| [`.github/workflows/release.yml`](.github/workflows/release.yml) | ریلیز خودکار در هر پوش به `main` |

## 🚀 شروع سریع

### ۱. باز کردن نوت‌بوک

نوت‌بوک اصلی در مسیر [`notebook/whispersub.ipynb`](notebook/whispersub.ipynb) قرار دارد:

- در Colab: **File → Open notebook → GitHub** و آدرس این مخزن، یا کلیک روی دکمهٔ «Open in Colab» در بالا
- Runtime را روی GPU بگذارید: **Runtime → Change runtime type → GPU**
- بخش‌های نوت‌بوک را به ترتیب (۱ تا ۴) اجرا کنید

> 💡 مدل `large-v3` حدود **۱۱ گیگابایت** VRAM می‌خواهد. اگر GPU کم‌حافظه دارید، `WHISPER_MODEL_NAME` را به `turbo` یا `medium` تغییر دهید.

### ۲. تنظیم Secrets ایمیل

در Colab از بخش **Secrets** (آیکون 🔑) سه مقدار زیر را اضافه و دسترسی Notebook را برای هرکدام فعال کنید:

| نام Secret | توضیح |
|---|---|
| `SENDER_EMAIL` | آدرس Gmail فرستنده |
| `SENDER_APP_PASSWORD` | App Password جیمیل — نه رمز اصلی حساب |
| `RECEIVER_EMAIL` | آدرس ایمیل گیرندهٔ گزارش‌ها |

> ℹ️ این Secretها **اختیاری** هستند؛ اگر تنظیم نشوند، برنامه بدون اعلان ایمیلی به‌طور کامل اجرا می‌شود (فیکس P1-1؛ جزئیات در [`docs/bug.md`](docs/bug.md)).

### ۳. معرفی فایل‌های ورودی

لیست فایل‌ها را در بخش ۲ نوت‌بوک تنظیم کنید. هر جفت به شکل `(مسیر فایل ورودی، پوشهٔ خروجی)` است:

```python
input_output_pairs = [
    ('/content/drive/MyDrive/Videos/lesson1.mp4', '/content/drive/MyDrive/Subs/lesson1'),
]
```

برای ساخت خودکار لیست از یک پوشه:

```python
input_output_pairs = build_pairs_from_folder(source_folder, destination_root)
```

### فرمت‌های پشتیبانی‌شده

| نوع | پسوندها |
|---|---|
| 🎬 ویدیو | `.mp4` `.mkv` `.avi` `.mov` `.webm` `.wmv` `.flv` `.ts` |
| 🎵 صوت | `.wav` `.mp3` `.flac` `.ogg` `.aac` `.m4a` `.amr` `.aiff` `.wma` `.alac` |

## 📦 خروجی‌ها

برای هر فایل ورودی، در پوشهٔ خروجی مربوطه در Google Drive:

```
📁 <پوشهٔ خروجی>
├── <name>- Whisper [Large-v3 Model].srt          ← زیرنویس استاندارد
├── <name>- Whisper [Large-v3 Model].txt           ← متن ساده (بدون تایم‌لاین)
├── <name>- Whisper [Large-v3 Model]_timeline.txt  ← متن ساده (با تایم‌لاین)
├── <name>_Metadata.txt                            ← گزارش کامل متادیتا
└── <name>_Whisper_Transcription.zip               ← بستهٔ همهٔ خروجی‌ها
```

متادیتا شامل مدت‌زمان، تعداد لاین/کلمه/کاراکتر، حجم فایل‌ها، زمان پردازش و فهرست فایل‌های تولیدی است.

## ⚙️ تنظیمات مهم

| تنظیم | محل | توضیح |
|---|---|---|
| `WHISPER_MODEL_NAME` | بخش ۲ | مدل Whisper (`tiny` … `large-v3`، `turbo`) |
| `language='en'` | بخش ۴ (`model.transcribe`) | زبان گفتار؛ برای زبان دیگر تغییر دهید یا حذف کنید تا خودکار تشخیص داده شود |
| `condition_on_previous_text=False` | بخش ۴ | جلوگیری از تکرار/هذیان مدل در سکوت یا نویز طولانی |
| `SESSION_AUDIO_DIR` | بخش ۲ | مسیر کش WAV در سشن Colab (پس از پایان سشن پاک می‌شود) |

## 🏷️ نسخه‌گذاری و ریلیز خودکار

- نام فایل‌ها **بدون شمارهٔ نسخه** است؛ نسخهٔ فعلی در [`VERSION`](VERSION) و در کامنت ابتدای هر دو فایل نگهداری می‌شود.
- با هر پوش به `main` (بدون برچسب `[skip ci]`) ورک‌فلو ریلیز به‌طور خودکار:
  1. نسخهٔ پچ (`1.0.x`) را یک پله بالا می‌برد
  2. کامنت «اطلاعات ریلیز» (نسخه، تاریخ، کامیت، لینک) را در ابتدای **هر دو فایل** درج می‌کند (به‌صورت idempotent)
  3. `src/whispersub.py` را از روی نوت‌بوک بازتولید می‌کند
  4. کامیت نسخه را با `[skip ci]` ثبت و **ریلیز GitHub** را با هر دو فایل پیوست‌شده می‌سازد
- کامیت‌هایی که نباید ریلیز بسازند (مستندات، تنظیمات و …) با `[skip ci]` علامت می‌خورند.

## 🐞 باگ‌های شناخته‌شده

گزارش کامل ۲۶ باگ مستندشده (P1 تا P4) در [`docs/bug.md`](docs/bug.md) است:

- ✅ ۱۲ مورد فیکس یا منتفی شده — از جمله ترمیم JSON نامعتبر نوت‌بوک، حذف خروجی‌های VTT/ASS/PDF/DOCX، حذف دوتایی‌شدن منابع `py`/`ipynb`، فیکس P1-1 (کرش بدون Secrets ایمیل)، P2-2 (حذف مسیرهای شخصی) و P2-1 (خطای اجرای مجدد)
- ⬜ ۱۷ مورد باز — همه در سطح P3/P4؛ بدون مورد بحرانی یا بالا

## 🤝 قواعد توسعه

- **فقط `notebook/whispersub.ipynb` را ادیت کنید**؛ فایل `py` هر بار در ریلیز از روی نوت‌بوک بازتولید می‌شود و ادیت دستی از بین می‌رود.
- پیام‌های کامیت فارسی و به سبک Conventional Commits نوشته می‌شوند (مثل `fix:` و `docs:`).
- کامیت‌های غیرِریلیز با `[skip ci]` علامت می‌خورند تا ورک‌فلو اجرا نشود.
- هر باگ پس از فیکس در `docs/bug.md` تیک می‌خورد (`- [x]`) و هش کامیت فیکس کنارش ثبت می‌شود.

## 📄 مجوز

این پروژه تحت مجوز [MIT](LICENSE) منتشر شده است.

</div>
