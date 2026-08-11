# ============================================
# FACERO — БОТ ДЛЯ ОЦЕНКИ ВНЕШНОСТИ
# ДЛЯ JUSTRUNMY.APP
# ============================================

import os
import logging
import sqlite3
from datetime import datetime, timedelta
from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters, ContextTypes
import numpy as np
import cv2
import mediapipe as mp

# ===== НАСТРОЙКИ =====
TOKEN = "8950596380:AAFb8-lnPD5WTr7fu9W8PSJkdXswG2VBWI8"
PORT = 5000

# ===== ЛОГИРОВАНИЕ =====
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===== ИНИЦИАЛИЗАЦИЯ БАЗЫ ДАННЫХ =====
def init_db():
    conn = sqlite3.connect('facero.db')
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            telegram_id INTEGER UNIQUE,
            username TEXT,
            first_name TEXT,
            requests_today INTEGER DEFAULT 0,
            last_date TEXT,
            extra_requests INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    c.execute('''
        CREATE TABLE IF NOT EXISTS payments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            amount INTEGER,
            requests INTEGER,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()
    logger.info("База данных создана")

# ===== ФУНКЦИИ БАЗЫ ДАННЫХ =====
def get_user(user_id):
    conn = sqlite3.connect('facero.db')
    c = conn.cursor()
    c.execute("SELECT requests_today, last_date, extra_requests FROM users WHERE telegram_id=?", (user_id,))
    row = c.fetchone()
    conn.close()
    if row:
        return {'requests_today': row[0], 'last_date': row[1], 'extra_requests': row[2]}
    return None

def create_user(user_id, username, first_name):
    conn = sqlite3.connect('facero.db')
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("INSERT OR IGNORE INTO users (telegram_id, username, first_name, last_date) VALUES (?, ?, ?, ?)",
              (user_id, username, first_name, today))
    conn.commit()
    conn.close()

def get_remaining(user_id):
    user = get_user(user_id)
    if not user:
        return 5
    today = datetime.now().strftime("%Y-%m-%d")
    if user['last_date'] != today:
        return 5 + user['extra_requests']
    remaining = 5 - user['requests_today'] + user['extra_requests']
    return max(0, remaining)

def use_analysis(user_id):
    conn = sqlite3.connect('facero.db')
    c = conn.cursor()
    today = datetime.now().strftime("%Y-%m-%d")
    c.execute("SELECT last_date, extra_requests FROM users WHERE telegram_id=?", (user_id,))
    row = c.fetchone()
    if row:
        last_date, extra = row
        if last_date != today:
            c.execute("UPDATE users SET requests_today = 0, last_date = ? WHERE telegram_id=?", (today, user_id))
        if extra > 0:
            c.execute("UPDATE users SET extra_requests = extra_requests - 1 WHERE telegram_id=?", (user_id,))
        else:
            c.execute("UPDATE users SET requests_today = requests_today + 1 WHERE telegram_id=?", (user_id,))
    conn.commit()
    conn.close()

def add_extra_requests(user_id, count):
    conn = sqlite3.connect('facero.db')
    c = conn.cursor()
    c.execute("UPDATE users SET extra_requests = extra_requests + ? WHERE telegram_id=?", (count, user_id))
    conn.commit()
    conn.close()

# ===== АНАЛИЗ ЛИЦА (ИСПРАВЛЕННЫЙ) =====
def analyze_face(image_bytes):
    try:
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None, "Не удалось прочитать фото"
        h, w = img.shape[:2]
        scale = min(256/h, 256/w)
        new_w, new_h = int(w * scale), int(h * scale)
        img = cv2.resize(img, (new_w, new_h))
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # ИСПРАВЛЕННЫЙ ИМПОРТ MEDIAPIPE
        from mediapipe.python.solutions import face_mesh as fm
        face_mesh = fm.FaceMesh(
            static_image_mode=True,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=0.5
        )
        
        results = face_mesh.process(rgb)
        if not results.multi_face_landmarks:
            return None, "Лицо не найдено"
        lm = results.multi_face_landmarks[0].landmark

        def eyes():
            left = np.array([lm[33].x, lm[33].y])
            right = np.array([lm[263].x, lm[263].y])
            dist = np.linalg.norm(left - right)
            return round(max(1, min(10, 10 - abs(dist - 0.2) * 40)), 1)

        def symmetry():
            left_x = np.mean([lm[i].x for i in range(0, 468, 2) if i < 468])
            right_x = np.mean([lm[i].x for i in range(1, 468, 2) if i < 468])
            if left_x == 0 and right_x == 0:
                return 7.0
            sym = 1 - abs(left_x - right_x) * 4
            return round(max(1, min(10, 5 + sym * 5)), 1)

        def nose():
            tip = np.array([lm[1].x, lm[1].y])
            root = np.array([lm[168].x, lm[168].y])
            length = np.linalg.norm(tip - root)
            return round(max(1, min(10, 10 - abs(length - 0.3) * 20)), 1)

        def lips():
            top = np.mean([lm[i].y for i in [13, 14, 78, 308]])
            bottom = np.mean([lm[i].y for i in [17, 317, 402, 324]])
            height = abs(top - bottom)
            return round(max(1, min(10, 5 + height * 20)), 1)

        def jaw():
            jaw_pts = [lm[i] for i in range(0, 17)]
            width = max([p.x for p in jaw_pts]) - min([p.x for p in jaw_pts])
            return round(max(1, min(10, 5 + width * 20)), 1)

        def cheeks():
            left = np.mean([lm[i].y for i in [234, 227, 116, 117, 118]])
            right = np.mean([lm[i].y for i in [454, 447, 345, 346, 347]])
            avg = (left + right) / 2
            return round(max(1, min(10, 5 + (0.5 - avg) * 20)), 1)

        def skin():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            return round(max(1, min(10, 3 + (brightness / 255) * 7)), 1)

        def hair():
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            hair_region = gray[:int(h*0.3), :]
            if hair_region.size > 0:
                contrast = np.std(hair_region)
                return round(max(1, min(10, 5 + contrast / 20)), 1)
            return 7.0

        def brows():
            left = np.mean([lm[i].y for i in [46, 53, 52, 65, 55]])
            right = np.mean([lm[i].y for i in [276, 283, 282, 295, 285]])
            avg_y = (left + right) / 2
            return round(max(1, min(10, 10 - abs(avg_y - 0.3) * 20)), 1)

        scores = {
            "Глаза": eyes(),
            "Симметрия": symmetry(),
            "Нос": nose(),
            "Губы": lips(),
            "Челюсть": jaw(),
            "Скулы": cheeks(),
            "Кожа": skin(),
            "Причёска": hair(),
            "Брови": brows()
        }
        total = round(sum(scores.values()) / len(scores), 1)
        return scores, total
    except Exception as e:
        logger.error(f"Ошибка анализа: {e}")
        return None, str(e)

# ===== ОБРАБОТЧИКИ БОТА =====

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user = update.effective_user
    create_user(user.id, user.username, user.first_name)
    remaining = get_remaining(user.id)
    
    keyboard = [
        [InlineKeyboardButton("🧬 Оценка внешности", callback_data='rate')],
        [InlineKeyboardButton("🚀 Улучшить внешность", callback_data='improve')],
        [InlineKeyboardButton("💰 Купить запросы", callback_data='buy')]
    ]
    
    text = f"""👋 Привет, {user.first_name}!

FACERO — анализ внешности и персональный lookmaxing.

Получи оценку лица, найди сильные стороны и узнай, что можно улучшить.

Осталось сегодня: {remaining}/5 запросов
"""
    await update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

async def button_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    query = update.callback_query
    await query.answer()
    user_id = query.from_user.id
    data = query.data

    if data == 'rate' or data == 'improve':
        remaining = get_remaining(user_id)
        if remaining <= 0:
            await query.edit_message_text(
                "❌ Лимит запросов на сегодня исчерпан!\n\nКупи дополнительные запросы 👇",
                reply_markup=InlineKeyboardMarkup([
                    [InlineKeyboardButton("💰 Купить запросы", callback_data='buy')]
                ])
            )
            return
        context.user_data['mode'] = data
        await query.edit_message_text(
            f"📸 Отправь своё селфи\n\nДля точного анализа:\n• лицо должно быть полностью видно\n• смотри прямо в камеру\n• хорошее освещение\n\nОсталось: {remaining}/5 запросов"
        )

    elif data == 'buy':
        keyboard = [
            [InlineKeyboardButton("📦 30 запросов — 19⭐", callback_data='buy_30')],
            [InlineKeyboardButton("📦 80 запросов — 39⭐", callback_data='buy_80')],
            [InlineKeyboardButton("📦 200 запросов — 79⭐", callback_data='buy_200')],
            [InlineKeyboardButton("🏠 Главное меню", callback_data='menu')]
        ]
        await query.edit_message_text(
            "💰 Дополнительные запросы\n\nЗапросы не сгорают и действуют всегда.\nВыбери пакет:",
            reply_markup=InlineKeyboardMarkup(keyboard)
        )

    elif data.startswith('buy_'):
        count = int(data.split('_')[1])
        add_extra_requests(user_id, count)
        await query.edit_message_text(
            f"✅ Добавлено {count} запросов!\n\nТеперь у тебя {get_remaining(user_id)} доступных анализов.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("🧬 Оценка внешности", callback_data='rate')],
                [InlineKeyboardButton("🏠 Главное меню", callback_data='menu')]
            ])
        )

    elif data == 'menu':
        await start(update, context)

async def photo_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.message.from_user.id
    mode = context.user_data.get('mode', 'rate')
    
    remaining = get_remaining(user_id)
    if remaining <= 0:
        await update.message.reply_text(
            "❌ Лимит запросов исчерпан!\nКупи дополнительные запросы.",
            reply_markup=InlineKeyboardMarkup([
                [InlineKeyboardButton("💰 Купить запросы", callback_data='buy')]
            ])
        )
        return

    msg = await update.message.reply_text("🔍 Анализирую лицо...\nПодожди несколько секунд ⏳")
    
    try:
        photo_file = await update.message.photo[-1].get_file()
        image_bytes = await photo_file.download_as_bytearray()
        
        result = analyze_face(image_bytes)
        
        if result[0] is None:
            await msg.edit_text(f"⚠️ {result[1]}\n\nПопробуй отправить другое фото.")
            return
        
        scores, total = result
        use_analysis(user_id)
        new_remaining = get_remaining(user_id)
        
        emojis = {
            "Глаза": "👁", "Симметрия": "⚖️", "Нос": "👃",
            "Губы": "👄", "Челюсть": "🗿", "Скулы": "💎",
            "Кожа": "🧴", "Причёска": "💇", "Брови": "🪶"
        }
        
        text = "✨ АНАЛИЗ ЗАВЕРШЁН\n\n"
        for key, value in scores.items():
            text += f"{emojis.get(key, '•')} {key} — {value}/10\n"
        
        text += "\n" + "━" * 25 + "\n"
        text += f"⭐ ОБЩАЯ ОЦЕНКА: {total}/10\n"
        text += "━" * 25 + "\n\n"
        text += f"Осталось запросов: {new_remaining}"
        
        keyboard = [
            [InlineKeyboardButton("🔄 Повторить анализ", callback_data=mode)],
            [InlineKeyboardButton("🏠 Главное меню", callback_data='menu')]
        ]
        
        await msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
        
    except Exception as e:
        logger.error(f"Ошибка обработки фото: {e}")
        await msg.edit_text("❌ Произошла ошибка при анализе. Попробуй позже.")

# ===== FLASK + WEBHOOK =====
app = Flask(__name__)
bot_app = None

async def setup_webhook():
    global bot_app
    bot_app = Application.builder().token(TOKEN).build()
    bot_app.add_handler(CommandHandler("start", start))
    bot_app.add_handler(CallbackQueryHandler(button_handler))
    bot_app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    
    WEBHOOK_URL = f"https://{os.environ.get('APP_HOST', 'localhost')}:{PORT}/webhook"
    await bot_app.bot.set_webhook(WEBHOOK_URL)
    return bot_app

@app.route('/webhook', methods=['POST'])
async def webhook():
    if not bot_app:
        return "Not ready", 500
    update = Update.de_json(request.get_json(force=True), bot_app.bot)
    await bot_app.process_update(update)
    return "ok", 200

@app.route('/')
def index():
    return "✅ FACERO Бот работает!", 200

# ===== ЗАПУСК =====
if __name__ == '__main__':
    init_db()
    import asyncio
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(setup_webhook())
    app.run(host='0.0.0.0', port=PORT)