# ============================================
# FACERO — FULLY WORKING
# RENDER.COM
# ============================================

import os
import logging
import sqlite3
from datetime import datetime
from flask import Flask, request
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import Application, CommandHandler, CallbackQueryHandler, MessageHandler, filters
import numpy as np
import cv2
import mediapipe as mp

TOKEN = os.environ.get("BOT_TOKEN")
PORT = 8443

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
bot_app = None

def init_db():
    conn = sqlite3.connect('facero.db')
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        telegram_id INTEGER UNIQUE,
        username TEXT,
        first_name TEXT,
        requests_today INTEGER DEFAULT 0,
        last_date TEXT,
        extra_requests INTEGER DEFAULT 0
    )''')
    conn.commit()
    conn.close()
    logger.info("DB ready")

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

def start(update: Update, context):
    user = update.effective_user
    create_user(user.id, user.username, user.first_name)
    remaining = get_remaining(user.id)

    keyboard = [
        [InlineKeyboardButton("🧬 Оценка", callback_data='rate')],
        [InlineKeyboardButton("🚀 Улучшить", callback_data='improve')],
        [InlineKeyboardButton("💰 Купить", callback_data='buy')]
    ]

    text = f"👋 Привет, {user.first_name}!\nFACERO — анализ внешности.\nОсталось: {remaining}/5"
    update.message.reply_text(text, reply_markup=InlineKeyboardMarkup(keyboard))

def button_handler(update: Update, context):
    query = update.callback_query
    query.answer()
    user_id = query.from_user.id
    data = query.data

    if data in ['rate', 'improve']:
        remaining = get_remaining(user_id)
        if remaining <= 0:
            query.edit_message_text("❌ Лимит исчерпан. Купи запросы.")
            return
        context.user_data['mode'] = data
        query.edit_message_text(f"📸 Отправь селфи\nОсталось: {remaining}/5")

    elif data == 'buy':
        keyboard = [
            [InlineKeyboardButton("30 запросов — 19⭐", callback_data='buy_30')],
            [InlineKeyboardButton("80 запросов — 39⭐", callback_data='buy_80')],
            [InlineKeyboardButton("200 запросов — 79⭐", callback_data='buy_200')],
            [InlineKeyboardButton("🏠 Меню", callback_data='menu')]
        ]
        query.edit_message_text("💰 Выбери пакет:", reply_markup=InlineKeyboardMarkup(keyboard))

    elif data.startswith('buy_'):
        count = int(data.split('_')[1])
        add_extra_requests(user_id, count)
        query.edit_message_text(f"✅ Добавлено {count} запросов!")

    elif data == 'menu':
        start(update, context)

def photo_handler(update: Update, context):
    user_id = update.message.from_user.id
    mode = context.user_data.get('mode', 'rate')
    remaining = get_remaining(user_id)
    if remaining <= 0:
        update.message.reply_text("❌ Лимит исчерпан!")
        return

    msg = update.message.reply_text("🔍 Анализирую лицо...")
    try:
        photo = update.message.photo[-1].get_file()
        image_bytes = photo.download_as_bytearray()
        result = analyze_face(image_bytes)
        if result[0] is None:
            msg.edit_text(f"⚠️ {result[1]}")
            return
        scores, total = result
        use_analysis(user_id)
        new_remaining = get_remaining(user_id)

        emojis = {"Глаза": "👁", "Симметрия": "⚖️", "Нос": "👃", "Губы": "👄", "Челюсть": "🗿", "Скулы": "💎", "Кожа": "🧴", "Причёска": "💇", "Брови": "🪶"}
        text = "✨ АНАЛИЗ ЗАВЕРШЁН\n\n"
        for k, v in scores.items():
            text += f"{emojis.get(k, '•')} {k} — {v}/10\n"
        text += f"\n⭐ ОБЩАЯ ОЦЕНКА: {total}/10\n"
        text += f"\nОсталось: {new_remaining} запросов"

        keyboard = [
            [InlineKeyboardButton("🔄 Повторить", callback_data=mode)],
            [InlineKeyboardButton("🏠 Меню", callback_data='menu')]
        ]
        msg.edit_text(text, reply_markup=InlineKeyboardMarkup(keyboard))
    except Exception as e:
        msg.edit_text(f"❌ Ошибка: {e}")

def setup_bot():
    global bot_app
    bot_app = Application.builder().token(TOKEN).build()
    bot_app.add_handler(CommandHandler("start", start))
    bot_app.add_handler(CallbackQueryHandler(button_handler))
    bot_app.add_handler(MessageHandler(filters.PHOTO, photo_handler))
    logger.info("✅ Бот настроен")

@app.route('/webhook', methods=['POST'])
def webhook():
    if not bot_app:
        return "Bot not ready", 500
    update = Update.de_json(request.get_json(force=True), bot_app.bot)
    bot_app.process_update(update)
    return "ok", 200

@app.route('/')
def index():
    return "✅ FACERO Бот работает!", 200

@app.route('/set_webhook')
def set_webhook():
    if not bot_app:
        return "Bot not initialized", 500
    url = f"https://{os.environ.get('RENDER_EXTERNAL_HOSTNAME', 'localhost')}/webhook"
    bot_app.bot.set_webhook(url)
    return f"✅ Webhook set to {url}"

if __name__ == '__main__':
    init_db()
    setup_bot()
    app.run(host='0.0.0.0', port=PORT)
