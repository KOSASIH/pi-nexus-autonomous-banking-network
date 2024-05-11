# Original code
def handle_message(message):
    if message.text == "/start":
        #...

# Improved code
from telegram.ext import CommandHandler, MessageHandler

def handle_start(update, context):
    """Handle the /start command"""
    #...

def handle_message(update, context):
    """Handle incoming messages"""
    if update.message.text == "/start":
        handle_start(update, context)
    else:
        #...

telegram_handler = CommandHandler("start", handle_start)
telegram_handler = MessageHandler(Filters.text, handle_message)
