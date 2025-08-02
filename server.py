from data_ingestion import fetch_live_data
from ai_training import train_model, predict
from db import init_db, save_prediction
from telegram_notifier import send_telegram_message
from config import MODEL_PATH

def main():
    print("🚀 Starting SuperBrain Bot...")

    # 1. Ensure DB exists
    init_db()

    # 2. Fetch live data
    data = fetch_live_data()
    if data is None or data.empty:
        print("⚠️ No live data fetched.")
        return

    # 3. Train model (optional: daily/weekly retrain)
    model = train_model(data)
    model.save(MODEL_PATH)

    # 4. Predict
    predictions = predict(model, data)

    # 5. Save to DB
    save_prediction(predictions)

    # 6. Send to Telegram
    send_telegram_message(f"New AI Predictions:\n{predictions}")

if __name__ == "__main__":
    main()
