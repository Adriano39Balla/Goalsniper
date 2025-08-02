from sqlalchemy import create_engine, text
from config import DATABASE_URL

engine = create_engine(DATABASE_URL)

def init_db():
    with engine.connect() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS predictions (
                id SERIAL PRIMARY KEY,
                prediction TEXT NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
    print("✅ Database initialized.")

def save_prediction(predictions):
    with engine.connect() as conn:
        for pred in predictions:
            conn.execute(text("INSERT INTO predictions (prediction) VALUES (:pred)"), {"pred": str(pred)})
    print("💾 Predictions saved to database.")
