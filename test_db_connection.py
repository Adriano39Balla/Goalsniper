import os
import psycopg2
from psycopg2 import OperationalError

# Get the database URL from the environment
DATABASE_URL = os.getenv("DATABASE_URL")

if not DATABASE_URL:
    print("❌ DATABASE_URL not set in environment variables.")
    exit(1)

def test_connection():
    try:
        conn = psycopg2.connect(DATABASE_URL)
        print("✅ Successfully connected to the database!")
        
        # Create a cursor and run a simple query
        cur = conn.cursor()
        cur.execute("SELECT version();")
        db_version = cur.fetchone()
        print(f"Database version: {db_version[0]}")
        
        cur.close()
        conn.close()
        print("✅ Connection closed successfully.")
    except OperationalError as e:
        print("❌ Failed to connect to the database.")
        print(f"Error: {e}")

if __name__ == "__main__":
    test_connection()
