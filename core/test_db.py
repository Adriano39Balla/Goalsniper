import sys
import os
sys.path.insert(0, '/app')

from core.database import DatabaseManager

db = DatabaseManager()
print("Methods available in DatabaseManager:")
for method in dir(db):
    if not method.startswith('_'):
        print(f"  - {method}")

print("\nChecking for get_cursor method...")
if hasattr(db, 'get_cursor'):
    print("✓ get_cursor method exists!")
else:
    print("✗ get_cursor method NOT found!")
