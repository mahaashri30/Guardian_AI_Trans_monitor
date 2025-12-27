# db.py – InsForge PostgreSQL connection helper
import os
import psycopg2
from psycopg2.extras import RealDictCursor

DB_URL = os.getenv(
    "DB_URL",
    "postgresql://postgres:69e99917f4e5deb1d3cf4aa5517ad66e@7qqa3ax5.ap-southeast.database.insforge.app:5432/insforge?sslmode=require"  # ← replace with your InsForge URL
)

def get_conn():
    return psycopg2.connect(DB_URL, cursor_factory=RealDictCursor)
