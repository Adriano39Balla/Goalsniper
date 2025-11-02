import os
import time
import logging
import psycopg2
from typing import Optional, Dict, Any
from psycopg2.pool import SimpleConnectionPool
from urllib.parse import urlparse, parse_qsl
import socket

from .config import config

log = logging.getLogger("goalsniper.database")

class DatabaseManager:
    def __init__(self):
        self.pool: Optional[SimpleConnectionPool] = None
        self._initialized = False

    def _parse_pg_url(self, url: str) -> dict:
        pr = urlparse(url)
        if pr.scheme not in ("postgresql", "postgres"):
            raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
        params = dict(parse_qsl(pr.query))
        params.setdefault("sslmode", "require")
        return {
            "user": pr.username or "",
            "password": pr.password or "",
            "host": pr.hostname or "",
            "port": int(pr.port or 5432),
            "dbname": (pr.path or "").lstrip("/") or "postgres",
            "params": params,
        }

    def _resolve_ipv4(self, host: str, port: int) -> Optional[str]:
        try:
            infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
            for _family, _socktype, _proto, _canon, sockaddr in infos:
                ip, _p = sockaddr
                return ip
        except Exception:
            return None

    def _make_conninfo(self, parts: dict, port: int, hostaddr: Optional[str]) -> str:
        base = [
            f"host={self._q(parts['host'])}",
            f"port={port}",
            f"dbname={self._q(parts['dbname'])}",
        ]
        if parts["user"]:
            base.append(f"user={self._q(parts['user'])}")
        if parts["password"]:
            base.append(f"password={self._q(parts['password'])}")
        if hostaddr:
            base.append(f"hostaddr={self._q(hostaddr)}")
        base.append("sslmode=require")
        return " ".join(base)

    def _q(self, v: str) -> str:
        s = "" if v is None else str(v)
        if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
            return s
        s = s.replace("\\", "\\\\").replace("'", "\\'")
        return f"'{s}'"

    def _conninfo_candidates(self) -> list[str]:
        parts = self._parse_pg_url(config.database.url)
        ports: list[int] = []
        if config.database.prefer_pooled:
            ports.append(6543)
        if parts["port"] not in ports:
            ports.append(parts["port"])
        
        cands: list[str] = []
        for p in ports:
            ip = config.database.hostaddr or self._resolve_ipv4(parts["host"], p)
            if ip:
                cands.append(self._make_conninfo(parts, p, ip))
            cands.append(self._make_conninfo(parts, p, None))
        return cands

    def initialize(self):
        """Initialize database connection pool with retry/backoff"""
        if self._initialized:
            return

        candidates = self._conninfo_candidates()
        delay = 1.0
        last_error = "unknown"
        
        for attempt in range(6):
            for dsn in candidates:
                try:
                    self.pool = SimpleConnectionPool(
                        minconn=1, 
                        maxconn=config.database.pool_max, 
                        dsn=dsn
                    )
                    masked = dsn.replace("password=", "password=**** ")
                    log.info("[DB] Connected (pool=%d) using DSN: %s", config.database.pool_max, masked)
                    self._initialized = True
                    return
                except psycopg2.OperationalError as e:
                    last_error = str(e)
                    continue
            
            if attempt == 5:
                raise psycopg2.OperationalError(
                    f"DB pool init failed after retries. Last error: {last_error}. "
                    "Hint: set DB_HOSTADDR=<IPv4> or enable Supabase IPv4 addon, and prefer 6543."
                )
            time.sleep(delay)
            delay *= 2

    def get_connection(self):
        """Get a database connection from pool"""
        if not self._initialized:
            self.initialize()
        
        try:
            return self.pool.getconn()
        except Exception:
            self._initialized = False
            self.initialize()
            return self.pool.getconn()

    def return_connection(self, conn):
        """Return connection to pool"""
        try:
            self.pool.putconn(conn)
        except Exception as e:
            log.warning("[DB] Error returning connection to pool: %s", e)
            try:
                conn.close()
            except:
                pass

    def get_cursor(self):
        """Get a database cursor context manager"""
        return DatabaseCursor()

    def init_schema(self):
        """Initialize database schema"""
        with self.get_cursor() as c:
            c.execute("""CREATE TABLE IF NOT EXISTS tips (
                match_id BIGINT, 
                league_id BIGINT, 
                league TEXT,
                home TEXT, 
                away TEXT, 
                market TEXT, 
                suggestion TEXT,
                confidence DOUBLE PRECISION, 
                confidence_raw DOUBLE PRECISION,
                score_at_tip TEXT, 
                minute INTEGER, 
                created_ts BIGINT,
                odds DOUBLE PRECISION, 
                book TEXT, 
                ev_pct DOUBLE PRECISION,
                sent_ok INTEGER DEFAULT 1,
                PRIMARY KEY (match_id, created_ts)
            )""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS matches (
                match_id BIGINT PRIMARY KEY,
                league_id BIGINT,
                league TEXT,
                home_team TEXT,
                away_team TEXT,
                match_time BIGINT,
                status TEXT,
                home_score INTEGER,
                away_score INTEGER,
                last_updated BIGINT
            )""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS alerts (
                alert_id SERIAL PRIMARY KEY,
                match_id BIGINT,
                alert_type TEXT,
                message TEXT,
                created_ts BIGINT,
                sent BOOLEAN DEFAULT FALSE
            )""")
            
            c.execute("""CREATE TABLE IF NOT EXISTS system_logs (
                log_id SERIAL PRIMARY KEY,
                level TEXT,
                message TEXT,
                timestamp BIGINT,
                component TEXT
            )""")

db = DatabaseManager()

class DatabaseCursor:
    """Context manager for database operations"""
    
    def __init__(self):
        self.conn = None
        self.cur = None
        
    def __enter__(self):
        self.conn = db.get_connection()
        self.conn.autocommit = True
        self.cur = self.conn.cursor()
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            if self.cur:
                self.cur.close()
        except Exception as e:
            log.warning("[DB] Error closing cursor: %s", e)
        finally:
            if self.conn:
                db.return_connection(self.conn)
    
    def execute(self, sql: str, params: tuple | list = ()):
        try:
            self.cur.execute(sql, params or ())
            return self.cur
        except Exception as e:
            log.error("DB execute failed: %s\nSQL: %s\nParams: %s", e, sql, params)
            raise
    
    def fetchone(self):
        return self.cur.fetchone()
    
    def fetchall(self):
        return self.cur.fetchall()
