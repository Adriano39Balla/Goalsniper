import argparse, json, os, logging
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2 import OperationalError
import socket  # <<< ADD for IPv4 resolution

from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (
    brier_score_loss, accuracy_score, log_loss, precision_score,
    roc_auc_score, precision_recall_curve
)

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(levelname)s - %(message)s")
logger = logging.getLogger("trainer")

# ───────────────────────── Feature sets (match main.py) ───────────────────────── #

FEATURES: List[str] = [
    "minute",
    "goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff",
    "sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a",
    "cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff",
    "red_h","red_a","red_sum",
    "yellow_h","yellow_a",
]

# Prematch snapshots (as saved by main.save_prematch_snapshot via extract_prematch_features)
PRE_FEATURES: List[str] = [
    "pm_ov25_h","pm_ov35_h","pm_btts_h",
    "pm_ov25_a","pm_ov35_a","pm_btts_a",
    "pm_ov25_h2h","pm_ov35_h2h","pm_btts_h2h",
    # live placeholders for shape-compat
    "minute","goals_h","goals_a","goals_sum","goals_diff",
    "xg_h","xg_a","xg_sum","xg_diff","sot_h","sot_a","sot_sum",
    "sh_total_h","sh_total_a","cor_h","cor_a","cor_sum",
    "pos_h","pos_a","pos_diff","red_h","red_a","red_sum",
    "yellow_h","yellow_a",
]

EPS = 1e-6

# ─────────────────────── Env knobs ─────────────────────── #

RECENCY_HALF_LIFE_DAYS = float(os.getenv("RECENCY_HALF_LIFE_DAYS", "120"))  # 0 disables
RECENCY_MONTHS         = int(os.getenv("RECENCY_MONTHS", "36"))             # training window cap
MARKET_CUTOFFS_RAW     = os.getenv("MARKET_CUTOFFS", "BTTS=75,1X2=80,OU=88")
TIP_MAX_MINUTE_ENV     = os.getenv("TIP_MAX_MINUTE", "")
OU_TRAIN_LINES_RAW     = os.getenv("OU_TRAIN_LINES", os.getenv("OU_LINES", "2.5,3.5"))

def _parse_market_cutoffs(s: str) -> Dict[str,int]:
    out: Dict[str,int] = {}
    for tok in (s or "").split(","):
        tok = tok.strip()
        if not tok or "=" not in tok: continue
        k,v = tok.split("=",1)
        try: out[k.strip().upper()] = int(float(v.strip()))
        except: pass
    return out

MARKET_CUTOFFS = _parse_market_cutoffs(MARKET_CUTOFFS_RAW)
try:
    TIP_MAX_MINUTE: Optional[int] = int(float(TIP_MAX_MINUTE_ENV)) if TIP_MAX_MINUTE_ENV.strip() else None
except Exception:
    TIP_MAX_MINUTE = None

def _parse_ou_lines(raw: str) -> List[float]:
    vals: List[float] = []
    for t in (raw or "").split(","):
        t = t.strip()
        if not t: continue
        try: vals.append(float(t))
        except: pass
    return vals or [2.5, 3.5]

# ─────────────────────── DB utils (IPv4 + SSL + pooled-first + backoff) ─────────────────────── #

def _resolve_ipv4(host: str, port: int) -> Optional[str]:
    try:
        infos = socket.getaddrinfo(host, port, socket.AF_INET, socket.SOCK_STREAM)
        for _family, _socktype, _proto, _canonname, sockaddr in infos:
            ip, _ = sockaddr
            return ip
    except Exception as e:
        logger.warning("[DNS] IPv4 resolve failed for %s:%s: %s — using hostname fallback", host, port, e)
    return None

def _parse_pg_url(url: str) -> Dict[str, Any]:
    # very small parser; url like postgresql://user:pass@host:port/db?sslmode=require
    from urllib.parse import urlparse, parse_qsl
    pr = urlparse(url)
    if pr.scheme not in ("postgresql","postgres"):
        raise SystemExit("DATABASE_URL must start with postgresql:// or postgres://")
    user = pr.username or ""
    password = pr.password or ""
    host = pr.hostname or ""
    port = pr.port or 5432
    dbname = (pr.path or "").lstrip("/") or "postgres"
    params = dict(parse_qsl(pr.query))
    params.setdefault("sslmode","require")
    return {"user":user,"password":password,"host":host,"port":int(port),"dbname":dbname,"params":params}

def _q(v: str) -> str:
    s = "" if v is None else str(v)
    if s == "" or all(ch not in s for ch in (" ", "'", "\\", "\t", "\n")):
        return s
    s = s.replace("\\","\\\\").replace("'","\\'")
    return f"'{s}'"

def _make_conninfo(parts: Dict[str, Any], port: int, hostaddr: Optional[str]) -> str:
    base = [
        f"host={_q(parts['host'])}",
        f"port={port}",
        f"dbname={_q(parts['dbname'])}",
    ]
    if parts["user"]: base.append(f"user={_q(parts['user'])}")
    if parts["password"]: base.append(f"password={_q(parts['password'])}")
    if hostaddr: base.append(f"hostaddr={_q(hostaddr)}")
    base.append("sslmode=require")
    return " ".join(base)

def _conninfo_candidates(url: str) -> List[str]:
    parts = _parse_pg_url(url)
    env_hostaddr = os.getenv("DB_HOSTADDR")
    prefer_pooled = os.getenv("DB_PREFER_POOLED","1") not in ("0","false","False","no","NO")
    ports = []
    if prefer_pooled: ports.append(6543)
    if parts["port"] not in ports: ports.append(parts["port"])
    cands: List[str] = []
    for p in ports:
        ipv4 = env_hostaddr or _resolve_ipv4(parts["host"], p)
        if ipv4: cands.append(_make_conninfo(parts, p, ipv4))
        cands.append(_make_conninfo(parts, p, None))
    return cands

def _connect(db_url: Optional[str]):
    url = db_url or os.getenv("DATABASE_URL")
    if not url: raise SystemExit("DATABASE_URL must be set.")
    cands = _conninfo_candidates(url)
    delay = 1.0
    last = None
    for attempt in range(6):
        for dsn in cands:
            try:
                conn = psycopg2.connect(dsn)
                conn.autocommit = True
                logger.info("[DB] train_models connected with DSN: %s", dsn.replace("password=", "password=**** "))
                return conn
            except OperationalError as e:
                last = e
                continue
        if attempt == 5:
            raise OperationalError(f"Could not connect after retries. Last error: {last}")
        time.sleep(delay)
        delay *= 2

def _read_sql(conn, sql: str, params: Tuple = ()) -> pd.DataFrame:
    return pd.read_sql_query(sql, conn, params=params)

def _exec(conn, sql: str, params: Tuple = ()) -> None:
    with conn.cursor() as cur: cur.execute(sql, params)

def _set_setting(conn, key: str, value: str) -> None:
    _exec(conn,
          "INSERT INTO settings(key,value) VALUES(%s,%s) "
          "ON CONFLICT(key) DO UPDATE SET value=EXCLUDED.value",
          (key, value))

def _ensure_training_tables(conn) -> None:
    _exec(conn, "CREATE TABLE IF NOT EXISTS settings (key TEXT PRIMARY KEY, value TEXT)")
    _exec(conn, """
        CREATE TABLE IF NOT EXISTS prematch_snapshots (
            match_id   BIGINT PRIMARY KEY,
            created_ts BIGINT,
            payload    TEXT
        )
    """)
    _exec(conn, "CREATE INDEX IF NOT EXISTS idx_pre_snap_ts ON prematch_snapshots (created_ts DESC)")

def _get_setting_json(conn, key: str) -> Optional[dict]:
    try:
        df = _read_sql(conn, "SELECT value FROM settings WHERE key=%s", (key,))
        if df.empty: return None
        return json.loads(df.iloc[0]["value"])
    except Exception:
        return None

# ─────────────────────── Data loaders ─────────────────────── #
# (… unchanged … all your existing training code remains …)
# The rest of the file is identical to your current train_models.py
# except for the DB section above which now prefers pooled (6543),
# enforces SSL, prefers IPv4, and retries with backoff.

# ─────────────────────── Modeling helpers ─────────────────────── #
# (keep all your existing code from here onward)
# … (UNCHANGED CONTENT BELOW THIS POINT — omitted for brevity in this snippet)
