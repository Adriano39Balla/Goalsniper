#!/usr/bin/env python3
"""
Database connection manager for Supabase (PostgreSQL)
"""
import os
import psycopg2
from psycopg2 import pool, extras
from psycopg2.extras import RealDictCursor
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import urllib.parse
import time
import ssl
from typing import Optional, Dict, Any, List, Tuple
from contextlib import contextmanager
from datetime import datetime

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.max_retries = 5
        self.retry_delay = 2
        self.setup_connection_pool()
        self.query_timeout = 30  # seconds
    
    def get_connection_string(self) -> str:
        """Build connection string from environment variables"""
        try:
            # Check for Railway/Heroku style DATABASE_URL
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                logger.info("Using DATABASE_URL for connection")
                parsed_url = urllib.parse.urlparse(database_url)
                
                # Ensure proper PostgreSQL URL format
                if parsed_url.scheme == 'postgres':
                    return database_url
                elif parsed_url.scheme == 'postgresql':
                    return database_url
                # Handle other URL formats
                return f"postgresql://{parsed_url.netloc}{parsed_url.path}"
            
            # Supabase specific environment variables
            host = os.getenv('SUPABASE_HOST')
            database = os.getenv('SUPABASE_DB', 'postgres')
            user = os.getenv('SUPABASE_USER', 'postgres')
            password = os.getenv('SUPABASE_PASSWORD')
            port = os.getenv('SUPABASE_PORT', '5432')
            
            if not all([host, user, password]):
                logger.error("Missing required database environment variables")
                logger.error(f"Host: {'Present' if host else 'Missing'}, "
                           f"User: {'Present' if user else 'Missing'}, "
                           f"Password: {'Present' if password else 'Missing'}")
                raise ValueError("Missing required database environment variables")
            
            # Construct connection string for Supabase
            conn_string = f"host={host} dbname={database} user={user} password={password} port={port} sslmode=require"
            logger.debug(f"Built connection string for host: {host}")
            return conn_string
            
        except Exception as e:
            logger.error(f"Error building connection string: {e}")
            raise
    
    def setup_connection_pool(self):
        """Setup connection pool for Supabase with retry logic and SSL"""
        for attempt in range(self.max_retries):
            try:
                conn_string = self.get_connection_string()
                
                logger.info(f"Attempting database connection (attempt {attempt + 1}/{self.max_retries})...")
                
                # Create SSL context
                ssl_context = ssl.create_default_context()
                ssl_context.check_hostname = True
                ssl_context.verify_mode = ssl.CERT_REQUIRED
                
                # Create connection pool with SSL
                self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=20,
                    dsn=conn_string,
                    connect_timeout=10,
                    sslmode='require',
                    sslrootcert='./ssl/ca.pem' if os.path.exists('./ssl/ca.pem') else None
                )
                
                # Test connection
                test_conn = self.connection_pool.getconn()
                cursor = test_conn.cursor()
                cursor.execute("SELECT version();")
                version = cursor.fetchone()
                cursor.close()
                self.connection_pool.putconn(test_conn)
                
                logger.success(f"Database connection pool established. PostgreSQL version: {version[0]}")
                return
                
            except psycopg2.OperationalError as e:
                logger.warning(f"Connection attempt {attempt + 1} failed: {e}")
                if attempt < self.max_retries - 1:
                    wait_time = self.retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"Failed to establish database connection after {self.max_retries} attempts")
                    
                    # Fallback: Try without SSL if SSL fails
                    try:
                        logger.info("Attempting connection without SSL as fallback...")
                        conn_string_no_ssl = self.get_connection_string().replace("sslmode=require", "sslmode=disable")
                        
                        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                            minconn=1,
                            maxconn=10,
                            dsn=conn_string_no_ssl,
                            connect_timeout=10
                        )
                        
                        logger.warning("Connected without SSL (less secure - for development only)")
                        return
                    except Exception as ssl_error:
                        logger.error(f"SSL fallback also failed: {ssl_error}")
                        raise
                        
            except Exception as e:
                logger.error(f"Unexpected error setting up connection pool: {e}")
                if attempt < self.max_retries - 1:
                    time.sleep(self.retry_delay)
                else:
                    raise
    
    @contextmanager
    def get_connection(self):
        """Context manager for database connections"""
        connection = None
        try:
            if not self.connection_pool:
                self.setup_connection_pool()
            connection = self.connection_pool.getconn()
            yield connection
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
        finally:
            if connection:
                self.connection_pool.putconn(connection)
    
    def execute_query(self, query, params=None, fetch=True, commit=True):
        """Execute SQL query with optional parameters"""
        connection = None
        cursor = None
        start_time = time.time()
        
        try:
            connection = self.connection_pool.getconn()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Set statement timeout
            cursor.execute(f"SET statement_timeout = {self.query_timeout * 1000};")
            
            # Execute query
            if params:
                cursor.execute(query, params)
            else:
                cursor.execute(query)
            
            if fetch and cursor.description:
                result = cursor.fetchall()
                if commit:
                    connection.commit()
                
                execution_time = time.time() - start_time
                
                if result:
                    df = pd.DataFrame(result)
                    logger.debug(f"Query returned {len(df)} rows in {execution_time:.3f}s")
                    return df
                else:
                    logger.debug(f"Query returned no rows in {execution_time:.3f}s")
                    return pd.DataFrame()
            else:
                rows_affected = cursor.rowcount
                if commit:
                    connection.commit()
                
                execution_time = time.time() - start_time
                logger.debug(f"Query affected {rows_affected} rows in {execution_time:.3f}s")
                
                return rows_affected
                
        except psycopg2.IntegrityError as e:
            if connection:
                connection.rollback()
            logger.warning(f"Database integrity error: {e}")
            raise
        except psycopg2.ProgrammingError as e:
            if connection:
                connection.rollback()
            logger.error(f"SQL programming error: {e}")
            logger.error(f"Query: {query[:500]}")
            if params:
                logger.error(f"Params: {params}")
            raise
        except psycopg2.OperationalError as e:
            if connection:
                connection.rollback()
            logger.error(f"Database operational error: {e}")
            raise
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database query error: {e}")
            logger.error(f"Query: {query[:500]}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.connection_pool.putconn(connection)
    
    def batch_insert(self, table_name, data, columns, batch_size=1000, on_conflict='DO NOTHING'):
        """Insert data in batches with error handling"""
        if not data:
            logger.warning("No data to insert")
            return 0
        
        total_inserted = 0
        
        try:
            # Create placeholders for columns
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join([f'"{col}"' for col in columns])
            
            # Create query with conflict handling
            query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT {on_conflict}
            """
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                
                with self.get_connection() as conn:
                    cursor = conn.cursor()
                    try:
                        cursor.executemany(query, batch)
                        inserted = cursor.rowcount
                        total_inserted += inserted
                        conn.commit()
                        logger.debug(f"Batch {i//batch_size + 1}: Inserted {inserted} rows")
                    except Exception as batch_error:
                        conn.rollback()
                        logger.error(f"Error in batch {i//batch_size + 1}: {batch_error}")
                        
                        # Try inserting row by row
                        for row_idx, row_data in enumerate(batch):
                            try:
                                cursor.execute(query, row_data)
                                total_inserted += 1
                                conn.commit()
                            except Exception as row_error:
                                conn.rollback()
                                logger.warning(f"Failed to insert row {row_idx}: {row_error}")
            
            logger.info(f"Total inserted {total_inserted} rows into {table_name}")
            
        except Exception as e:
            logger.error(f"Batch insert error: {e}")
            raise
        
        return total_inserted
    
    def create_tables(self, tables_sql):
        """Create database tables if they don't exist"""
        try:
            logger.info(f"Creating/verifying {len(tables_sql)} tables...")
            
            for i, table_sql in enumerate(tables_sql):
                try:
                    self.execute_query(table_sql, fetch=False)
                    logger.debug(f"Created/verified table {i + 1}/{len(tables_sql)}")
                except Exception as e:
                    logger.warning(f"Table creation {i + 1} may have failed: {e}")
                    # Continue with other tables
            
            logger.success(f"Created/verified {len(tables_sql)} tables")
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            raise
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start_time = time.time()
            result = self.execute_query(
                "SELECT 1 as health, NOW() as timestamp, pg_database_size(current_database()) as db_size", 
                fetch=True
            )
            query_time = time.time() - start_time
            
            if result.empty:
                return {
                    'status': 'unhealthy',
                    'error': 'No response from database',
                    'response_time': round(query_time, 3)
                }
            
            db_size_bytes = result.iloc[0]['db_size']
            db_size_mb = db_size_bytes / (1024 * 1024)
            
            # Check connection pool status
            pool_status = {
                'minconn': 1,
                'maxconn': 20,
                'current_connections': self.connection_pool._used if self.connection_pool else 0,
                'available_connections': self.connection_pool._available if self.connection_pool else 0
            }
            
            return {
                'status': 'healthy',
                'response_time': round(query_time, 3),
                'timestamp': result.iloc[0]['timestamp'].isoformat() if hasattr(result.iloc[0]['timestamp'], 'isoformat') else str(result.iloc[0]['timestamp']),
                'database_size_mb': round(db_size_mb, 2),
                'connection_pool': pool_status
            }
        except Exception as e:
            return {
                'status': 'unhealthy',
                'error': str(e),
                'response_time': None,
                'timestamp': None
            }
    
    def close_all_connections(self):
        """Close all connections in pool"""
        try:
            if self.connection_pool:
                self.connection_pool.closeall()
                logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
    
    def backup_database(self, backup_path: str = None):
        """Create database backup"""
        try:
            if not backup_path:
                backup_path = f"backups/db_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.sql"
            
            os.makedirs(os.path.dirname(backup_path), exist_ok=True)
            
            # Get connection string
            conn_string = self.get_connection_string()
            
            # Use pg_dump for backup
            import subprocess
            cmd = ['pg_dump', '-F', 'c', '-f', backup_path]
            
            # Parse connection string
            if 'postgresql://' in conn_string:
                cmd.extend(['-d', conn_string])
            else:
                # Parse host= port= dbname= user= password= format
                import re
                params = dict(re.findall(r'(\w+)=([^ ]+)', conn_string))
                cmd.extend([
                    '-h', params.get('host', 'localhost'),
                    '-p', params.get('port', '5432'),
                    '-U', params.get('user', 'postgres'),
                    '-d', params.get('dbname', 'postgres')
                ])
                
                # Set password in environment
                env = os.environ.copy()
                env['PGPASSWORD'] = params.get('password', '')
            
            result = subprocess.run(cmd, capture_output=True, text=True, env=env if 'env' in locals() else None)
            
            if result.returncode == 0:
                logger.success(f"Database backup created: {backup_path}")
                return backup_path
            else:
                logger.error(f"Backup failed: {result.stderr}")
                return None
                
        except Exception as e:
            logger.error(f"Error creating database backup: {e}")
            return None
    
    def execute_with_retry(self, query, params=None, max_retries=3, fetch=True):
        """Execute query with retry logic"""
        for attempt in range(max_retries):
            try:
                return self.execute_query(query, params, fetch=fetch)
            except psycopg2.OperationalError as e:
                if attempt < max_retries - 1:
                    logger.warning(f"Query failed (attempt {attempt + 1}), retrying...: {e}")
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    logger.error(f"Query failed after {max_retries} attempts: {e}")
                    raise
            except Exception as e:
                logger.error(f"Query failed: {e}")
                raise
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close_all_connections()


# Utility function for database operations
def execute_with_retry(db_manager, query, params=None, max_retries=3):
    """Execute query with retry logic"""
    for attempt in range(max_retries):
        try:
            return db_manager.execute_query(query, params)
        except psycopg2.OperationalError as e:
            if attempt < max_retries - 1:
                logger.warning(f"Query failed (attempt {attempt + 1}), retrying...: {e}")
                time.sleep(2 ** attempt)  # Exponential backoff
            else:
                logger.error(f"Query failed after {max_retries} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise
