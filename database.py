#!/usr/bin/env python3
"""
Database connection manager for Supabase (PostgreSQL)
"""
import os
import psycopg2
from psycopg2 import pool
from psycopg2.extras import RealDictCursor
import pandas as pd
from loguru import logger
from dotenv import load_dotenv
import urllib.parse
import time
from typing import Optional, Dict, Any

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.max_retries = 3
        self.retry_delay = 2
        self.setup_connection_pool()
    
    def get_connection_string(self) -> str:
        """Build connection string from environment variables"""
        try:
            # Check for Railway/Heroku style DATABASE_URL
            database_url = os.getenv('DATABASE_URL')
            if database_url:
                logger.info("Using DATABASE_URL for connection")
                # Parse the URL and ensure proper format
                parsed_url = urllib.parse.urlparse(database_url)
                if parsed_url.scheme == 'postgres':
                    return database_url
                elif parsed_url.scheme == 'postgresql':
                    return database_url
                # Convert other formats
                return f"postgresql://{parsed_url.netloc}{parsed_url.path}"
            
            # Supabase specific environment variables
            host = os.getenv('SUPABASE_HOST')
            database = os.getenv('SUPABASE_DB', 'postgres')
            user = os.getenv('SUPABASE_USER', 'postgres')
            password = os.getenv('SUPABASE_PASSWORD')
            port = os.getenv('SUPABASE_PORT', '5432')
            
            if not all([host, user, password]):
                logger.error("Missing required database environment variables")
                raise ValueError("Missing required database environment variables")
            
            # Construct connection string for Supabase
            conn_string = f"host={host} dbname={database} user={user} password={password} port={port} sslmode=require"
            logger.debug(f"Built connection string for host: {host}")
            return conn_string
            
        except Exception as e:
            logger.error(f"Error building connection string: {e}")
            raise
    
    def setup_connection_pool(self):
        """Setup connection pool for Supabase with retry logic"""
        for attempt in range(self.max_retries):
            try:
                conn_string = self.get_connection_string()
                
                logger.info(f"Attempting database connection (attempt {attempt + 1}/{self.max_retries})...")
                
                # Create connection pool
                self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                    minconn=1,
                    maxconn=20,
                    dsn=conn_string,
                    connect_timeout=10
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
                    logger.info(f"Retrying in {self.retry_delay} seconds...")
                    time.sleep(self.retry_delay)
                    self.retry_delay *= 2  # Exponential backoff
                else:
                    logger.error(f"Failed to establish database connection after {self.max_retries} attempts")
                    
                    # Fallback: Try without SSL if SSL fails
                    try:
                        logger.info("Attempting connection without SSL...")
                        conn_string_without_ssl = self.get_connection_string().replace("sslmode=require", "sslmode=disable")
                        self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                            minconn=1,
                            maxconn=5,
                            dsn=conn_string_without_ssl,
                            connect_timeout=10
                        )
                        logger.warning("Connected without SSL (less secure)")
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
    
    def get_connection(self):
        """Get connection from pool"""
        try:
            if not self.connection_pool:
                self.setup_connection_pool()
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    def release_connection(self, connection):
        """Release connection back to pool"""
        try:
            if connection and self.connection_pool:
                self.connection_pool.putconn(connection)
        except Exception as e:
            logger.error(f"Failed to release database connection: {e}")
    
    def execute_query(self, query, params=None, fetch=True, commit=True):
        """Execute SQL query with optional parameters"""
        connection = None
        cursor = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            # Log query for debugging (without sensitive data)
            if params:
                logger.debug(f"Executing query with params: {query[:200]}...")
            else:
                logger.debug(f"Executing query: {query[:200]}...")
            
            cursor.execute(query, params)
            
            if fetch and cursor.description:
                result = cursor.fetchall()
                if commit:
                    connection.commit()
                
                if result:
                    df = pd.DataFrame(result)
                    logger.debug(f"Query returned {len(df)} rows")
                    return df
                else:
                    logger.debug("Query returned no rows")
                    return pd.DataFrame()
            else:
                if commit:
                    connection.commit()
                rows_affected = cursor.rowcount
                logger.debug(f"Query affected {rows_affected} rows")
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
            logger.error(f"Query: {query}")
            if params:
                logger.error(f"Params: {params}")
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
                self.release_connection(connection)
    
    def batch_insert(self, table_name, data, columns, batch_size=1000):
        """Insert data in batches with error handling"""
        if not data:
            logger.warning("No data to insert")
            return 0
        
        total_inserted = 0
        connection = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Create placeholders for columns
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join([f'"{col}"' for col in columns])
            
            # Create query
            query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
            """
            
            # Process in batches
            for i in range(0, len(data), batch_size):
                batch = data[i:i + batch_size]
                try:
                    cursor.executemany(query, batch)
                    inserted = cursor.rowcount
                    total_inserted += inserted
                    logger.debug(f"Batch {i//batch_size + 1}: Inserted {inserted} rows")
                except Exception as batch_error:
                    logger.error(f"Error in batch {i//batch_size + 1}: {batch_error}")
                    connection.rollback()
                    # Try inserting row by row
                    for row_idx, row_data in enumerate(batch):
                        try:
                            cursor.execute(query, row_data)
                            total_inserted += 1
                        except Exception as row_error:
                            logger.warning(f"Failed to insert row {row_idx}: {row_error}")
                            connection.rollback()
            
            connection.commit()
            logger.info(f"Total inserted {total_inserted} rows into {table_name}")
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Batch insert error: {e}")
            raise
        finally:
            if connection:
                cursor.close()
                self.release_connection(connection)
        
        return total_inserted
    
    def create_tables(self, tables_sql):
        """Create database tables if they don't exist"""
        connection = None
        cursor = None
        
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            for i, table_sql in enumerate(tables_sql):
                try:
                    cursor.execute(table_sql)
                    logger.debug(f"Created table {i + 1}/{len(tables_sql)}")
                except Exception as e:
                    logger.warning(f"Table creation {i + 1} may have failed: {e}")
                    connection.rollback()
            
            connection.commit()
            logger.success(f"Created/verified {len(tables_sql)} tables")
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Error creating tables: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
            if connection:
                self.release_connection(connection)
    
    def health_check(self) -> Dict[str, Any]:
        """Check database health"""
        try:
            start_time = time.time()
            result = self.execute_query("SELECT 1 as health, NOW() as timestamp", fetch=True)
            query_time = time.time() - start_time
            
            return {
                'status': 'healthy',
                'response_time': round(query_time, 3),
                'timestamp': result.iloc[0]['timestamp'] if not result.empty else None,
                'connection_pool': {
                    'minconn': 1,
                    'maxconn': 20,
                    'current_connections': self.connection_pool._used if self.connection_pool else 0
                }
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
