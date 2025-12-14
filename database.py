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

load_dotenv()

class DatabaseManager:
    def __init__(self):
        self.connection_pool = None
        self.setup_connection_pool()
    
    def setup_connection_pool(self):
        """Setup connection pool for Supabase"""
        try:
            self.connection_pool = psycopg2.pool.SimpleConnectionPool(
                1, 20,
                host=os.getenv('SUPABASE_HOST'),
                database=os.getenv('SUPABASE_DB'),
                user=os.getenv('SUPABASE_USER'),
                password=os.getenv('SUPABASE_PASSWORD'),
                port=os.getenv('SUPABASE_PORT', 5432),
                sslmode='require'
            )
            logger.success("Database connection pool established")
        except Exception as e:
            logger.error(f"Failed to create database connection pool: {e}")
            raise
    
    def get_connection(self):
        """Get connection from pool"""
        try:
            return self.connection_pool.getconn()
        except Exception as e:
            logger.error(f"Failed to get database connection: {e}")
            raise
    
    def release_connection(self, connection):
        """Release connection back to pool"""
        try:
            self.connection_pool.putconn(connection)
        except Exception as e:
            logger.error(f"Failed to release database connection: {e}")
    
    def execute_query(self, query, params=None, fetch=True):
        """Execute SQL query with optional parameters"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor(cursor_factory=RealDictCursor)
            
            cursor.execute(query, params)
            
            if fetch and cursor.description:
                result = cursor.fetchall()
                df = pd.DataFrame(result)
                return df
            else:
                connection.commit()
                return None
                
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Database query error: {e}")
            logger.error(f"Query: {query}")
            logger.error(f"Params: {params}")
            raise
            
        finally:
            if connection:
                self.release_connection(connection)
    
    def batch_insert(self, table_name, data, columns):
        """Insert data in batches"""
        connection = None
        try:
            connection = self.get_connection()
            cursor = connection.cursor()
            
            # Create placeholders for columns
            placeholders = ', '.join(['%s'] * len(columns))
            columns_str = ', '.join(columns)
            
            # Create query
            query = f"""
            INSERT INTO {table_name} ({columns_str})
            VALUES ({placeholders})
            ON CONFLICT DO NOTHING
            """
            
            # Execute batch insert
            cursor.executemany(query, data)
            connection.commit()
            
            logger.info(f"Inserted {cursor.rowcount} rows into {table_name}")
            
        except Exception as e:
            if connection:
                connection.rollback()
            logger.error(f"Batch insert error: {e}")
            raise
            
        finally:
            if connection:
                self.release_connection(connection)
    
    def close_all_connections(self):
        """Close all connections in pool"""
        try:
            self.connection_pool.closeall()
            logger.info("All database connections closed")
        except Exception as e:
            logger.error(f"Error closing connections: {e}")
