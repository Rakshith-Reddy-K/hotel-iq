import os
from psycopg2 import pool
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()

# Module-level variable to store the pool
_pool = None


def initialize_pool():
    """Initialize connection pool"""
    global _pool
    
    if _pool is not None:
        return _pool
    
    try:
        _pool = pool.ThreadedConnectionPool(
            minconn=1,
            maxconn=20,
            host=os.getenv('CLOUD_DB_HOST'),
            port=os.getenv('CLOUD_DB_PORT'),
            database=os.getenv('CLOUD_DB_NAME'),
            user=os.getenv('CLOUD_DB_USER'),
            password=os.getenv('CLOUD_DB_PASSWORD')
        )
        print("Connection pool created successfully")
        return _pool
    except Exception as e:
        print(f"Error creating connection pool: {e}")
        raise


def get_pool():
    """Get the connection pool, initializing if necessary"""
    global _pool
    if _pool is None:
        initialize_pool()
    return _pool


@contextmanager
def get_connection():
    """Get connection from pool with context manager"""
    pool_instance = get_pool()
    conn = pool_instance.getconn()
    try:
        yield conn
        conn.commit()
    except Exception as e:
        conn.rollback()
        raise e
    finally:
        pool_instance.putconn(conn)


def close_all():
    """Close all connections in pool"""
    global _pool
    if _pool:
        _pool.closeall()
        print("All connections closed")
        _pool = None


# Initialize pool on module import
initialize_pool()