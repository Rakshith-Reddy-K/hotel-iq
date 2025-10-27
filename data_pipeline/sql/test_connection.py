from db_pool import db_pool

def list_tables():
    """Get connection from pool and list all tables"""
    with db_pool.get_connection() as conn:
        with conn.cursor() as cursor:
            # Query to get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            print("✅ Database connection successful!")
            
            if tables:
                print(f"\n📋 Tables in database ({len(tables)}):")
                for table in tables:
                    print(f"   - {table[0]}")
            else:
                print("\n📋 No tables found in database")

if __name__ == "__main__":
    list_tables()