from sql.db_pool import get_connection

def seed_data():
    print(" Seeding sample booking data...")
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Check if user exists to avoid duplicates
            cursor.execute("SELECT id FROM bookings WHERE room_number = '101'")
            if cursor.fetchone():
                print(" Sample data already exists.")
                return

            # Create a confirmed booking for John Doe in Room 101
            cursor.execute("""
                INSERT INTO bookings 
                (booking_reference, room_number, guest_first_name, guest_last_name, check_in_date, check_out_date, status)
                VALUES 
                ('REF12345', '101', 'John', 'Doe', CURRENT_DATE, CURRENT_DATE + 5, 'confirmed');
            """)
            print(" Created Booking: Room 101 / Guest: Doe")

if __name__ == "__main__":
    seed_data()