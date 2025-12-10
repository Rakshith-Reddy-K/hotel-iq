import uuid # 1. We import this to generate unique IDs
from sql.db_pool import get_connection

def seed_data():
    print("Seeding new sample booking data...")
    
    # YOUR SPECIFIC HOTEL ID
    HOTEL_ID = 111418 
    
    # Generate a unique reference every time the script runs
    # This creates a string like "REF-A1B2C3"
    unique_ref = f"REF-{str(uuid.uuid4())[:6].upper()}"
    
    # Let's try Room 102 to avoid conflicts with the previous Room 101 attempt
    target_room = '102'

    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Check if this specific room is already taken
            cursor.execute(
                "SELECT id FROM bookings WHERE hotel_id = %s AND room_number = %s", 
                (HOTEL_ID, target_room)
            )
            if cursor.fetchone():
                print(f"Sample data already exists for Hotel {HOTEL_ID} in Room {target_room}.")
                return

            # Create a confirmed booking for Jane Smith in Room 102
            cursor.execute("""
                INSERT INTO bookings 
                (hotel_id, booking_reference, room_number, guest_first_name, guest_last_name, check_in_date, check_out_date, status)
                VALUES 
                (%s, %s, %s, 'Jane', 'Smith', CURRENT_DATE, CURRENT_DATE + 5, 'confirmed');
            """, (HOTEL_ID, unique_ref, target_room))
            
            print(f"✅ Created Booking: Hotel {HOTEL_ID} / Room {target_room} / Ref: {unique_ref}")

if __name__ == "__main__":
    seed_data()