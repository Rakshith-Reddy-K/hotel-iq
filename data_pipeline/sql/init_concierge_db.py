from sql.queries import create_bookings_table, create_guest_requests_table

if __name__ == "__main__":
    print("Initializing Concierge Database Tables...")
    create_bookings_table()
    create_guest_requests_table()
    print("Database initialization complete.")