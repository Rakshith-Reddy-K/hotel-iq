import json
from typing import Dict, List, Optional
from sql.db_pool import get_connection

def list_tables():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Query to get all tables
            cursor.execute("""
                SELECT table_name 
                FROM information_schema.tables 
                WHERE table_schema = 'public'
                ORDER BY table_name;
            """)
            tables = cursor.fetchall()
            
            print("Database connection successful!")
            
            if tables:
                print(f"\nTables in database ({len(tables)}):")
                for table in tables:
                    print(f"   - {table[0]}")
            else:
                print("\nNo tables found in database")
            
            return tables

def create_hotels_table():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hotels (
                    hotel_id SERIAL PRIMARY KEY,
                    official_name VARCHAR(255) NOT NULL,
                    star_rating INTEGER CHECK (star_rating >= 1 AND star_rating <= 5),
                    description TEXT,
                    address TEXT,
                    city VARCHAR(100),
                    state VARCHAR(100),
                    zip_code VARCHAR(20),
                    country VARCHAR(100),
                    phone VARCHAR(50),
                    email VARCHAR(255),
                    website TEXT,
                    overall_rating DECIMAL(3, 2),
                    total_reviews INTEGER,
                    cleanliness_rating DECIMAL(3, 2),
                    service_rating DECIMAL(3, 2),
                    location_rating DECIMAL(3, 2),
                    value_rating DECIMAL(3, 2),
                    year_opened DATE,
                    last_renovation DATE,
                    total_rooms INTEGER,
                    number_of_floors INTEGER,
                    images JSONB,
                    additional_info TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                );
            """)
            
            conn.commit()
            print("Hotels table created successfully")

def create_rooms_table():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS rooms (
                    room_id SERIAL PRIMARY KEY,
                    hotel_id INTEGER REFERENCES hotels(hotel_id) ON DELETE CASCADE,
                    
                    room_type VARCHAR(100) NOT NULL,
                    bed_configuration VARCHAR(100),
                    room_size_sqft INTEGER,
                    max_occupancy INTEGER,
                    price_range_min DECIMAL(10, 2),
                    price_range_max DECIMAL(10, 2),
                    
                    description TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    CONSTRAINT rooms_hotel_id_room_type_unique UNIQUE (hotel_id, room_type)
                );
            """)
            
            conn.commit()
            print("Rooms table created successfully")

def create_reviews_table():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS reviews (
                    review_id SERIAL PRIMARY KEY,
                    hotel_id INTEGER REFERENCES hotels(hotel_id) ON DELETE CASCADE,
                    
                    -- Review content
                    overall_rating DECIMAL(3, 2),
                    review_text TEXT,
                    
                    -- Reviewer info (if available)
                    reviewer_name VARCHAR(100),
                    
                    -- Metadata
                    review_date DATE,
                    source VARCHAR(50) DEFAULT 'tripadvisor',
                    
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    
                    CONSTRAINT reviews_hotel_reviewer_date_unique 
                    UNIQUE (hotel_id, reviewer_name, review_date)       
                );
            """)
            
            conn.commit()
            print(" Reviews table created successfully")

def create_amenities_table():
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS amenities (
                    amenity_id SERIAL PRIMARY KEY,
                    hotel_id INTEGER REFERENCES hotels(hotel_id) ON DELETE CASCADE,
                    category VARCHAR(50), -- 'connectivity', 'dining', 'recreation', 'services'
                    description TEXT,
                    details TEXT,
                    CONSTRAINT amenities_hotel_id_name_unique UNIQUE (hotel_id, category) 
                   );
                    
            """)
            
            conn.commit()
            print(" Amenities table created successfully")

def create_policies_table():
    """Create the policies table"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS policies (
                    policy_id SERIAL PRIMARY KEY,
                    hotel_id INTEGER REFERENCES hotels(hotel_id) ON DELETE CASCADE,
                    check_in_time TIME,
                    check_out_time TIME,
                    min_age_requirement INTEGER,
                    pet_policy TEXT,
                    smoking_policy TEXT,
                    children_policy TEXT,
                    extra_person_policy TEXT,
                    cancellation_policy TEXT,
                    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    CONSTRAINT policies_hotel_id_unique UNIQUE (hotel_id)
                );
            """)

            conn.commit()
            print(" Policies table created successfully")

def create_all_tables():
    try:
        print("Starting table creation...\n")
        
        # Create tables in dependency order
        create_hotels_table()
        create_rooms_table()
        create_reviews_table()
        create_amenities_table()
        create_policies_table()
        
        print("\nAll tables created successfully!")
        
        # List all created tables
        print("\n" + "="*50)
        list_tables()
        
    except Exception as e:
        print(f"Error creating tables: {e}")
        raise

def insert_into_hotel_table(transformed_hotel: Dict) -> Optional[int]:
    """
    Insert transformed hotel data into the database
    
    Args:
        transformed_hotel: Dictionary with pre-transformed hotel data
    
    Returns:
        hotel_id of inserted record or None if failed
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # Prepare JSONB fields
                special_features = transformed_hotel.get('special_features')
                awards = transformed_hotel.get('awards')
                images = transformed_hotel.get('images')
                
                cursor.execute("""
                    INSERT INTO hotels (
                        hotel_id,
                        official_name, star_rating, description, address,
                        phone, email, website,
                        overall_rating, total_reviews,
                        cleanliness_rating, service_rating,
                        location_rating, value_rating,
                        year_opened, last_renovation,
                        total_rooms, number_of_floors,
                        special_features, awards, images,
                        additional_info
                    ) VALUES (
                        %s,%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
                    ) RETURNING hotel_id;
                """, (
                     transformed_hotel.get('hotel_id'),
                    transformed_hotel.get('official_name'),
                    transformed_hotel.get('star_rating'),
                    transformed_hotel.get('description'),
                    transformed_hotel.get('address'),
                    transformed_hotel.get('phone'),
                    transformed_hotel.get('email'),
                    transformed_hotel.get('website'),
                    transformed_hotel.get('overall_rating'),
                    transformed_hotel.get('total_reviews'),
                    transformed_hotel.get('cleanliness_rating'),
                    transformed_hotel.get('service_rating'),
                    transformed_hotel.get('location_rating'),
                    transformed_hotel.get('value_rating'),
                    transformed_hotel.get('year_opened'),
                    transformed_hotel.get('last_renovation'),
                    transformed_hotel.get('total_rooms'),
                    transformed_hotel.get('number_of_floors'),
                    json.dumps(special_features) if special_features else None,
                    json.dumps(awards) if awards else None,
                    json.dumps(images) if images else None,
                    transformed_hotel.get('additional_info')
                ))
                
                hotel_id = cursor.fetchone()[0]
                conn.commit()
                
                print(f" Inserted hotel: {transformed_hotel.get('official_name')} (ID: {hotel_id})")
                return hotel_id
                
            except Exception as e:
                conn.rollback()
                print(f" Error inserting hotel: {e}")
                raise

def insert_into_rooms_table(hotel_id: int, transformed_rooms: List[Dict]) -> int:
    """
    Insert transformed room data
    
    Returns:
        Number of rooms inserted
    """
    inserted_count = 0
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                for room in transformed_rooms:
                    room['hotel_id'] = hotel_id
                    
                    # Prepare amenities as JSONB if present
                    amenities = room.get('amenities')
                    
                    cursor.execute("""
                        INSERT INTO rooms (
                            hotel_id, room_type, bed_configuration,
                            room_size_sqft, max_occupancy,
                            amenities,
                            price_range_min, price_range_max,
                            description
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s, %s, %s
                        );
                    """, (
                        hotel_id,
                        room.get('room_type'),
                        room.get('bed_configuration'),
                        room.get('room_size_sqft'),
                        room.get('max_occupancy'),
                        json.dumps(amenities) if amenities else None,
                        room.get('price_range_min'),
                        room.get('price_range_max'),
                        room.get('description')
                    ))
                    inserted_count += 1
                
                conn.commit()
                print(f" Inserted {inserted_count} rooms for hotel ID: {hotel_id}")
                
            except Exception as e:
                conn.rollback()
                print(f" Error inserting rooms: {e}")
                raise
    
    return inserted_count

def insert_into_amenities_table(hotel_id: int, transformed_amenities: List[Dict]) -> int:
    """
    Insert transformed amenity data
    
    Returns:
        Number of amenities inserted
    """
    inserted_count = 0
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                for amenity in transformed_amenities:
                    # Add hotel_id to amenity data
                    amenity['hotel_id'] = hotel_id
                    
                    cursor.execute("""
                        INSERT INTO amenities (
                            hotel_id, category, description, details
                        ) VALUES (
                            %s, %s, %s, %s
                        );
                    """, (
                        hotel_id,
                        amenity.get('category'),
                        amenity.get('description'),
                        json.dumps(amenity.get('details')) if amenity.get('details') else None
                    ))
                    inserted_count += 1
                
                conn.commit()
                print(f" Inserted {inserted_count} amenities for hotel ID: {hotel_id}")
                
            except Exception as e:
                conn.rollback()
                print(f" Error inserting amenities: {e}")
                raise
    
    return inserted_count

def insert_into_policies_table(hotel_id: int, transformed_policies: Dict) -> bool:
    """
    Insert transformed policy data
    
    Returns:
        True if successful
    """
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                # Add hotel_id to policies data
                transformed_policies['hotel_id'] = hotel_id
                
                cursor.execute("""
                    INSERT INTO policies (
                        hotel_id, check_in_time, check_out_time,
                        min_age_requirement, pet_policy,
                        smoking_policy, children_policy,
                        extra_person_policy, cancellation_policy
                    ) VALUES (
                        %s, %s, %s, %s, %s, %s, %s, %s, %s
                    );
                """, (
                    hotel_id,
                    transformed_policies.get('check_in_time'),
                    transformed_policies.get('check_out_time'),
                    transformed_policies.get('min_age_requirement'),
                    transformed_policies.get('pet_policy'),
                    transformed_policies.get('smoking_policy'),
                    transformed_policies.get('children_policy'),
                    transformed_policies.get('extra_person_policy'),
                    transformed_policies.get('cancellation_policy')
                ))
                
                conn.commit()
                print(f" Inserted policies for hotel ID: {hotel_id}")
                return True
                
            except Exception as e:
                conn.rollback()
                print(f" Error inserting policies: {e}")
                raise

def insert_into_reviews_table(hotel_id: int, transformed_reviews: List[Dict]) -> int:
    """
    Insert transformed review data
    
    Returns:
        Number of reviews inserted
    """
    if not transformed_reviews:
        return 0
    
    inserted_count = 0
    
    with get_connection() as conn:
        with conn.cursor() as cursor:
            try:
                for review in transformed_reviews:
                    # Add hotel_id to review data
                    review['hotel_id'] = hotel_id
                    
                    cursor.execute("""
                        INSERT INTO reviews (
                            hotel_id, overall_rating, title,
                            review_text, reviewer_name,
                            review_date, source
                        ) VALUES (
                            %s, %s, %s, %s, %s, %s, %s
                        );
                    """, (
                        hotel_id,
                        review.get('overall_rating'),
                        review.get('title'),
                        review.get('review_text'),
                        review.get('reviewer_name'),
                        review.get('review_date'),
                        review.get('source', 'tripadvisor')
                    ))
                    inserted_count += 1
                
                conn.commit()
                if inserted_count > 0:
                    print(f" Inserted {inserted_count} reviews for hotel ID: {hotel_id}")
                
            except Exception as e:
                conn.rollback()
                print(f" Error inserting reviews: {e}")
                raise
    
    return inserted_count

# def drop_all_tables():
    """Drop all tables (use with caution!)"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Drop in reverse dependency order
            tables = ['policies', 'amenities', 'reviews', 'rooms', 'hotels']
            
            for table in tables:
                cursor.execute(f"DROP TABLE IF EXISTS {table} CASCADE;")
                print(f"Dropped table: {table}")
            
            conn.commit()
            print(" All tables dropped successfully")

# def get_table_info(table_name):
    """Get detailed information about a specific table"""
    with get_connection() as conn:
        with conn.cursor() as cursor:
            # Get column information
            cursor.execute("""
                SELECT 
                    column_name,
                    data_type,
                    character_maximum_length,
                    is_nullable,
                    column_default
                FROM information_schema.columns
                WHERE table_name = %s
                ORDER BY ordinal_position;
            """, (table_name,))
            
            columns = cursor.fetchall()
            
            if columns:
                print(f"\n📊 Structure of '{table_name}' table:")
                print("-" * 80)
                for col in columns:
                    col_name, dtype, max_len, nullable, default = col
                    type_info = dtype
                    if max_len:
                        type_info += f"({max_len})"
                    nullable_info = "NULL" if nullable == 'YES' else "NOT NULL"
                    default_info = f"DEFAULT {default}" if default else ""
                    
                    print(f"  {col_name:25} {type_info:20} {nullable_info:10} {default_info}")
            else:
                print(f"Table '{table_name}' not found")
            
            # Get indexes
            cursor.execute("""
                SELECT indexname, indexdef
                FROM pg_indexes
                WHERE tablename = %s;
            """, (table_name,))
            
            indexes = cursor.fetchall()
            if indexes:
                print(f"\n🔍 Indexes on '{table_name}':")
                for idx_name, idx_def in indexes:
                    print(f"  - {idx_name}")

# Example usage
if __name__ == "__main__":
    # Create all tables
    create_all_tables()