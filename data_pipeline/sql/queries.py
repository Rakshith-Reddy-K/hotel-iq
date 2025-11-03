from typing import Dict, List
from sql.db_pool import get_connection
import pandas as pd
import os
from psycopg2.extras import execute_values

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

def insert_one_hotel_complete(hotel_data: Dict, rooms: List[Dict], amenities: List[Dict], 
                               policies: List[Dict], reviews: List[Dict]):
    try:
        with get_connection() as conn:
            with conn.cursor() as cursor:
                # Insert hotel
                cursor.execute("""
                    INSERT INTO hotels (
                        hotel_id, official_name, star_rating, description, address, city, state, 
                        zip_code, country, phone, email, website, overall_rating, 
                        total_reviews, cleanliness_rating, service_rating, location_rating, 
                        value_rating, year_opened, last_renovation, total_rooms, 
                        number_of_floors, images, additional_info
                    )
                    VALUES (
                        %(hotel_id)s, %(official_name)s, %(star_rating)s, %(description)s, 
                        %(address)s, %(city)s, %(state)s, %(zip_code)s, %(country)s, 
                        %(phone)s, %(email)s, %(website)s, %(overall_rating)s, 
                        %(total_reviews)s, %(cleanliness_rating)s, %(service_rating)s, 
                        %(location_rating)s, %(value_rating)s, %(year_opened)s, 
                        %(last_renovation)s, %(total_rooms)s, %(number_of_floors)s, 
                        %(images)s, %(additional_info)s
                    )
                    ON CONFLICT (hotel_id) DO NOTHING
                """, hotel_data)
                
                # Batch insert rooms
                if rooms:
                    execute_values(cursor, """
                        INSERT INTO rooms (
                            hotel_id, room_type, bed_configuration, room_size_sqft, 
                            max_occupancy, price_range_min, price_range_max, description
                        ) VALUES %s
                        ON CONFLICT (hotel_id, room_type) DO NOTHING
                    """, [
                        (r['hotel_id'], r['room_type'], r['bed_configuration'], 
                         r['room_size_sqft'], r['max_occupancy'], r['price_range_min'], 
                         r['price_range_max'], r['description'])
                        for r in rooms
                    ])
                
                # Batch insert amenities
                if amenities:
                    execute_values(cursor, """
                        INSERT INTO amenities (hotel_id, category, description, details)
                        VALUES %s
                        ON CONFLICT (hotel_id, category) DO NOTHING
                    """, [
                        (a['hotel_id'], a['category'], a['description'], a['details'])
                        for a in amenities
                    ])
                
                # Batch insert policies
                if policies:
                    execute_values(cursor, """
                        INSERT INTO policies (
                            hotel_id, check_in_time, check_out_time, min_age_requirement,
                            pet_policy, smoking_policy, children_policy, 
                            extra_person_policy, cancellation_policy
                        ) VALUES %s
                        ON CONFLICT (hotel_id) DO NOTHING
                    """, [
                        (p['hotel_id'], p['check_in_time'], p['check_out_time'], 
                         p['min_age_requirement'], p['pet_policy'], p['smoking_policy'], 
                         p['children_policy'], p['extra_person_policy'], p['cancellation_policy'])
                        for p in policies
                    ])
                
                # Batch insert reviews
                if reviews:
                    execute_values(cursor, """
                        INSERT INTO reviews (
                            hotel_id, overall_rating, review_text, reviewer_name, 
                            review_date, source
                        ) VALUES %s
                        ON CONFLICT (hotel_id, reviewer_name, review_date) DO NOTHING
                    """, [
                        (r['hotel_id'], r['overall_rating'], r['review_text'], 
                         r['reviewer_name'], r['review_date'], r['source'])
                        for r in reviews
                    ])
            
            print(f"Inserted hotel {hotel_data['hotel_id']} - {hotel_data['official_name']}")
        
    except Exception as e:
        print(f"Failed hotel {hotel_data.get('hotel_id')}: {str(e)}")
        raise

def clean_dict_for_db(d: Dict) -> Dict:
    """Convert NaN values to None in dictionary"""
    cleaned = {}
    for key, value in d.items():
        if pd.isna(value):
            cleaned[key] = None
        elif isinstance(value, float) and (value != value):
            cleaned[key] = None
        else:
            cleaned[key] = value
    return cleaned


def bulk_insert_from_csvs(csv_dir: str = 'data/processed/boston'):
    hotels_df = pd.read_csv(os.path.join(csv_dir, 'batch_hotels.csv'))
    rooms_df = pd.read_csv(os.path.join(csv_dir, 'batch_rooms.csv'))
    amenities_df = pd.read_csv(os.path.join(csv_dir, 'batch_amenities.csv'))
    policies_df = pd.read_csv(os.path.join(csv_dir, 'batch_policies.csv'))
    reviews_df = pd.read_csv(os.path.join(csv_dir, 'batch_reviews.csv'))
    
    rooms_by_hotel = rooms_df.groupby('hotel_id')
    amenities_by_hotel = amenities_df.groupby('hotel_id')
    policies_by_hotel = policies_df.groupby('hotel_id')
    reviews_by_hotel = reviews_df.groupby('hotel_id')
    
    count = 0
    errors = 0
    
    for _, hotel_row in hotels_df.iterrows():
        hotel_id = hotel_row['hotel_id']
        
        try:
            hotel_dict = clean_dict_for_db(hotel_row.to_dict())
            hotel_rooms = []
            if hotel_id in rooms_by_hotel.groups:
                hotel_rooms = [clean_dict_for_db(r) for r in rooms_by_hotel.get_group(hotel_id).to_dict('records')]
            hotel_amenities = []
            if hotel_id in amenities_by_hotel.groups:
                hotel_amenities = [clean_dict_for_db(a) for a in amenities_by_hotel.get_group(hotel_id).to_dict('records')]
            hotel_policies = []
            if hotel_id in policies_by_hotel.groups:
                hotel_policies = [clean_dict_for_db(p) for p in policies_by_hotel.get_group(hotel_id).to_dict('records')]
            hotel_reviews = []
            if hotel_id in reviews_by_hotel.groups:
                hotel_reviews = [clean_dict_for_db(r) for r in reviews_by_hotel.get_group(hotel_id).to_dict('records')]
            insert_one_hotel_complete(
                hotel_dict, 
                hotel_rooms, 
                hotel_amenities, 
                hotel_policies, 
                hotel_reviews
            )
            
            count += 1
            if count % 5 == 0:
                print(f"Inserted {count}/{len(hotels_df)} hotels")
                
        except Exception as e:
            errors += 1
            print(f"Failed to insert hotel {hotel_id}: {str(e)}")
            continue
    
    print(f"\nDone! Successfully inserted {count} hotels, {errors} errors")
    return {'success': count, 'errors': errors}