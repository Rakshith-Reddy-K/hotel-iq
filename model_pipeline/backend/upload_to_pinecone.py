"""
Upload Hotels and Reviews to Pinecone
======================================

This script uploads hotel and review data from CSV files to Pinecone vector database.
Uses data from the data/processed/{city}/ folder.

Usage:
    python upload_to_pinecone.py [--hotels] [--reviews] [--all]
    CITY=boston python upload_to_pinecone.py --all
"""

import os
import sys
import pandas as pd
from pathlib import Path
from pinecone import Pinecone, ServerlessSpec
from langchain_huggingface import HuggingFaceEmbeddings
from tqdm import tqdm
from dotenv import load_dotenv
import argparse

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import path as path_util

# Load environment variables
load_dotenv()

# Initialize embeddings (same as in config.py)
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def get_pinecone_client():
    """Initialize and return Pinecone client."""
    api_key = os.getenv("PINECONE_API_KEY")
    if not api_key:
        raise ValueError("PINECONE_API_KEY not found in .env file")
    return Pinecone(api_key=api_key)


def create_indexes_if_needed(pc):
    """Create Pinecone indexes if they don't exist."""
    existing_indexes = [index.name for index in pc.list_indexes()]
    
    # Hotels index
    hotel_index_name = os.getenv("HOTEL_INDEX_NAME", "hoteliq-hotels")
    if hotel_index_name not in existing_indexes:
        print(f"Creating index: {hotel_index_name}")
        pc.create_index(
            name=hotel_index_name,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Created {hotel_index_name}")
    else:
        print(f"✓ Index {hotel_index_name} already exists")
    
    # Reviews index
    review_index_name = os.getenv("REVIEW_INDEX_NAME", "hoteliq-reviews")
    if review_index_name not in existing_indexes:
        print(f"Creating index: {review_index_name}")
        pc.create_index(
            name=review_index_name,
            dimension=384,  # all-MiniLM-L6-v2 dimension
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
        print(f"✅ Created {review_index_name}")
    else:
        print(f"✓ Index {review_index_name} already exists")


def upload_hotels(csv_path):
    """
    Upload hotels from CSV to Pinecone.
    Uses the correct column names from the CSV file.
    """
    print("\n" + "="*60)
    print("📤 Uploading Hotels to Pinecone")
    print("="*60)
    
    # Initialize Pinecone
    pc = get_pinecone_client()
    index_name = os.getenv("HOTEL_INDEX_NAME", "hoteliq-hotels")
    index = pc.Index(index_name)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} hotels from {csv_path}")
    print(f"📋 CSV columns: {list(df.columns)}")
    
    # Upload in batches
    batch_size = 100
    vectors = []
    uploaded_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing hotels"):
        try:
            # Create description text for embedding
            # Use the actual column names from CSV
            hotel_name = str(row.get("official_name", "Unknown Hotel"))
            description = str(row.get("description", ""))
            
            if not description or description == "nan":
                description = f"{hotel_name} located at {row.get('address', '')}"
            
            # Generate embedding
            embedding = embeddings.embed_query(description)
            
            # Prepare metadata with ALL relevant fields from CSV
            metadata = {
                "hotel_id": str(row["hotel_id"]),
                "hotel_name": hotel_name,  # Store as hotel_name in Pinecone
                "official_name": hotel_name,  # Also store as official_name
                "star_rating": str(row.get("star_rating", "N/A")),
                "city": str(row.get("city", "")),
                "state": str(row.get("state", "")),
                "zip_code": str(row.get("zip_code", "")),
                "address": str(row.get("address", "")),
                "phone": str(row.get("phone", "")),
                "website": str(row.get("website", "")),
                "overall_rating": str(row.get("overall_rating", "N/A")),
                "total_reviews": str(row.get("total_reviews", "")),
                "description": description[:1000]  # Truncate to 1000 chars for metadata
            }
            
            # Remove nan values
            metadata = {k: v for k, v in metadata.items() if v != "nan"}
            
            # Add to batch
            vectors.append({
                "id": f"hotel_{row['hotel_id']}",
                "values": embedding,
                "metadata": metadata
            })
            
            # Upload batch
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors)
                uploaded_count += len(vectors)
                vectors = []
                
        except Exception as e:
            print(f"\n⚠️ Error processing hotel {row.get('hotel_id', 'unknown')}: {e}")
            continue
    
    # Upload remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
        uploaded_count += len(vectors)
    
    print(f"\n✅ Uploaded {uploaded_count} hotels to Pinecone index: {index_name}")
    return uploaded_count


def upload_reviews(csv_path):
    """
    Upload reviews from CSV to Pinecone.
    Uses the correct column names from the CSV file.
    """
    print("\n" + "="*60)
    print("📤 Uploading Reviews to Pinecone")
    print("="*60)
    
    # Initialize Pinecone
    pc = get_pinecone_client()
    index_name = os.getenv("REVIEW_INDEX_NAME", "hoteliq-reviews")
    index = pc.Index(index_name)
    
    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"📊 Loaded {len(df)} reviews from {csv_path}")
    print(f"📋 CSV columns: {list(df.columns)}")
    
    # Upload in batches
    batch_size = 100
    vectors = []
    uploaded_count = 0
    skipped_count = 0
    
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Processing reviews"):
        try:
            # Get review text - check multiple possible column names
            review_text = str(row.get("text", row.get("review_text", row.get("content", ""))))
            
            if not review_text or review_text == "nan" or len(review_text) < 10:
                skipped_count += 1
                continue
            
            # Generate embedding
            embedding = embeddings.embed_query(review_text)
            
            # Prepare metadata
            metadata = {
                "hotel_id": str(row.get("hotel_id", "")),
                "overall_rating": float(row.get("overall_rating", 0)) if pd.notna(row.get("overall_rating")) else 0.0,
                "review_date": str(row.get("review_date", row.get("date", ""))),
                "review_text": review_text[:1000]  # Truncate to 1000 chars
            }
            
            # Remove nan values
            metadata = {k: v for k, v in metadata.items() if str(v) != "nan"}
            
            # Add to batch
            vectors.append({
                "id": f"review_{idx}",
                "values": embedding,
                "metadata": metadata
            })
            
            # Upload batch
            if len(vectors) >= batch_size:
                index.upsert(vectors=vectors)
                uploaded_count += len(vectors)
                vectors = []
                
        except Exception as e:
            print(f"\n⚠️ Error processing review {idx}: {e}")
            skipped_count += 1
            continue
    
    # Upload remaining vectors
    if vectors:
        index.upsert(vectors=vectors)
        uploaded_count += len(vectors)
    
    print(f"\n✅ Uploaded {uploaded_count} reviews to Pinecone index: {index_name}")
    if skipped_count > 0:
        print(f"⚠️ Skipped {skipped_count} reviews (empty or invalid)")
    return uploaded_count


def main():
    """Main function to handle command-line arguments."""
    parser = argparse.ArgumentParser(description="Upload hotels and reviews to Pinecone")
    parser.add_argument("--hotels", action="store_true", help="Upload hotels only")
    parser.add_argument("--reviews", action="store_true", help="Upload reviews only")
    parser.add_argument("--all", action="store_true", help="Upload both hotels and reviews")
    parser.add_argument("--create-indexes", action="store_true", help="Create indexes if they don't exist")
    
    args = parser.parse_args()
    
    # If no arguments, default to --all
    if not (args.hotels or args.reviews or args.all):
        args.all = True
    
    try:
        # Get city from environment or default to boston
        city = os.getenv('CITY', 'boston')
        print(f"📍 Using city: {city}")
        
        # Use path utility to get correct file paths
        hotels_path = path_util.get_processed_hotels_path(city)
        reviews_path = path_util.get_processed_reviews_path(city)
        
        print(f"📂 Hotels CSV: {hotels_path}")
        print(f"📂 Reviews CSV: {reviews_path}")
        
        # Create indexes if requested
        if args.create_indexes:
            pc = get_pinecone_client()
            create_indexes_if_needed(pc)
        
        # Upload data
        if args.hotels or args.all:
            upload_hotels(hotels_path)
        
        if args.reviews or args.all:
            upload_reviews(reviews_path)
        
        print("\n" + "="*60)
        print("✅ Upload complete!")
        print("="*60)
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

