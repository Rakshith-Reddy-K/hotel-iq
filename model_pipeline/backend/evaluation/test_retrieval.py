import sys,os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from agents.pinecone_retrieval import retrieve_reviews_by_query, retrieve_hotels_by_description
def main():
    print("Running retrieval sanity checks")
    
    # Hotel Retrieval 
    hotels = retrieve_hotels_by_description("luxury hotel with pool", top_k=3)
    print("Hotels found:", [h.metadata for h in hotels])
    
    print("\n=== Testing Review Retrieval ===")
    reviews = retrieve_reviews_by_query("clean", top_k=3) 
    
    if reviews:
        print("Top review snippets:")
        for r in reviews:
            print("-", r.page_content[:200])
    else:
        print("⚠️ Retrieved 0 reviews. This likely means the 'hoteliq-reviews' index is empty (Stage B was skipped or failed).")


if __name__ == "__main__":
    try:
        main()
        print("\n✅ All retrieval tests completed!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
