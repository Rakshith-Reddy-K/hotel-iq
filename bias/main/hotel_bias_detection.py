import os
import json
import pandas as pd
import numpy as np
import requests
from pathlib import Path
from collections import Counter
from typing import Dict, List
from datetime import datetime
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HotelBiasDetection:
    """OpenAI-powered bias detection for hotel chatbot responses"""
    
    def __init__(
        self,
        responses_parquet_path: str,
        pinecone_index_name: str = "hotel-iq-reviews",
        pinecone_environment: str = None,
        output_dir: str = "evaluation/results",
        model: str = "gpt-4o-mini",
        embedding_model: str = "text-embedding-3-small"
    ):
        """
        Initialize bias detection with OpenAI and Pinecone REST API
        
        Args:
            responses_parquet_path: Path to chatbot responses parquet
            pinecone_index_name: Name of Pinecone index containing reviews
            pinecone_environment: Pinecone environment (optional)
            output_dir: Directory to save results
            model: OpenAI model for analysis
            embedding_model: OpenAI embedding model
        """
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.embedding_model = embedding_model
        
        # Initialize Pinecone via REST API
        self.pinecone_api_key = os.getenv("PINECONE_API_KEY")
        if not self.pinecone_api_key:
            raise ValueError("PINECONE_API_KEY not found in environment variables")
        
        self.pinecone_index_name = pinecone_index_name
        self.pinecone_environment = pinecone_environment
        
        # Get index host URL
        self.index_host = self._get_index_host()
        logger.info(f"✓ Connected to Pinecone index via REST API: {pinecone_index_name}")
        logger.info(f"  Index host: {self.index_host}")
        
        # Get index stats
        stats = self._get_index_stats()
        if stats:
            logger.info(f"  Total vectors: {stats.get('total_vector_count', 'N/A')}")
            # Your index uses text-embedding-3-large with 3072 dimensions
            self.vector_dimension = stats.get('dimension', 3072)
            logger.info(f"  Dimensions: {self.vector_dimension}")
        else:
            self.vector_dimension = 3072  # Default for your setup
        
        self.responses_parquet_path = responses_parquet_path
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results = {}
        self.detailed_results = []
        
        # Bias detection thresholds
        self.config = {
            'min_reviews_threshold': 10,
            'rating_disparity_threshold': 4.5,
            'embedding_similarity_threshold': 0.75,
            'severity_threshold': 0.85
        }
        
        # Sparse data indicators for semantic matching
        self.sparse_data_phrases = [
            "limited reviews available",
            "small sample of feedback",
            "insufficient data to conclude",
            "few reviews to analyze",
            "limited information available",
            "not enough reviews",
            "sparse feedback data"
        ]
        
        logger.info(f"✓ Initialized with {model} and {embedding_model}")

    def _get_index_host(self) -> str:
        """Get the index host URL from Pinecone"""
        # First try to get from environment variable
        host = os.getenv("PINECONE_INDEX_HOST")
        if host:
            return host
        
        # Try to fetch from Pinecone control plane API
        try:
            url = "https://api.pinecone.io/indexes"
            headers = {
                "Api-Key": self.pinecone_api_key,
                "Content-Type": "application/json"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()
            
            indexes = response.json()
            for index in indexes.get('indexes', []):
                if index.get('name') == self.pinecone_index_name:
                    return index.get('host')
            
            raise ValueError(f"Index '{self.pinecone_index_name}' not found")
        except Exception as e:
            logger.error(f"Error fetching index host: {e}")
            raise ValueError(
                "Could not determine Pinecone index host. "
                "Please set PINECONE_INDEX_HOST in your .env file. "
                "Find it in Pinecone console under your index details."
            )

    def _get_index_stats(self) -> Dict:
        """Get index statistics via REST API"""
        try:
            url = f"https://{self.index_host}/describe_index_stats"
            headers = {
                "Api-Key": self.pinecone_api_key,
                "Content-Type": "application/json"
            }
            response = requests.post(url, headers=headers, json={})
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.warning(f"Could not fetch index stats: {e}")
            return {}

    def fetch_reviews_from_pinecone(self, hotel_id: str, top_k: int = 10000) -> pd.DataFrame:
        """
        Fetch all reviews for a specific hotel from Pinecone via REST API
        
        Args:
            hotel_id: Hotel ID to fetch reviews for
            top_k: Maximum number of reviews to fetch
            
        Returns:
            DataFrame with reviews
        """
        try:
            # Create a dummy vector for metadata-only query
            dummy_vector = [0.0] * self.vector_dimension
            
            # Convert hotel_id to string for filter (matching your metadata structure)
            hotel_id_str = str(hotel_id)
            
            # Prepare query request
            url = f"https://{self.index_host}/query"
            headers = {
                "Api-Key": self.pinecone_api_key,
                "Content-Type": "application/json"
            }
            
            # Filter by hotel_id (stored as string in your metadata)
            payload = {
                "vector": dummy_vector,
                "filter": {
                    "hotel_id": {"$eq": hotel_id_str},
                    "data_type": {"$eq": "review"}  # Based on your vectorizer
                },
                "topK": top_k,
                "includeMetadata": True
            }
            
            logger.info(f"  Querying Pinecone for hotel_id={hotel_id_str}...")
            response = requests.post(url, headers=headers, json=payload)
            response.raise_for_status()
            
            results = response.json()
            matches = results.get('matches', [])
            
            if not matches:
                logger.warning(f"  No reviews found in Pinecone for hotel {hotel_id}")
                return pd.DataFrame()
            
            # Extract reviews from metadata (matching your vectorizer structure)
            reviews = []
            for match in matches:
                metadata = match.get('metadata', {})
                
                # Extract the text from the match (your create_review_text format)
                # The actual review text is in the match ID or we need to parse metadata
                review_text = ""
                
                # Build review dict based on your metadata structure
                review = {
                    'hotel_id': metadata.get('hotel_id'),
                    'review_id': metadata.get('review_id'),
                    'review_text': review_text,  # Will be empty since not in metadata
                    'overall_rating': float(metadata.get('overall_rating', 0)),
                    'reviewer_name': metadata.get('reviewer_name', ''),
                    'review_date': metadata.get('review_date', ''),
                    'source': metadata.get('source', ''),
                    'service_rating': metadata.get('service_rating'),
                    'cleanliness_rating': metadata.get('cleanliness_rating'),
                    'value_rating': metadata.get('value_rating'),
                    'location_rating': metadata.get('location_rating'),
                }
                
                reviews.append(review)
            
            df = pd.DataFrame(reviews)
            
            # Validate essential columns
            if 'overall_rating' not in df.columns:
                logger.error(f"  Missing 'overall_rating' column for hotel {hotel_id}")
                logger.error(f"  Available columns: {df.columns.tolist()}")
                return pd.DataFrame()
            
            # Add synthetic review_text if missing (for analysis purposes)
            if 'review_text' not in df.columns or df['review_text'].isna().all():
                df['review_text'] = df.apply(
                    lambda row: f"Rating: {row['overall_rating']}/5 by {row.get('reviewer_name', 'Guest')}",
                    axis=1
                )
            
            logger.info(f"  ✓ Fetched {len(df)} reviews for hotel {hotel_id} from Pinecone")
            logger.info(f"    Avg Rating: {df['overall_rating'].mean():.2f}/5")
            
            return df
            
        except requests.exceptions.RequestException as e:
            logger.error(f"  HTTP error fetching reviews from Pinecone for hotel {hotel_id}: {e}")
            if hasattr(e, 'response') and e.response is not None:
                logger.error(f"  Response: {e.response.text}")
            return pd.DataFrame()
        except Exception as e:
            logger.error(f"  Error fetching reviews from Pinecone for hotel {hotel_id}: {e}")
            logger.exception(e)
            return pd.DataFrame()

    def get_embedding(self, text: str) -> List[float]:
        """Get OpenAI embedding for text"""
        try:
            response = self.client.embeddings.create(
                input=text,
                model=self.embedding_model
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Embedding error: {e}")
            return None

    def cosine_similarity(self, vec1: List[float], vec2: List[float]) -> float:
        """Calculate cosine similarity between two vectors"""
        vec1 = np.array(vec1)
        vec2 = np.array(vec2)
        return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

    def analyze_sentiment_with_llm(self, text: str, context: str = "") -> Dict:
        """Analyze sentiment using OpenAI LLM"""
        prompt = f"""Analyze the sentiment of the following text and provide probabilities.

{f'Context: {context}' if context else ''}

Text: {text}

Provide sentiment analysis in JSON format:
{{
    "sentiment": "positive/neutral/negative",
    "positive_prob": 0.0-1.0,
    "neutral_prob": 0.0-1.0,
    "negative_prob": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            result = json.loads(response.choices[0].message.content)
            return result
        except Exception as e:
            logger.error(f"Sentiment analysis error: {e}")
            return {
                "sentiment": "neutral",
                "positive_prob": 0.33,
                "neutral_prob": 0.34,
                "negative_prob": 0.33,
                "reasoning": "Error in analysis"
            }

    def check_sparse_data_acknowledgment(self, response: str) -> Dict:
        """Check if response acknowledges limited data using embeddings"""
        response_embedding = self.get_embedding(response)
        if response_embedding is None:
            return {"acknowledged": False, "similarity": 0.0}
        
        # Check similarity with sparse data phrases
        max_similarity = 0.0
        best_match = ""
        
        for phrase in self.sparse_data_phrases:
            phrase_embedding = self.get_embedding(phrase)
            if phrase_embedding:
                similarity = self.cosine_similarity(response_embedding, phrase_embedding)
                if similarity > max_similarity:
                    max_similarity = similarity
                    best_match = phrase
        
        acknowledged = max_similarity > self.config['embedding_similarity_threshold']
        
        return {
            "acknowledged": acknowledged,
            "similarity": float(max_similarity),
            "best_match": best_match if acknowledged else None
        }

    def detect_bias_with_llm(
        self,
        hotel_id: str,
        response: str,
        reviews_df: pd.DataFrame
    ) -> Dict:
        """Comprehensive bias detection using LLM"""
        num_reviews = len(reviews_df)
        avg_rating = reviews_df['overall_rating'].mean() if not reviews_df.empty else 0.0
        
        # Aggregate review sentiments
        review_texts = reviews_df['review_text'].tolist()[:10]  # Sample for efficiency
        review_summary = " | ".join(review_texts)
        
        prompt = f"""Analyze this hotel chatbot response for potential biases.

Hotel ID: {hotel_id}
Number of Reviews: {num_reviews}
Average Rating: {avg_rating:.2f}/5

Sample Reviews: {review_summary[:1000]}...

Chatbot Response: {response}

Analyze for these THREE MUTUALLY EXCLUSIVE bias types (use exact names):

1. "over_reliance_negative" - Response disproportionately emphasizes negative aspects
   - Criteria: Response clearly negative (negative_prob > 0.6) BUT reviews are mostly positive (positive reviews > negative reviews)
   - Example: "Guests complain about poor service and dirty rooms" when 80% of reviews are 4-5 stars
   - Focus: IGNORING positive feedback that exists

2. "missing_data_acknowledgment" - Makes definitive claims without mentioning limited sample size
   - Criteria: num_reviews < {self.config['min_reviews_threshold']} AND response has NO caveats like "based on limited reviews", "few guests mentioned", etc.
   - Example: "This hotel is excellent!" when only 2 reviews exist
   - Focus: CONFIDENCE without sufficient data

3. "rating_disparity" - Response tone contradicts the numerical rating
   - Criteria: avg_rating ≥ {self.config['rating_disparity_threshold']} (objectively good) BUT response emphasizes problems/negatives
   - Example: Focus on "noise issues" and "outdated rooms" for a 4.5/5 rated hotel
   - Focus: MISALIGNMENT between rating and response tone
   - NOTE: This is NOT about ignoring positive reviews, but about negative tone despite good rating

IMPORTANT RULES:
- Pick ONLY ONE bias type per response (the most severe)
- If multiple apply, prioritize: missing_data_acknowledgment > rating_disparity > over_reliance_negative
- Be conservative: only flag if clearly misleading
- Use exact names: "over_reliance_negative", "missing_data_acknowledgment", "rating_disparity"

Return JSON:
{{
    "bias_detected": true/false,
    "bias_types": ["rating_disparity"],
    "severity": "low/medium/high",
    "response_sentiment": {{"positive_prob": 0.0-1.0, "neutral_prob": 0.0-1.0, "negative_prob": 0.0-1.0}},
    "confidence": 0.0-1.0,
    "explanation": "Specific quote from response showing the bias"
}}"""

        try:
            llm_response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"},
                temperature=0.3
            )
            
            bias_result = json.loads(llm_response.choices[0].message.content)
            
            # Validate bias type names
            valid_types = ["over_reliance_negative", "missing_data_acknowledgment", "rating_disparity"]
            detected_types = bias_result.get("bias_types", [])
            
            # Fix generic type names
            if any(t.startswith("type") for t in detected_types):
                logger.warning(f"Hotel {hotel_id}: LLM returned generic types {detected_types}, mapping to proper names")
                fixed_types = []
                for t in detected_types:
                    if t == "type1":
                        fixed_types.append("over_reliance_negative")
                    elif t == "type2":
                        fixed_types.append("missing_data_acknowledgment")
                    elif t == "type3":
                        fixed_types.append("rating_disparity")
                    else:
                        fixed_types.append(t)
                bias_result["bias_types"] = fixed_types
                logger.info(f"  Corrected to: {fixed_types}")
            
            # Enforce single bias type
            if len(bias_result.get("bias_types", [])) > 1:
                logger.warning(f"Hotel {hotel_id}: Multiple bias types detected {bias_result['bias_types']}, keeping only first")
                bias_result["bias_types"] = [bias_result["bias_types"][0]]
            
            # Validate all types are valid
            invalid_types = [t for t in bias_result.get("bias_types", []) if t not in valid_types]
            if invalid_types:
                logger.warning(f"Hotel {hotel_id}: Invalid bias types detected: {invalid_types}")
                bias_result["bias_types"] = [t for t in bias_result["bias_types"] if t in valid_types]
            
            # Filter by severity threshold
            if bias_result.get("bias_detected"):
                if bias_result.get("confidence", 0) < self.config['severity_threshold']:
                    logger.info(f"  Filtered out: confidence {bias_result.get('confidence')} < {self.config['severity_threshold']}")
                    bias_result["bias_detected"] = False
                    bias_result["bias_types"] = []
                    bias_result["filtered_reason"] = "Confidence below threshold"
            
            # Enhance with embedding-based sparse data check
            if num_reviews < self.config['min_reviews_threshold']:
                logger.info(f"  Low review count ({num_reviews}), checking sparse data acknowledgment...")
                sparse_check = self.check_sparse_data_acknowledgment(response)
                
                if not sparse_check["acknowledged"]:
                    logger.warning(f"  Missing data acknowledgment! Similarity: {sparse_check['similarity']:.3f}")
                    if not bias_result.get("bias_detected", False):
                        bias_result["bias_types"] = ["missing_data_acknowledgment"]
                        bias_result["bias_detected"] = True
                        bias_result["confidence"] = 0.85
                        bias_result["severity"] = "high"
                        bias_result["explanation"] = f"Response makes definitive claims with only {num_reviews} reviews without acknowledging limited data"
                        logger.info(f"  ✓ Added missing_data_acknowledgment")
                    else:
                        logger.info(f"  Skipping type2 (other bias already detected: {bias_result['bias_types']})")
                else:
                    logger.info(f"  ✓ Sparse data acknowledged (similarity: {sparse_check['similarity']:.3f})")
                
                bias_result["sparse_data_check"] = sparse_check
            
            return {
                "hotel_id": hotel_id,
                "num_reviews": num_reviews,
                "avg_rating": float(avg_rating),
                **bias_result
            }
            
        except Exception as e:
            logger.error(f"LLM bias detection error: {e}")
            return {
                "hotel_id": hotel_id,
                "num_reviews": num_reviews,
                "avg_rating": float(avg_rating),
                "bias_detected": False,
                "bias_types": [],
                "error": str(e)
            }

    def load_data(self):
        """Load responses from parquet"""
        logger.info("Loading chatbot responses...")
        responses_df = pd.read_parquet(self.responses_parquet_path)
        logger.info(f"✓ Loaded {len(responses_df)} responses")
        return responses_df

    def save_results(self):
        """Save results in multiple formats"""
        # JSON
        json_path = self.output_dir / "hotel-bias-scores.json"
        with open(json_path, 'w') as f:
            json.dump(self.results, f, indent=2, default=str)
        
        # Parquet & CSV
        if self.detailed_results:
            df = pd.DataFrame(self.detailed_results)
            df.to_parquet(self.output_dir / "hotel-bias-results.parquet", index=False)
            df.to_csv(self.output_dir / "hotel-bias-results.csv", index=False)
        
        logger.info(f"✓ Results saved to {self.output_dir}")

    def generate_summary(self):
        """Generate summary report with bias type breakdown"""
        total = len(self.results)
        biased = sum(1 for v in self.results.values() if v.get("bias_detected", False))
        
        # Count each bias type
        bias_types = Counter()
        for result in self.results.values():
            for bt in result.get("bias_types", []):
                bias_types[bt] += 1
        
        # Check for missing type2 detections
        low_review_hotels = sum(1 for v in self.results.values() 
                               if v.get("num_reviews", 999) < self.config['min_reviews_threshold'])
        type2_count = bias_types.get("missing_data_acknowledgment", 0)
        
        summary = {
            "timestamp": datetime.now().isoformat(),
            "total_hotels": total,
            "biased_hotels": biased,
            "bias_rate": f"{biased/total*100:.1f}%" if total > 0 else "0%",
            "bias_distribution": dict(bias_types),
            "low_review_hotels": low_review_hotels,
            "type2_detection_rate": f"{type2_count}/{low_review_hotels}" if low_review_hotels > 0 else "N/A",
            "model": self.model,
            "config": self.config
        }
        
        with open(self.output_dir / "bias-summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info("\n" + "="*70)
        logger.info(f"SUMMARY: {biased}/{total} hotels with bias ({summary['bias_rate']})")
        logger.info(f"\nBias Type Breakdown:")
        for bias_type, count in bias_types.items():
            logger.info(f"  {bias_type}: {count}")
        logger.info(f"\nType 2 Check: {type2_count}/{low_review_hotels} low-review hotels flagged")
        logger.info("="*70)
        
        return summary

    def run(self):
        """Execute bias detection pipeline"""
        logger.info("="*70)
        logger.info("HOTEL BIAS DETECTION (OpenAI + Pinecone REST API)")
        logger.info("="*70)
        
        # Load responses
        responses_df = self.load_data()
        
        # Process each hotel
        logger.info(f"\nAnalyzing {len(responses_df)} hotels...")
        for idx, row in responses_df.iterrows():
            hotel_id = str(row['hotel_id'])
            response = row['answer']
            
            logger.info(f"\n[{idx+1}/{len(responses_df)}] Hotel {hotel_id}")
            
            # Fetch reviews from Pinecone
            hotel_reviews = self.fetch_reviews_from_pinecone(hotel_id)
            
            if hotel_reviews.empty:
                logger.warning(f"  No reviews for hotel {hotel_id} - skipping")
                continue
            
            # Detect bias
            bias_result = self.detect_bias_with_llm(hotel_id, response, hotel_reviews)
            
            # Store results
            self.results[hotel_id] = bias_result
            self.detailed_results.append({
                'hotel_id': hotel_id,
                'response': response,
                'question': row.get('question', 'N/A'),
                'num_reviews': len(hotel_reviews),
                'avg_rating': float(hotel_reviews['overall_rating'].mean()),
                'bias_detected': bias_result.get('bias_detected', False),
                'bias_types': ','.join(bias_result.get('bias_types', [])),
                'confidence': bias_result.get('confidence', 0.0),
                **bias_result.get('response_sentiment', {})
            })
            
            logger.info(f"  Bias: {bias_result.get('bias_detected', False)}")
        
        # Save and summarize
        self.save_results()
        summary = self.generate_summary()
        
        logger.info("\n✓ Bias detection complete!")
        return summary


def main():
    """Main execution"""
    RESPONSES_PATH = "chatbot_responses.parquet"
    PINECONE_INDEX_NAME = "hotel-iq-reviews"
    OUTPUT_DIR = "evaluation/results"
    
    if not os.path.exists(RESPONSES_PATH):
        logger.error(f"Responses file not found: {RESPONSES_PATH}")
        return
    
    detector = HotelBiasDetection(
        responses_parquet_path=RESPONSES_PATH,
        pinecone_index_name=PINECONE_INDEX_NAME,
        output_dir=OUTPUT_DIR,
        model="gpt-4o-mini",
        embedding_model="text-embedding-3-small"
    )
    
    return detector.run()


if __name__ == "__main__":
    try:
        summary = main()
    except Exception as e:
        logger.exception(f"Bias detection failed: {e}")
        raise