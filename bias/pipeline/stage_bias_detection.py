import os
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# Add parent directory to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Load API Keys
# os.environ['HF_TOKEN'] = os.getenv("HF_TOKEN", "")
os.environ['OPENAI_API_KEY'] = os.getenv("OPENAI_API_KEY", "")
# os.environ['GROQ_API_KEY'] = os.getenv("GROQ_API_KEY", "")
# os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Now this should work
from main.hotel_bias_detection import HotelBiasDetection

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGE_NAME = "Hotel Bias Detection"


class BiasDetectionPipeline:
    def __init__(self):
        pass
    
    def detect_bias(self):
        """Run bias detection on chatbot responses"""
        
        # Paths
        # REVIEWS_PATH = "processed_boston_reviews.csv"
        RESPONSES_PATH = "response/chatbot_responses.parquet"
        OUTPUT_DIR = "evaluation/results"
        
        # Check files exist
        # if not Path(REVIEWS_PATH).exists():
        #     logger.error(f"Reviews file not found: {REVIEWS_PATH}")
        #     raise FileNotFoundError(f"Missing: {REVIEWS_PATH}")
        
        if not Path(RESPONSES_PATH).exists():
            logger.error(f"Chatbot responses file not found: {RESPONSES_PATH}")
            raise FileNotFoundError(f"Missing: {RESPONSES_PATH}")
        
        # Run bias detection
        detector = HotelBiasDetection(
            # reviews_path=REVIEWS_PATH,
            responses_parquet_path=RESPONSES_PATH,
            output_dir=OUTPUT_DIR
        )
        
        summary = detector.run()
        
        return summary
    
    def log_results_to_mlflow(self):
        """Optional: Log to MLflow"""
        pass


if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = BiasDetectionPipeline()
        summary = pipeline.detect_bias()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
        
    except Exception as e:
        logger.exception(e)
        raise
