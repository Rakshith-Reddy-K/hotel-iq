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
from main.hotel_failure_detection import HotelFailureDetection

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

class FailureDetectionPipeline:
    
    def detect_failure(self):
        """Run failure detection and send email alerts"""
        
        results_path = Path("evaluation/results/bias-summary.json")
        
        if not results_path.exists():
            logger.error("Bias results not found")
            logger.error("Run stage_bias_detection.py first!")
            raise FileNotFoundError(f"Missing: {results_path}")
        
        detector = HotelFailureDetection()
        passed = detector.detect()
        
        return passed

if __name__ == "__main__":
    try:
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = BiasDetectionPipeline()
        summary = pipeline.detect_bias()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
        
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        
        pipeline = FailureDetectionPipeline()
        passed = pipeline.detect_failure()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
        
        

        if summary:
            biased_count = summary.get('biased_hotels', 0)
            total_count = summary.get('total_hotels', 0)
            bias_rate_str = summary.get('bias_rate', '0%')
            bias_rate = float(bias_rate_str.rstrip('%'))
            
            if not passed:
                logger.error(f"FAILED: {bias_rate_str} exceeds threshold%")
                logger.info("="*70)
                sys.exit(1)
            else:
                logger.info(f"PASSED: {bias_rate_str} ≤ threshold%")
                logger.info("="*70)
                sys.exit(0)
        else:
            logger.error("No summary generated")
            sys.exit(1) 

    except Exception as e:
        logger.exception(e)
        sys.exit(1)
