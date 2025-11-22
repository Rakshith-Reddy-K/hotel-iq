# pipeline/stage_failure_detection.py

import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

sys.path.append(str(Path(__file__).parent.parent))

from main.hotel_failure_detection import HotelFailureDetection

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

STAGE_NAME = "Hotel Failure Detection"


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
        
        pipeline = FailureDetectionPipeline()
        passed = pipeline.detect_failure()
        
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\n")
        
        if not passed:
            exit(1)
        
    except Exception as e:
        logger.exception(e)
        raise