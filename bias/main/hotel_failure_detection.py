# hotel_failure_detection.py

import os
import json
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dotenv import load_dotenv

load_dotenv()

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class HotelFailureDetection:
    """Monitor bias detection results and send Gmail alerts"""
    
    def __init__(self):
        bias_path = Path('evaluation/results/bias-summary.json')
        
        if not bias_path.exists():
            raise FileNotFoundError(f"Bias results not found: {bias_path}")
        
        with open(bias_path, 'r') as f:
            self.bias_summary = json.load(f)
        
        # Thresholds
        self.thresholds = {
            'max_bias_rate': 0.3,
            'max_rating_disparity': 5,
            'max_over_reliance_negative': 3,
        }
        
        # Gmail configuration
        self.sender_email = os.getenv("SENDER_EMAIL")
        self.sender_password = os.getenv("SENDER_PASSWORD", "").replace(" ", "")
        self.receiver_email = os.getenv("RECEIVER_EMAIL")
    
    
    def check_metrics(self):
        """Check if bias metrics exceed thresholds"""
        
        total_hotels = self.bias_summary['total_hotels']
        biased_hotels = self.bias_summary['biased_hotels']
        
        # Parse bias_rate (it's a string like "95.5%")
        bias_rate_str = self.bias_summary['bias_rate']
        bias_rate = float(bias_rate_str.replace('%', '')) / 100
        
        bias_distribution = self.bias_summary.get('bias_distribution', {})
        
        failures = []
        
        # Check overall bias rate
        if bias_rate > self.thresholds['max_bias_rate']:
            failures.append({
                'metric': 'Overall Bias Rate',
                'value': f"{bias_rate*100:.1f}%",
                'threshold': f"{self.thresholds['max_bias_rate']*100:.1f}%"
            })
        
        # Check specific bias types
        for bias_type, count in bias_distribution.items():
            threshold_key = f"max_{bias_type}"
            
            if threshold_key in self.thresholds:
                if count > self.thresholds[threshold_key]:
                    failures.append({
                        'metric': bias_type.replace('_', ' ').title(),
                        'value': count,
                        'threshold': self.thresholds[threshold_key]
                    })
        
        return failures
    
    
    def send_email_gmail(self, subject: str, body: str):
        """Send email using Gmail SMTP"""
        
        if not all([self.sender_email, self.sender_password, self.receiver_email]):
            logger.warning("Gmail credentials not configured, printing to console")
            logger.info(f"\n{'='*70}")
            logger.info(f"📧 EMAIL: {subject}")
            logger.info(f"{'='*70}")
            logger.info(body)
            logger.info(f"{'='*70}\n")
            return
        
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.receiver_email
            msg['Subject'] = subject
            msg.attach(MIMEText(body, 'plain'))
            
            # Try port 465 (SSL) - more likely to work
            try:
                logger.info("Attempting to send via port 465 (SSL)...")
                with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=30) as server:
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
                
                logger.info(f"✓ Email sent via Gmail (SSL) to {self.receiver_email}")
                return
                
            except Exception as e1:
                logger.warning(f"Port 465 failed: {e1}")
                
                # Try port 587 (TLS) as backup
                logger.info("Attempting to send via port 587 (TLS)...")
                with smtplib.SMTP("smtp.gmail.com", 587, timeout=30) as server:
                    server.starttls()
                    server.login(self.sender_email, self.sender_password)
                    server.send_message(msg)
                
                logger.info(f"✓ Email sent via Gmail (TLS) to {self.receiver_email}")
                
        except Exception as e:
            logger.error(f"❌ Failed to send email: {e}")
            logger.error("\nYour network is blocking SMTP. Try:")
            logger.error("1. Different network (mobile hotspot)")
            logger.error("2. VPN")
            logger.error("3. Use SendGrid instead")
            logger.info(f"\nMessage that failed to send:")
            logger.info(f"Subject: {subject}")
            logger.info(body)
    
    
    def send_failure_alert(self, failures):
        """Send failure notification"""
        
        subject = "🚨 Hotel Bias Detection - FAILURE ALERT"
        
        body = "BIAS DETECTION FAILURE\n"
        body += "="*70 + "\n\n"
        body += "The following metrics exceeded acceptable thresholds:\n\n"
        
        for failure in failures:
            body += f"{failure['metric']}\n"
            body += f"   Current Value: {failure['value']}\n"
            body += f"   Threshold: {failure['threshold']}\n\n"
        
        body += "\nSUMMARY:\n"
        body += f"• Total hotels analyzed: {self.bias_summary['total_hotels']}\n"
        body += f"• Hotels with bias: {self.bias_summary['biased_hotels']}\n"
        body += f"• Bias detection rate: {self.bias_summary['bias_rate']}\n"
        body += f"• Timestamp: {self.bias_summary['timestamp']}\n\n"
        
        if 'bias_distribution' in self.bias_summary:
            body += "BIAS TYPE BREAKDOWN:\n"
            for bias_type, count in self.bias_summary['bias_distribution'].items():
                body += f"• {bias_type.replace('_', ' ').title()}: {count}\n"
            body += "\n"
        
        body += f"CRITICAL: {self.bias_summary['bias_rate']} of hotels show bias!\n\n"
        
        body += "ACTION REQUIRED:\n"
        body += "• Review evaluation/results/hotel-bias-results.csv\n"
        body += "• Check chatbot prompts - seems overly negative\n"
        rating_disparity_count = self.bias_summary.get('bias_distribution', {}).get('rating_disparity', 0)
        body += f"• {rating_disparity_count} hotels have rating_disparity (negative despite good ratings)\n"        
        body += "• Investigate why chatbot ignores positive reviews\n"
        self.send_email_gmail(subject, body)
    
    def send_success_alert(self):
        """Send success notification"""
        
        subject = "✅ Hotel Bias Detection - SUCCESS"
        
        bias_distribution = self.bias_summary.get('bias_distribution', {})
        
        body = "BIAS DETECTION SUCCESS\n"
        body += "="*70 + "\n\n"
        body += "All metrics are within acceptable thresholds!\n\n"
        
        body += "RESULTS:\n"
        body += f"• Total hotels analyzed: {self.bias_summary['total_hotels']}\n"
        body += f"• Hotels with bias: {self.bias_summary['biased_hotels']}\n"
        body += f"• Bias detection rate: {self.bias_summary['bias_rate']}\n\n"
        
        if bias_distribution:
            body += "BIAS TYPE DISTRIBUTION:\n"
            for bias_type, count in bias_distribution.items():
                body += f"• {bias_type.replace('_', ' ').title()}: {count}\n"
        
        body += f"\nTimestamp: {self.bias_summary['timestamp']}\n"
        
        self.send_email_gmail(subject, body)
    
    
    def detect(self):
        """Run failure detection"""
        
        logger.info("="*70)
        logger.info("HOTEL FAILURE DETECTION")
        logger.info("="*70)
        
        logger.info(f"\nAnalyzing results from: {self.bias_summary['timestamp']}")
        logger.info(f"Total hotels: {self.bias_summary['total_hotels']}")
        logger.info(f"Biased hotels: {self.bias_summary['biased_hotels']}")
        logger.info(f"Bias rate: {self.bias_summary['bias_rate']}")
        
        logger.info("\nChecking against thresholds...")
        
        failures = self.check_metrics()
        
        if failures:
            logger.warning(f"\n⚠️  {len(failures)} METRIC(S) EXCEEDED THRESHOLDS:")
            logger.warning("="*70)
            
            for failure in failures:
                logger.warning(f"❌ {failure['metric']}: {failure['value']} (Threshold: {failure['threshold']})")
            
            logger.warning("="*70)
            
            self.send_failure_alert(failures)
            return False
            
        else:
            logger.info("\n✅ ALL METRICS WITHIN ACCEPTABLE THRESHOLDS")
            logger.info("="*70)
            
            self.send_success_alert()
            return True


def main():
    detector = HotelFailureDetection()
    passed = detector.detect()
    
    if not passed:
        logger.error("\n❌ Quality check failed!")
        exit(1)
    else:
        logger.info("\n✓ Quality check passed!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.exception(f"Failure detection failed: {e}")
        raise