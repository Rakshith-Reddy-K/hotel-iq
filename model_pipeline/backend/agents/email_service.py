import os
import smtplib
from pathlib import Path
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime
from typing import Dict, Any

from dotenv import load_dotenv

from .booking_state import GuestInfo
from logger_config import get_logger

logger = get_logger(__name__)

BACKEND_DIR = Path(__file__).resolve().parent
ENV_PATH = BACKEND_DIR / ".env"
load_dotenv(ENV_PATH)

SMTP_HOST = os.getenv("HOTEL_SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = int(os.getenv("HOTEL_SMTP_PORT", "587"))

# Prefer new names, fall back to old ones if present
SMTP_USERNAME = os.getenv("HOTEL_SMTP_USERNAME") or os.getenv("SMTP_USER") or ""
SMTP_PASSWORD = os.getenv("HOTEL_SMTP_PASSWORD", "")

SENDER_EMAIL = os.getenv("HOTEL_SMTP_FROM_EMAIL") or os.getenv("HOTEL_SENDER_EMAIL") or SMTP_USERNAME
SENDER_NAME = os.getenv("HOTEL_SENDER_NAME", "HotelIQ Booking")

SMTP_USE_TLS = os.getenv("HOTEL_SMTP_USE_TLS", "true").lower() == "true"


def _smtp_config_ok() -> bool:
    """
    Check if required SMTP configuration is present and log it once.
    """
    missing = []
    for key, val in {
        "SMTP_HOST": SMTP_HOST,
        "SMTP_PORT": SMTP_PORT,
        "SMTP_USERNAME": SMTP_USERNAME,
        "SMTP_PASSWORD": SMTP_PASSWORD,
        "SENDER_EMAIL": SENDER_EMAIL,
    }.items():
        if not val:
            missing.append(key)

    if missing:
        logger.warning(
            "SMTP credentials not configured",
            missing=missing,
            env_path=str(ENV_PATH),
        )
        return False

    logger.info(
        "SMTP configuration loaded",
        host=SMTP_HOST,
        port=SMTP_PORT,
        username=SMTP_USERNAME,
        sender=SENDER_EMAIL,
        use_tls=SMTP_USE_TLS,
    )
    return True


def generate_email_html(
    guest_info: GuestInfo,
    booking_id: str,
    hotel_name: str,
    hotel_info: Dict[str, Any],
) -> str:
    """
    Generate HTML email content for booking confirmation.
    """
    check_in = guest_info.check_in_date.strftime("%B %d, %Y")
    check_out = guest_info.check_out_date.strftime("%B %d, %Y")
    nights = (guest_info.check_out_date - guest_info.check_in_date).days

    hotel_address = hotel_info.get("address", "")
    hotel_phone = hotel_info.get("phone", "")
    hotel_star = hotel_info.get("star_rating", "")

    html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; line-height: 1.6; color: #333; }}
            .container {{ max-width: 600px; margin: 0 auto; padding: 20px; }}
            .header {{ background-color: #2563eb; color: white; padding: 20px; text-align: center; }}
            .content {{ background-color: #f9fafb; padding: 20px; }}
            .booking-details {{ background-color: white; padding: 20px; margin: 20px 0; border-radius: 8px; }}
            .detail-row {{ display: flex; justify-content: space-between; padding: 10px 0; border-bottom: 1px solid #e5e7eb; }}
            .detail-label {{ font-weight: bold; }}
            .footer {{ text-align: center; padding: 20px; color: #6b7280; font-size: 12px; }}
            .button {{ background-color: #2563eb; color: white; padding: 12px 24px; text-decoration: none; border-radius: 6px; display: inline-block; margin: 20px 0; }}
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>Booking Confirmation</h1>
                <p>Thank you for choosing HotelIQ!</p>
            </div>

            <div class="content">
                <p>Dear {guest_info.first_name} {guest_info.last_name},</p>

                <p>Your booking has been confirmed! We're excited to welcome you.</p>

                <div class="booking-details">
                    <h2>Booking Details</h2>

                    <div class="detail-row">
                        <span class="detail-label">Booking ID:</span>
                        <span>{booking_id}</span>
                    </div>

                    <div class="detail-row">
                        <span class="detail-label">Hotel:</span>
                        <span>{hotel_name} {"⭐" * int(float(hotel_star)) if hotel_star else ""}</span>
                    </div>

                    <div class="detail-row">
                        <span class="detail-label">Check-in:</span>
                        <span>{check_in}</span>
                    </div>

                    <div class="detail-row">
                        <span class="detail-label">Check-out:</span>
                        <span>{check_out}</span>
                    </div>

                    <div class="detail-row">
                        <span class="detail-label">Duration:</span>
                        <span>{nights} night(s)</span>
                    </div>

                    <div class="detail-row">
                        <span class="detail-label">Guests:</span>
                        <span>{guest_info.num_guests}</span>
                    </div>

                    {f'<div class="detail-row"><span class="detail-label">Address:</span><span>{hotel_address}</span></div>' if hotel_address else ''}

                    {f'<div class="detail-row"><span class="detail-label">Phone:</span><span>{hotel_phone}</span></div>' if hotel_phone else ''}
                </div>

                <div style="background-color: #fef3c7; padding: 15px; border-radius: 8px; margin: 20px 0;">
                    <p style="margin: 0;"><strong>Important:</strong> Full payment is required at check-in. Please bring a valid ID and payment method.</p>
                </div>

                <p>If you have any questions, please contact the hotel directly or reply to this email.</p>

                <p>We look forward to hosting you!</p>

                <p>Best regards,<br>
                The HotelIQ Team</p>
            </div>

            <div class="footer">
                <p>This is an automated confirmation email from HotelIQ.</p>
                <p>Booking Date: {datetime.now().strftime("%B %d, %Y at %I:%M %p")}</p>
            </div>
        </div>
    </body>
    </html>
    """

    return html


async def send_booking_confirmation_email(
    guest_info: GuestInfo,
    booking_id: str,
    hotel_name: str,
    hotel_info: Dict[str, Any],
) -> bool:
    """
    Send booking confirmation email to guest.

    Returns:
        True if email sent successfully, False otherwise
    """
    if not _smtp_config_ok():
        return False

    try:
        # Create message
        msg = MIMEMultipart("alternative")
        msg["From"] = f"{SENDER_NAME} <{SENDER_EMAIL}>"
        msg["To"] = guest_info.email
        msg["Subject"] = f"Booking Confirmation - {booking_id} - {hotel_name}"

        # Generate HTML content
        html_content = generate_email_html(guest_info, booking_id, hotel_name, hotel_info)

        # Attach HTML
        html_part = MIMEText(html_content, "html")
        msg.attach(html_part)

        # Send email
        if SMTP_USE_TLS:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as server:
                server.starttls()
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)
        else:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT) as server:
                server.login(SMTP_USERNAME, SMTP_PASSWORD)
                server.send_message(msg)

        logger.info(
            "Booking confirmation email sent",
            booking_id=booking_id,
            email=guest_info.email,
        )
        return True

    except Exception as e:
        logger.error("Failed to send booking confirmation email", error=str(e))
        return False
