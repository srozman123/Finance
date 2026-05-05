# Copy this file to config.py and fill in your credentials.
# See the setup instructions at the top of phase5_monitor.py.
#
# Gmail App Password setup:
# 1. Enable 2-Factor Authentication at myaccount.google.com/security
# 2. Search "App Passwords" in the Security settings search bar
# 3. Generate a new App Password for "Mail" — Google gives you a 16-char code
# 4. Paste that code into EMAIL_PASSWORD below and set EMAIL_ENABLED = True

EMAIL_ENABLED        = False
EMAIL_FROM           = 'your.gmail@gmail.com'
EMAIL_TO             = 'your.gmail@gmail.com'
EMAIL_PASSWORD       = 'your_app_password_here'

# Send the morning briefing email even when there are no alerts.
# The monitor checks whether the current local hour is 8 (08:xx) and, if so,
# treats the run as the daily briefing and bypasses alert-count suppression.
ALWAYS_SEND_MORNING  = True
