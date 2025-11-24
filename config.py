"""
Configuration file for Legal Contract Analysis System
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """Base configuration class"""
    
    # Flask settings
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'your-secret-key-change-in-production'
    UPLOAD_FOLDER = os.environ.get('UPLOAD_FOLDER') or 'uploads'
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB max file size
    
    # Claude API settings
    CLAUDE_API_KEY = os.environ.get('CLAUDE_API_KEY')
    CLAUDE_MODEL = os.environ.get('CLAUDE_MODEL') or 'claude-sonnet-4-5-20250929'
    
    # Analysis settings
    TRANSACTIONAL_THRESHOLD = float(os.environ.get('TRANSACTIONAL_THRESHOLD', '0.3'))
    ADEQUACY_THRESHOLD = float(os.environ.get('ADEQUACY_THRESHOLD', '0.3'))
    BATCH_SIZE = int(os.environ.get('BATCH_SIZE', '5'))
    
    # Logging
    LOG_LEVEL = os.environ.get('LOG_LEVEL', 'INFO')
    LOG_FILE = os.environ.get('LOG_FILE', 'contract_analysis.log')

class DevelopmentConfig(Config):
    """Development configuration"""
    DEBUG = True
    
class ProductionConfig(Config):
    """Production configuration"""
    DEBUG = False

# Configuration dictionary
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
} 