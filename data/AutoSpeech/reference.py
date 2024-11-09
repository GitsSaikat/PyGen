import os
import sys
import logging
import yaml
from typing import Dict, Any
from pathlib import Path
import argparse

import Autospeech 
from autospeech.audiospeech_preprocessing import AudioSpeechPreprocessing
from autospeech.speech_recognition import SpeechRecognition
from autospeech.natural_language_processing import NaturalLanguageProcessing
from autospeech.word_embeddings import WordEmbeddings
from autospeech.realtime_processing import RealTimeProcessing

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('autospeech.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ConfigurationManager:
    """Secure configuration management"""
    
    def __init__(self, config_path: str):
        self.config_path = Path(config_path)
        self.config: Dict[str, Any] = {}
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration securely"""
        try:
            if not self.config_path.exists():
                raise FileNotFoundError(f"Config file not found: {self.config_path}")
                
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
                
            return self.config
            
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            raise

class AutoSpeechProcessor:
    """Main processing class with security features"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.preprocessor = AudioSpeechPreprocessing()
        self.recognizer = SpeechRecognition()
        self.nlp = NaturalLanguageProcessing()
        
    def validate_audio_file(self, file_path: str) -> bool:
        """Validate audio file security"""
        try:
            # Check file exists
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"Audio file not found: {file_path}")
                
            # Validate file extension
            allowed_extensions = {'.wav', '.mp3', '.flac'}
            if not Path(file_path).suffix.lower() in allowed_extensions:
                raise ValueError(f"Unsupported audio format. Use: {allowed_extensions}")
                
            # Check file size
            max_size = 50 * 1024 * 1024  # 50MB
            if os.path.getsize(file_path) > max_size:
                raise ValueError(f"File too large. Maximum size: {max_size/1024/1024}MB")
                
            return True
            
        except Exception as e:
            logger.error(f"File validation error: {str(e)}")
            raise
            
    def process_audio(self, audio_path: str) -> Dict[str, Any]:
        """Process audio with error handling"""
        try:
            # Validate input
            self.validate_audio_file(audio_path)
            
            # Load and preprocess
            logger.info(f"Processing audio file: {audio_path}")
            audio_data = Autospeech.load_audio(audio_path)
            preprocessed_audio = self.preprocessor.preprocess_audio(audio_data)
            
            # Recognition
            recognized_speech = self.recognizer.recognize_speech(preprocessed_audio)
            
            # NLP processing
            nlp_result = self.nlp.process_text(recognized_speech)
            
            return {
                'text': recognized_speech,
                'analysis': nlp_result
            }
            
        except Exception as e:
            logger.error(f"Processing error: {str(e)}")
            raise

def main():
    """Main function with CLI interface"""
    try:
        # Parse arguments
        parser = argparse.ArgumentParser(description='AutoSpeech Processor')
        parser.add_argument('--config', type=str, default='config.yml',
                          help='Path to configuration file')
        parser.add_argument('--input', type=str, required=True,
                          help='Path to input audio file')
        args = parser.parse_args()
        
        # Load configuration
        config_manager = ConfigurationManager(args.config)
        config = config_manager.load_config()
        
        # Initialize processor
        processor = AutoSpeechProcessor(config)
        
        # Process audio
        result = processor.process_audio(args.input)
        
        # Output results
        logger.info("Processing completed successfully")
        print(result)
        
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        sys.exit(1)

if __name__ == '__main__':
    main()
