from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
api_key = os.getenv('GOOGLE_GEMINI_API_KEY')

# Your training code here
if __name__ == '__main__':
    print("Training script initialized")
