
import os
import sys
from dotenv import load_dotenv
from google import genai
from google.genai import types

load_dotenv()

api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    
if not api_key:
    print("No API key found")
    sys.exit(1)

client = genai.Client(api_key=api_key, http_options=types.HttpOptions(client_args={"http2": False}))

config = types.GenerateContentConfig(
    thinking_config={"include_thoughts": True},
    temperature=0.7,
    max_output_tokens=100
)

models_to_try = ["gemini-2.0-flash"]

for model in models_to_try:
    print(f"\nTrying model: {model}")
    try:
        response_stream = client.models.generate_content_stream(
            model=model,
            contents="Explain how TCP works in 1 sentence.",
            config=config
        )

        for chunk in response_stream:
            if not chunk.candidates or not chunk.candidates[0].content or not chunk.candidates[0].content.parts:
                continue
            
            for part in chunk.candidates[0].content.parts:
                print(f"\n--- Part Debug ({model}) ---")
                print(f"Type: {type(part)}")
                # print(f"Dir: {dir(part)}")
                print(f"Part object repr: {repr(part)}")
                
                if hasattr(part, "thought"):
                    print(f"Has thought attr: {part.thought}")
                else:
                    print("No 'thought' attribute")
                    
                if hasattr(part, "text"):
                    print(f"Has text attr: {part.text}")
                else:
                    print("No 'text' attribute")
        print(f"Success with {model}")
        
    except Exception as e:
        print(f"Error with {model}: {e}")
