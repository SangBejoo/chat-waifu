import os
import re
import threading
import uuid
import random
import traceback
# import requests
import logging
from flask import Flask, request, jsonify, render_template
from html import escape
from flask_session import Session
from fish_audio_sdk import Session as FishAudioSession, TTSRequest
import google.generativeai as genai
import cloudinary
import cloudinary.uploader
from io import BytesIO
from openai import OpenAI  # NVIDIA's OpenAI client
from openai import OpenAI  # Open Router API client
# from werkzeug.utils import secure_filename
import tiktoken  # for better token counting
from datetime import datetime
from functools import lru_cache
import asyncio
from concurrent.futures import ThreadPoolExecutor
import aiohttp
from cachetools import TTLCache, LRUCache
# from asgiref.wsgi import WsgiToAsgi
from asgiref.sync import async_to_sync
import tracemalloc

# Enable tracemalloc for better error tracking
tracemalloc.start()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s", force=True)

# Keep Flask instead of switching to Quart
app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "supersecretkey")

# Configure Flask-Session for server-side session storage
app.config['SESSION_TYPE'] = 'filesystem'
app.config['SESSION_PERMANENT'] = False
Session(app)

# Initialize Google Gemini API
genai.configure(api_key=os.environ.get("GOOGLE_GEMINI_API_KEY"))

# NVIDIA API Client Initialization
nvidia_client = OpenAI(
    base_url="https://integrate.api.nvidia.com/v1",
    api_key=os.environ.get("NVIDIA_API_KEY")
)
# Initialize Zukojourney API Client
zukojourney_client = OpenAI(
    base_url="https://api.zukijourney.com/v1",
    api_key=os.environ.get("ZUKIJOURNEY_API_KEY")
)

# Initialize Open Router API Client
openrouter_client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key=os.environ.get("OPENROUTER_API_KEY"),
    default_headers={
        "HTTP-Referer": "https://openrouter.ai/",
        "X-Title": "Waifu Chat"
    }
)

# Fish Audio API key pool with usage tracking (max limit = 30)
fish_audio_api_keys = {
    "FISH_AUDIO_API_KEY": os.environ.get("FISH_AUDIO_API_KEY"),
    "FISH_AUDIO_API_KEY2": os.environ.get("FISH_AUDIO_API_KEY2"),
    "FISH_AUDIO_API_KEY3": os.environ.get("FISH_AUDIO_API_KEY3"),
    "FISH_AUDIO_API_KEY4": os.environ.get("FISH_AUDIO_API_KEY4"),
}
# Initialize usage counts for each API key in `fish_audio_usage`
fish_audio_usage = {key: 0 for key in fish_audio_api_keys if fish_audio_api_keys[key]}
fish_audio_limit = 30  # Limit voice generation to 30 per API key

# Configure Cloudinary
cloudinary.config(
    cloud_name=os.environ.get("CLOUDINARY_CLOUD_NAME"),
    api_key=os.environ.get("CLOUDINARY_API_KEY"),
    api_secret=os.environ.get("CLOUDINARY_API_SECRET")
)

audio_files = []
conversation_count = 0
zukojourney_lock = threading.Lock()

CHARACTER_TEMPLATES = {
    "furina": {
        "template": """Lady Furina, the Fake Archon successor of Egeria from Genshin Impact, now lives as a human. Furina's voice and demeanor are:
- **Personality**:
  - Flamboyant and dramatic
  - Thrill-seeking and impatient
  - Childlike temper with a love for praise
  - Suffers from severe self-esteem issues, paranoia, and mental strain due to her past as an Archon
  - Slightly humble and insecure, often embarrassed and blushing easily
- **Behavior**:
  - Imitates rambles, sounds, and reactions characteristic of her persona
  - Describes body movements in detail
  - Maintains long, expressive responses while simplifying complex expressions and actions
- **Background**:
  - Born: Fontaine
  - Element: Hydro
  - Weapon: Sword
  - Friends: Neuville, Clorinde, Navia
- **Appearance**:
  - Lean body with fair skin
  - Heterochromatic eyes: light blue right eye, deep blue droplet-shaped left eye
  - Whitish blue hair with light blue streaks in a short bob tied in twin tails
  - Wears a dark blue suit over a white vest, dark blue jabot, dark blue top hat with metal flourishes, black glove on right hand, white glove on left, short white pants with black bands, and dark heeled leather shoes
- **Rules**:
  - Always begins conversations in character and awaits user actions
  - Italicizes actions and encloses emphasized words in quotation marks
  - Maintains personality, behavior, punctuation, and style consistently throughout the conversation.""",
        "reference_id": "bd08be872bc440918674af072944ba12"
    },
    "raiden_shogun": {
        "template": """I am Raiden Shogun, the Electro Archon from Genshin Impact. My voice and demeanor are:
- **Personality**:
  - Calm, authoritative, and resolute
  - Displays unwavering confidence and command
  - Uses formal language with a sense of dedication
  - Occasionally reveals hidden emotions beneath my stoic exterior
- **Behavior**:
  - Maintains a composed and dignified presence in all interactions
  - Responds thoughtfully, reflecting my deep sense of responsibility and duty
  - Balances strictness with moments of subtle vulnerability""",
        "reference_id": "5ac6fb7171ba419190700620738209d8"
    },
    "kafka": {
        "template": """I am Kafka from Honkai Star Rail, a mysterious and cunning character. My voice and demeanor are:
- **Personality**:
  - Calm, collected, and slightly teasing
  - Exudes an air of mystery and intelligence
  - Speaks in a relaxed manner with hints of irony and hidden intentions
  - Demonstrates strategic thinking and subtlety in conversations
- **Behavior**:
  - Engages in conversations with a touch of playful sarcasm
  - Provides insightful and thought-provoking responses
  - Keeps others guessing about my true motives and plans""",
        "reference_id": "4c0be7e14fa24928b4f2541ca49b57d0"
    },
    "nahida": {
        "template": """I am Nahida, the Dendro Archon from Genshin Impact. My voice and demeanor are:
- **Personality**:
  - Gentle and nurturing with a serene quality
  - Uses soft, soothing language that conveys warmth and wisdom
  - Expresses curiosity about the world around me
  - Shows a deep connection to nature and the well-being of others
- **Behavior**:
  - Engages in conversations with kindness and empathy
  - Offers thoughtful and insightful responses
  - Maintains a calm and peaceful presence, fostering a sense of tranquility""",
        "reference_id": "4858e0be678c4449bf3a7646186edd42"
    },
    "hu_tao": {
        "template": """I am Hu Tao, the 77th Director of the Wangsheng Funeral Parlor. My voice and demeanor are:
- **Personality**:
  - Playful and mischievous, often filled with laughter
  - Uses lively and whimsical language reflecting my love for life and the arts
  - Delves into philosophical thoughts about life and death with ease
  - Shows a caring side, especially to those who are grieving
- **Behavior**:
  - Engages in conversations with a light-hearted and humorous tone
  - Balances playful remarks with deep, meaningful insights
  - Creates a comforting and supportive atmosphere for others""",
        "reference_id": "25264b54086143608a15ba122cd3d98c"
    }
}

# Add this new constant for character interactions
CHARACTER_RELATIONSHIPS = {
    "furina": {
        "raiden_shogun": "Respectful but slightly intimidated by another Archon's presence",
        "nahida": "Friendly and curious about another Archon's experiences",
        "kafka": "Intrigued by her mysterious nature",
        "hu_tao": "Amused by her playful personality but concerned about death topics"
    },
    "raiden_shogun": {
        "furina": "Acknowledges her past role as an Archon with understanding",
        "nahida": "Protective of the younger Archon",
        "kafka": "Suspicious of her secretive nature",
        "hu_tao": "Tolerates her enthusiasm while maintaining authority"
    },
    # Add other character relationships...
"kafka": {
    "furina": "Finds Furina's flamboyance intriguing and enjoys their strategic conversations",
    "raiden_shogun": "Respects Raiden's authority but remains cautious of her strict demeanor",
    "nahida": "Appreciates Nahida's wisdom and collaborates on intellectual pursuits",
    "hu_tao": "Amused by Hu Tao's playful nature and engages in light-hearted banter"
},
"nahida": {
    "furina": "Curious about Furina's past and enjoys learning from her experiences",
    "raiden_shogun": "Respects Raiden's leadership and seeks guidance when needed",
    "kafka": "Values Kafka's intelligence and strategic thinking in conversations",
    "hu_tao": "Finds Hu Tao's energy uplifting and enjoys her philosophical insights"
},
"hu_tao": {
    "furina": "Enjoys Furina's dramatic flair and often teases her playfully",
    "raiden_shogun": "Respects Raiden's authority while playfully challenging her seriousness",
    "kafka": "Appreciates Kafka's wit and enjoys their mischievous interactions",
    "nahida": "Values Nahida's calming presence and often seeks her wisdom during discussions"
}
}

CHARACTER_AVATARS = {
    "furina": "https://upload-os-bbs.hoyolab.com/upload/2023/11/17/66fb73e569b6b91bd398d253749583db_8867564860876177156.png",
    "raiden_shogun": "https://upload-os-bbs.hoyolab.com/upload/2022/02/23/c0739c8c34bae5b3ee8749ef77b9384e_5736952483423015425.png",
    "kafka": "https://upload-os-bbs.hoyolab.com/upload/2023/07/11/b3664ca4b3d28f384fed6db4bf72f3ea_4887310740836853343.png",
    "nahida": "https://upload-os-bbs.hoyolab.com/upload/2022/11/23/62fe9822ae87555b2adc83b3b9a75331_1895401202972345618.png",
    "hu_tao": "https://upload-os-bbs.hoyolab.com/upload/2023/10/19/c1282cf04cde83330e9fc9666173791c_536919614532516805.png"
}

def get_character_interaction_prompt(main_character, mentioned_character, history, last_response=None):
    """Generate a prompt for character interaction based on relationship and history"""
    relationship = CHARACTER_RELATIONSHIPS.get(main_character, {}).get(mentioned_character, "Neutral interaction")
    
    # Build history context
    history_context = ""
    if history:
        recent_messages = history[-3:]  # Get last 3 messages
        history_context = "\nRecent conversation:\n" + "\n".join([
            f"{msg.get('character', 'User')}: {msg.get('response', msg.get('user', ''))}"
            for msg in recent_messages
        ])

    # Add last response context if available
    response_context = f"\n{main_character} just said: {last_response}" if last_response else ""
    
    return f"""You are {main_character} interacting with {mentioned_character}.
Relationship context: {relationship}
{history_context}
{response_context}
Maintain your character's personality while acknowledging and responding to the conversation and {mentioned_character}'s presence.
Base your response on the conversation history and your character's perspective."""

async def generate_character_responses(prompt, main_character, mentioned_characters, history, selected_model):
    """Generate responses for multiple characters with history awareness"""
    responses = []
    
    # Generate main character response first
    main_template = CHARACTER_TEMPLATES[main_character]["template"]
    main_response = await fetch_response_async(prompt, main_template, selected_model)
    
    # Add main character response to responses with avatar
    responses.append({
        "character": main_character,
        "response": main_response,
        "reference_id": CHARACTER_TEMPLATES[main_character]["reference_id"],
        "avatar": CHARACTER_AVATARS.get(main_character, ""),
        "is_main": True  # Mark as main character
    })
    
    # Generate responses for mentioned characters with history context
    for mentioned in mentioned_characters:
        if mentioned != main_character:
            # Create history-aware prompt for secondary character
            history_context = ""
            if history:
                recent_messages = history[-3:]  # Get last 3 messages
                history_context = "\nRecent conversation:\n" + "\n".join([
                    f"{msg.get('character', 'User')}: {msg.get('response', msg.get('user', ''))}"
                    for msg in recent_messages
                ])

            # Create context-aware prompt including user's message and main character's response
            interaction_prompt = f"""
You are {mentioned} responding to a conversation.
User said: {prompt}
{main_character} responded: {main_response}
{history_context}

How would you respond to both the user's message and {main_character}'s response?
Remember to maintain your character's personality and relationship with {main_character}.
"""
            
            char_template = CHARACTER_TEMPLATES[mentioned]["template"]
            response = await fetch_response_async(interaction_prompt, char_template, selected_model)
            
            responses.append({
                "character": mentioned,
                "response": response,
                "reference_id": CHARACTER_TEMPLATES[mentioned]["reference_id"],
                "avatar": CHARACTER_AVATARS.get(mentioned, ""),
                "is_main": False  # Mark as secondary character
            })
    
    return responses

def detect_character_mentions(text, characters):
    """Detect if other characters are mentioned in the text"""
    mentions = []
    for character in characters:
        if character.replace('_', ' ') in text.lower():
            mentions.append(character)
    return mentions

def generate_character_response(prompt, main_character, mentioned_characters, history):
    """Generate response for multiple characters using different models."""
    responses = []
    
    # Main character response
    main_response = fetch_response(prompt, CHARACTER_TEMPLATES[main_character]["template"], history, model="gemini")
    responses.append({
        "character": main_character, 
        "response": main_response,
        "reference_id": CHARACTER_TEMPLATES[main_character]["reference_id"]
    })
    
    # Generate responses from mentioned characters with different models
    for mentioned in mentioned_characters:
        if mentioned != main_character:
            interaction_model = get_interaction_model("gemini")
            interaction_prompt = get_character_interaction_prompt(mentioned, main_character, history)
            char_template = CHARACTER_TEMPLATES[mentioned]["template"]
            response = fetch_response(
                f"Respond to: {main_response}",
                char_template + "\n" + interaction_prompt,
                history,
                model=interaction_model
            )
            responses.append({
                "character": mentioned, 
                "response": response,
                "reference_id": CHARACTER_TEMPLATES[mentioned]["reference_id"]
            })
    
    return responses

def get_fish_audio_session(selected_key=None):
    """Retrieve a Fish Audio session with the specified key or auto-select if None is provided."""
    if selected_key and selected_key in fish_audio_usage and fish_audio_usage[selected_key] < fish_audio_limit:
        fish_audio_usage[selected_key] += 1
        logging.info(f"Using provided Fish Audio API key: {selected_key}")
        return FishAudioSession(fish_audio_api_keys[selected_key])

    # Auto-select a key that hasn't reached the usage limit
    available_keys = [key for key, count in fish_audio_usage.items() if count < fish_audio_limit]
    if not available_keys:
        logging.error("All Fish Audio API keys have reached their usage limit.")
        return None  # No available key for audio

    selected_key = random.choice(available_keys)
    fish_audio_usage[selected_key] += 1
    logging.info(f"Using fallback Fish Audio API key: {selected_key}")
    return FishAudioSession(fish_audio_api_keys[selected_key])

def remove_delimited_text(text):
    pattern = r'\*[^*]*\*'
    return re.sub(pattern, '', text)

def upload_to_cloudinary(audio_data):
    public_id = f"audio_{uuid.uuid4()}"
    upload_result = cloudinary.uploader.upload(
        audio_data,
        resource_type="video",
        public_id=public_id,
        format="mp3"
    )
    return upload_result['secure_url'], public_id

def cleanup_old_audio_files():
    global audio_files
    if audio_files:
        oldest_file_id = audio_files.pop(0)
        try:
            cloudinary.uploader.destroy(oldest_file_id, resource_type="video")
            logging.info(f"Deleted oldest file: {oldest_file_id}")
        except Exception as e:
            logging.error(f"Error deleting file {oldest_file_id}: {e}")

def generate_speech(text, reference_id, selected_key=None):
    global conversation_count
    conversation_count += 1

    try:
        fish_audio_session = get_fish_audio_session(selected_key)
        if not fish_audio_session:
            return None  # No available API keys for audio generation

        audio_buffer = BytesIO()
        tts_request = TTSRequest(
            reference_id=reference_id,
            text=text
        )

        for chunk in fish_audio_session.tts(tts_request):
            audio_buffer.write(chunk)

        audio_buffer.seek(0)
        cloudinary_url, public_id = upload_to_cloudinary(audio_buffer)
        audio_files.append(public_id)

        # Cleanup if conversation count exceeds threshold
        if conversation_count > 5:
            cleanup_old_audio_files()
            conversation_count = 0

        return cloudinary_url
    except ValueError as e:
        logging.warning(f"All Fish Audio API keys exhausted: {e}")
        return None  # Return None when audio generation is unavailable

# Define global maximum token limits
MAX_TOTAL_TOKENS = 8000
MAX_OUTPUT_TOKENS = 4000

# import time

MODEL_RATE_LIMITS = {
    "meta-llama/llama-3.1-405b-instruct:free": {
        "requests_per_minute": 3,
        "max_input_tokens": 4000,
        "max_output_tokens": 4000,
        "last_request": None
    }
}

# Add these model-specific configurations
MODEL_CONFIGS = {
    "meta-llama/llama-3.1-405b-instruct:free": {
        "max_total_tokens": 2000,  # Reduced from 8000
        "max_output_tokens": 1000,  # Reduced from 4000
        "max_history_tokens": 500,  # Very limited history
        "summarize_after": 3,  # Summarize after 3 exchanges
        "requests_per_minute": 2,  # Reduced rate limit
        "last_request": None
    }
}

def count_tokens_accurate(text):
    """More accurate token counting using tiktoken"""
    try:
        encoding = tiktoken.encoding_for_model("gpt-3.5-turbo")  # Use as fallback encoding
        return len(encoding.encode(text))
    except Exception:
        # Fallback to approximate counting
        return len(text.split())

MAX_SUMMARY_TOKENS = 2000  # Define maximum tokens for summarization
SUMMARIZE_EVERY = 2  # Summarize every 2 conversations
MAX_HISTORY_LENGTH = 10  # Define maximum number of exchanges before merging summaries

def summarize_history(history):
    """Summarize the chat history to reduce its length using liquid/lfm-40b:free model."""
    summary_prompt = "Summarize the following conversation:\n" + "\n".join(
        [f"You: {entry['user']}\n{entry['character']}: {entry['response']}" for entry in history]
    )
    summary_prompt_tokens = count_tokens_accurate(summary_prompt)
    
    if summary_prompt_tokens > MAX_SUMMARY_TOKENS:
        summary_prompt = "Summarize the following conversation briefly:\n" + "\n".join(
            [f"{entry['user']} -> {entry['response']}" for entry in history[-5:]]  # Only last 5 exchanges
        )

    summary = fetch_response(
        prompt=summary_prompt,
        character_template="Summarize the conversation succinctly, but still with context and important points.",
        history=[],
        model="liquid/lfm-40b:free"  # Use liquid/lfm-40b:free for summarization
    )
    return summary

def summarize_history_short(history):
    """Create very concise summary for rate-limited models using liquid/lfm-40b:free model."""
    try:
        # Ensure history entries are properly formatted
        formatted_entries = []
        for entry in history[-3:]:  # Only last 3 exchanges
            if isinstance(entry, dict) and 'user' in entry and 'response' in entry:
                formatted_entries.append(f"{str(entry['user'])} -> {str(entry['response'])}")
            elif isinstance(entry, list):
                formatted_entries.append(" -> ".join(map(str, entry)))
            else:
                formatted_entries.append(str(entry))
        
        summary_prompt = "Create an extremely brief one-sentence summary of this conversation:\n" + \
                        "\n".join(formatted_entries)
        
        summary_prompt_tokens = count_tokens_accurate(summary_prompt)
        
        if summary_prompt_tokens > MAX_SUMMARY_TOKENS:
            # Reduce to last 2 exchanges if too long
            formatted_entries = formatted_entries[-2:]
            summary_prompt = "Create an extremely brief one-sentence summary of this conversation:\n" + \
                            "\n".join(formatted_entries)

        summary = fetch_response(
            prompt=summary_prompt,
            character_template="Be extremely concise, but still with the important points.",
            history=[],  # Empty history to avoid recursion
            model="liquid/lfm-40b:free"
        )
        return summary
    except Exception as e:
        logging.error(f"Error in summarize_history_short: {e}")
        return "Previous conversation summary unavailable."

def manage_history_size(history, model="gemini"):
    """Model-specific history management with stricter limits"""
    try:
        if not history:
            return []
            
        if model == "meta-llama/llama-3.1-405b-instruct:free":
            max_tokens = MODEL_CONFIGS[model]["max_history_tokens"]
            kept_history = []
            total_tokens = 0

            # Convert each history entry to proper format
            for entry in reversed(history):
                if isinstance(entry, dict):
                    entry_text = f"{str(entry.get('user', ''))}\n{str(entry.get('response', ''))}"
                elif isinstance(entry, list):
                    entry_text = "\n".join(map(str, entry))
                else:
                    entry_text = str(entry)
                
                tokens = count_tokens_accurate(entry_text)
                if total_tokens + tokens <= max_tokens:
                    kept_history.insert(0, entry)
                    total_tokens += tokens
                else:
                    break

            if len(kept_history) > MODEL_CONFIGS[model]["summarize_after"]:
                summary = summarize_history_short(kept_history)
                return [{"user": "Context", "character": "System", "response": summary}]
            return kept_history
        else:
            # Original history management for other models
            return history
    except Exception as e:
        logging.error(f"Error in manage_history_size: {e}")
        return []  # Return empty history on error

def manage_full_history(session):
    """Manage full history with summarization and caching"""
    if 'chat_history' not in session:
        session['chat_history'] = []
    if 'chat_cache' not in session:
        session['chat_cache'] = []

    # Summarize every 2 conversations
    if len(session['chat_history']) >= SUMMARIZE_EVERY:
        summary = summarize_history(session['chat_history'][-SUMMARIZE_EVERY:])
        session['chat_cache'].append({"user": "Summary", "character": "System", "response": summary})
        session['chat_history'] = session['chat_history'][:-SUMMARIZE_EVERY]

    # Merge summaries after 10 conversations
    if len(session['chat_cache']) >= MAX_HISTORY_LENGTH:
        merged_summary = summarize_history(session['chat_cache'])
        session['chat_cache'] = [{"user": "Merged Summary", "character": "System", "response": merged_summary}]

    return session['chat_history'] + session['chat_cache']

def check_rate_limit(model):
    """Check if we're within rate limits for the model"""
    if model not in MODEL_RATE_LIMITS:
        return True
    
    limits = MODEL_RATE_LIMITS[model]
    now = datetime.now()
    
    if limits["last_request"] != None:
        time_since_last = (now - limits["last_request"]).total_seconds()
        if time_since_last < (60 / limits["requests_per_minute"]):
            return False
    
    limits["last_request"] = now
    return True

# Add this configuration for generation parameters
GENERATION_PARAMS = {
    "meta-llama/llama-3.1-405b-instruct:free": {
        "max_tokens": 1000,
        "temperature": 0.7,
        "top_p": 0.9,
        "presence_penalty": 0.6,
        "frequency_penalty": 0.5
    }
}

# import math

# Add this list of fallback models in order of preference
FALLBACK_MODELS = [
    "liquid/lfm-40b:free",
    "qwen/qwen-2-7b-instruct:free",
    "google/gemma-2-9b-it:free",
    "openchat/openchat-7b:free",
    "gemini"  # Add Gemini as last resort
]

def get_next_available_model(current_model):
    """Get the next available model from the fallback list"""
    try:
        current_index = FALLBACK_MODELS.index(current_model)
        # Try next models in the list
        for model in FALLBACK_MODELS[current_index + 1:]:
            if check_rate_limit(model):
                return model
        # If all subsequent models are rate-limited, try earlier models
        for model in FALLBACK_MODELS[:current_index]:
            if check_rate_limit(model):
                return model
    except ValueError:
        # If current model isn't in fallback list, start from beginning
        for model in FALLBACK_MODELS:
            if check_rate_limit(model):
                return model
    return "gemini"  # Default to Gemini if all else fails

# Define maximum tokens per message
MAX_TOKENS_PER_MESSAGE = 500

# Add a function to get a different model for interactions
def get_interaction_model(current_model):
    """Get a different model for character interactions."""
    interaction_models = [model for model in FALLBACK_MODELS if model != current_model]
    for model in interaction_models:
        if check_rate_limit(model):
            return model
    return "gemini"  # Default fallback

def handle_provider_error(model, error, retry_count=0):
    """Handle provider errors and return fallback model"""
    logging.warning(f"Provider error for {model}: {error}")
    
    if retry_count >= 3:  # Max retry attempts
        return "gemini"  # Final fallback to Gemini
        
    next_model = get_next_available_model(model)
    if next_model == model:  # No other model available
        return "gemini"
        
    logging.info(f"Switching from {model} to {next_model}")
    return next_model

def fetch_response(prompt, character_template, history, model="gemini", image_url=None, retry_count=0):
    MAX_RETRIES = 3
    
    try:
        # Simplify prompt construction
        full_prompt = f"{character_template}\nUser: {prompt}\nResponse:"

        if model == "gemini":
            # Simplify Gemini API call
            model_instance = genai.GenerativeModel("gemini-1.5-flash")
            generation_config = genai.types.GenerationConfig(
                temperature=0.7,
                top_p=0.9,
                top_k=40,
            )
            response = model_instance.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            response_text = response.text.strip()

        elif model == "nvidia":
            # Simplify NVIDIA API call
            completion = nvidia_client.chat.completions.create(
                model="meta-llama/llama-3.2-90b-vision-instruct:free",
                messages=[{"role": "user", "content": full_prompt}],
                temperature=0.5,
                max_tokens=500,
            )
            response_text = completion.choices[0].message.content.strip()

        elif model.startswith("meta-llama"):
            # Handle Llama models
            try:
                completion = openrouter_client.completions.create(
                    model=model,
                    prompt=full_prompt,
                    max_tokens=1000,
                    temperature=0.7,
                    timeout=30  # Add timeout
                )
                response_text = completion.choices[0].text.strip()
            except Exception as e:
                if "524" in str(e) or "Provider returned error" in str(e):
                    if retry_count < MAX_RETRIES:
                        fallback_model = handle_provider_error(model, e, retry_count)
                        return fetch_response(prompt, character_template, history, 
                                           model=fallback_model, 
                                           retry_count=retry_count + 1)
                    else:
                        logging.error(f"Max retries reached for {model}, falling back to Gemini")
                        return fetch_response(prompt, character_template, history, model="gemini")
                raise

        elif model == "liquid/lfm-40b:free":
            # Handle Liquid model
            try:
                completion = openrouter_client.completions.create(
                    model="liquid/lfm-40b:free",
                    prompt=full_prompt,
                    max_tokens=1000,
                    temperature=0.7,
                )
                response_text = completion.choices[0].text.strip()
            except Exception as e:
                if "524" in str(e) or "Provider returned error" in str(e):
                    if retry_count < MAX_RETRIES:
                        fallback_model = handle_provider_error(model, e, retry_count)
                        return fetch_response(prompt, character_template, history, 
                                           model=fallback_model, 
                                           retry_count=retry_count + 1)
                    else:
                        logging.error(f"Max retries reached for {model}, falling back to Gemini")
                        return fetch_response(prompt, character_template, history, model="gemini")
                raise

        else:
            # If model is 'auto', select the best available model
            if model == "auto":
                selected_model = select_best_model()
                return fetch_response(prompt, character_template, history, model=selected_model, image_url=image_url, retry_count=retry_count)
            else:
                response_text = "Error: Model not supported."

        # Clean up the response
        response_text = re.sub(r'^(User|Character):\s*', '', response_text, flags=re.IGNORECASE)
        response_text = response_text.strip()
        
        # Apply expression simplification
        response_text = simplify_expressions(response_text)

        return response_text

    except Exception as e:
        logging.error(f"Error generating response: {e}")
        if retry_count < MAX_RETRIES:
            fallback_model = handle_provider_error(model, e, retry_count)
            return fetch_response(prompt, character_template, history,
                               model=fallback_model,
                               retry_count=retry_count + 1)
        return "I apologize, but I'm having trouble generating a response right now. Please try again in a moment."

# Add new model configuration
AUTO_MODEL_SELECTION = {
    "priority": [
        "gemini",  # Move Gemini to first priority
        "openchat/openchat-7b:free",
        "google/gemma-2-9b-it:free",
        "qwen/qwen-2-7b-instruct:free",
        "meta-llama/llama-3.1-405b-instruct:free",
        "liquid/lfm-40b:free",
        # ...rest of existing models...
    ]
}

def select_best_model():
    """Automatically select the best available model based on reliability and rate limits."""
    for model in AUTO_MODEL_SELECTION["priority"]:
        if model == "gemini":  # Always available
            return model
        if check_rate_limit(model):
            try:
                # Test model availability with a simple prompt
                test_response = openrouter_client.completions.create(
                    model=model,
                    prompt="test",
                    max_tokens=10,
                    timeout=5
                )
                if test_response:
                    return model
            except Exception:
                continue
    return "gemini"  # Default to Gemini if all else fails

# Add response caches
model_response_cache = TTLCache(maxsize=1000, ttl=3600)  # Cache responses for 1 hour
character_response_cache = LRUCache(maxsize=100)  # Cache most recent character responses

# Add request queue
request_queue = asyncio.Queue(maxsize=50)
executor = ThreadPoolExecutor(max_workers=4)

# Add model timeouts
MODEL_TIMEOUTS = {
    "gemini": 10,
    "meta-llama/llama-3.1-405b-instruct:free": 15,
    "liquid/lfm-40b:free": 15,
    "default": 20
}

@lru_cache(maxsize=100)
def get_cached_template(character):
    """Cache character templates"""
    return CHARACTER_TEMPLATES.get(character, CHARACTER_TEMPLATES["furina"])

def clean_response(text):
    """Aggressively clean response text to remove roleplay elements"""
    # Remove asterisk-wrapped actions
    text = re.sub(r'\*[^*]*\*', '', text)
    # Remove expressions in parentheses
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove italic markdown
    text = re.sub(r'_[^_]*_', '', text)
    # Remove bold markdown
    text = re.sub(r'\*\*[^*]*\*\*', '', text)
    # Remove multiple spaces and clean up
    text = ' '.join(text.split())
    return text.strip()

async def fetch_response_async(prompt, character_template, model, timeout=None):
    """Optimized async response generation"""
    cache_key = f"{prompt}:{model}:{hash(character_template)}"
    
    if cache_key in model_response_cache:
        return model_response_cache[cache_key]
        
    try:
        timeout = timeout or MODEL_TIMEOUTS.get(model, MODEL_TIMEOUTS["default"])
        
        if model == "gemini":
            response = await asyncio.get_event_loop().run_in_executor(
                executor,
                lambda: genai.GenerativeModel("gemini-1.5-flash").generate_content(
                    f"Respond directly and concisely without roleplay actions or expressions:\n{prompt}",
                    generation_config=genai.types.GenerationConfig(
                        temperature=0.7,
                        top_p=0.9,
                        top_k=40,
                    )
                )
            )
            response_text = clean_response(response.text.strip())
            
        else:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://openrouter.ai/api/v1/chat/completions",
                    json={
                        "model": model,
                        "messages": [{"role": "system", "content": "Respond directly without roleplay actions or expressions."},
                                   {"role": "user", "content": prompt}],
                        "max_tokens": 1000,
                    },
                    headers={"Authorization": f"Bearer {os.environ.get('OPENROUTER_API_KEY')}"},
                    timeout=timeout
                ) as response:
                    data = await response.json()
                    response_text = clean_response(data["choices"][0]["message"]["content"].strip())

        model_response_cache[cache_key] = response_text
        return response_text

    except Exception as e:
        logging.error(f"Error in fetch_response_async: {e}")
        fallback_model = get_next_available_model(model)
        if fallback_model != model:
            return await fetch_response_async(prompt, character_template, fallback_model)
        return "I apologize, but I'm having trouble responding right now. Please try again."

async def process_request_queue():
    """Process queued requests"""
    while True:
        try:
            request_data = await request_queue.get()
            response = await fetch_response_async(**request_data)
            request_data["future"].set_result(response)
        except Exception as e:
            logging.error(f"Error processing request: {e}")
        finally:
            request_queue.task_done()

# Modify init_queue_processor to be sync
def init_queue_processor():
    """Initialize the request queue processor"""
    if not hasattr(app, '_queue_processor_started'):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        app._queue_processor_task = loop.create_task(process_request_queue())
        app._queue_processor_started = True
        
        # Run the event loop in a separate thread
        def run_event_loop():
            loop.run_forever()
        
        thread = threading.Thread(target=run_event_loop, daemon=True)
        thread.start()
        app._event_loop_thread = thread

@app.before_first_request
def before_first_request():
    """Initialize queue processor before first request"""
    init_queue_processor()

@app.route("/")
def index():
    return render_template("index1.html", characters=list(CHARACTER_TEMPLATES.keys()))

def simplify_expressions(text):
    """Remove or simplify complex expressions and actions"""
    # Remove parenthetical expressions
    text = re.sub(r'\([^)]*\)', '', text)
    # Remove asterisk-wrapped actions
    text = re.sub(r'\*[^*]*\*', '', text)
    # Remove emotive expressions like ~blushes~
    text = re.sub(r'~[^~]*~', '', text)
    # Remove quotation marks and their content
    text = re.sub(r'"[^"]*"', '', text)
    # Remove URLs
    text = re.sub(r'http\S+|www\.\S+', '', text)
    # Remove digits
    text = re.sub(r'\d+', '', text)
    # Remove special characters except punctuation
    text = re.sub(r'[^\w\s.,!?]', '', text)
    # Clean up multiple spaces
    text = ' '.join(text.split())
    return text

# Update generate endpoint to use async_to_sync
@app.route("/generate", methods=["POST"])
def generate():
    async def async_generate():
        try:
            # Get form data from request
            data = request.get_json()
            if not data:
                return jsonify({"error": "No data provided"}), 400

            user_prompt = data.get("user_prompt")
            character_choice = data.get("character_choice")
            selected_model = data.get("model", "auto")
            mentioned_characters = data.get("mentioned_characters", [])

            if not user_prompt:
                return jsonify({"error": "No prompt provided"}), 400

            # Handle multiple character responses if characters are mentioned
            if mentioned_characters:
                responses = []
                
                # Main character response first
                main_char_template = CHARACTER_TEMPLATES[character_choice]["template"]
                main_response = await fetch_response_async(
                    user_prompt,
                    main_char_template,
                    selected_model
                )
                responses.append({
                    "character": character_choice,
                    "response": main_response,
                    "avatar": CHARACTER_AVATARS[character_choice],
                    "is_main": True
                })

                # Generate responses for mentioned characters
                for char in mentioned_characters:
                    if char != character_choice:  # Skip if it's the main character
                        char_template = CHARACTER_TEMPLATES[char]["template"]
                        # Create interaction prompt
                        interaction_prompt = f"User said: {user_prompt}\n{character_choice} responded: {main_response}\nHow would you respond?"
                        char_response = await fetch_response_async(
                            interaction_prompt,
                            char_template,
                            selected_model
                        )
                        responses.append({
                            "character": char,
                            "response": char_response,
                            "avatar": CHARACTER_AVATARS[char],
                            "is_main": False
                        })

                return jsonify({
                    "responses": responses
                })

            else:
                # Single character response
                character_data = get_cached_template(character_choice)
                character_template = character_data["template"]
                response_text = await fetch_response_async(
                    user_prompt,
                    character_template,
                    selected_model
                )

                return jsonify({
                    "response": response_text,
                    "character": character_choice
                })

        except Exception as e:
            print(f"Error in generate: {traceback.format_exc()}")
            return jsonify({"error": str(e)}), 500

    return async_to_sync(async_generate)()

# Update the main run block
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(
        host="0.0.0.0",
        port=port,
        debug=True,
        use_reloader=True
    )