import os
import json
import re
import asyncio
from google import genai
from dotenv import load_dotenv
load_dotenv()

def create_prompt(hotel_name: str, location: str) -> str:
    not_found_json = '{"hotel": null, "rooms": null, "amenities": null, "policies": null, "special_features": null, "awards": null, "error": "Hotel doesn\'t exist"}'

    return f"""You are a hotel data researcher. Gather comprehensive information about:

**Hotel Name:** {hotel_name}
**Location:** {location}

First verify the hotel exists at this location. If not, return: {not_found_json}

**OUTPUT RULES:**
- Return EXACTLY ONE JSON code block (no text before/after).
- The response must be a single, valid JSON object (no top-level arrays).
- Escape newlines within JSON string values as \\n.
- Use null for any missing data fields. Do not omit fields.
- All data types must strictly match the schema below.

**REQUIRED DATA:**

1.  **Hotel Main Info:**
    *   official_name, star_rating (1-5, integer), description (2-3 paragraphs of prose, no bullet points), address, year_opened (YYYY-MM-DD), last_renovation (YYYY-MM-DD), total_rooms, number_of_floors, additional_info (2-3 paragraphs of prose, no bullet points)

2.  **Room Types (array):**
    *   room_type, bed_configuration, room_size_sqft (integer only), max_occupancy, price_range_min/max (decimal format XX.XX), description

3.  **Amenities by Category (array):**
    *   Categories must be one of: "Room Amenities", "Property Amenities", "Dining", "Services".
    *   Each object in the array should have: category, description, details (as a JSON object where applicable)

4.  **Policies (object):**
    *   check_in_time (HH:MM:SS), check_out_time (HH:MM:SS), min_age_requirement (integer), pet_policy, smoking_policy, children_policy, extra_person_policy, cancellation_policy

5.  **Special Features & Awards:**
    *   Both must be arrays of strings.

**SCHEMA:**
```json
{{
  "hotel": {{
    "official_name": "string (max 255)",
    "star_rating": 4,
    "description": "2-3 paragraphs prose, no bullets",
    "address": "string",
    "year_opened": "2015-01-01",
    "last_renovation": "2021-01-01",
    "total_rooms": 261,
    "number_of_floors": 5,
    "additional_info": "2-3 paragraphs prose, no bullets"
  }},
  "rooms": [
    {{
      "room_type": "string (max 100)",
      "bed_configuration": "string (max 100)",
      "room_size_sqft": 280,
      "max_occupancy": 2,
      "price_range_min": 189.00,
      "price_range_max": 279.00,
      "description": "string"
    }}
  ],
  "amenities": [
    {{
      "category": "string (max 50)",
      "description": "string",
      "details": {{}}
    }}
  ],
  "policies": {{
    "check_in_time": "15:00:00",
    "check_out_time": "11:00:00",
    "min_age_requirement": 21,
    "pet_policy": "string",
    "smoking_policy": "string",
    "children_policy": "string",
    "extra_person_policy": "string",
    "cancellation_policy": "string"
  }},
  "special_features": ["string"],
  "awards": ["string"]
}}
"""

async def get_hotel_data_async(hotel_name: str, location: str) -> dict:
    try:
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable not set.")
        
        client = genai.Client(api_key=api_key)
        model = "gemini-live-2.5-flash-preview"
        
        prompt = create_prompt(hotel_name, location)
        
        # Using Google Search for grounding
        tools = [{'google_search': {}}]
        config = {"response_modalities": ["TEXT"], "tools": tools}

        async with client.aio.live.connect(model=model, config=config) as session:
            await session.send_client_content(turns={"parts": [{"text": prompt}]})
            
            # Collect all response chunks
            full_response = ""
            async for chunk in session.receive():
                if chunk.server_content:
                    if chunk.text is not None:
                        full_response += chunk.text
                    
                    # Handle executable code if generated (for search)
                    model_turn = chunk.server_content.model_turn
                    if model_turn:
                        for part in model_turn.parts:
                            if part.executable_code is not None:
                                # Code execution happens automatically for search
                                pass
                            if part.code_execution_result is not None:
                                # Search results are incorporated automatically
                                pass
                    
                    # Check if turn is complete
                    if chunk.server_content.turn_complete:
                        break

            # Clean the response to ensure it's valid JSON
            if not full_response:
                return {"error": "No response received from API."}
                
            cleaned_text = full_response.strip()
            
            if cleaned_text.startswith('```'):
                cleaned_text = re.sub(r'^```(?:json)?\s*', '', cleaned_text, flags=re.IGNORECASE)
            
            # Extract JSON from markdown code blocks (similar to Perplexity handling)
            if '```' in cleaned_text:
                try:
                 
                    # Match pattern: ```json or ``` followed by content and closing ```
                    pattern = r"```(json)?\s*([\s\S]*?)```"
                    matches = re.findall(pattern, cleaned_text, flags=re.IGNORECASE)
                    
                    if matches:
                        # Filter out empty blocks and prioritize JSON blocks
                        valid_blocks = []
                        for lang, content in matches:
                            content = content.strip()
                            if content and len(content) > 10:  # Filter out empty or tiny blocks
                                # Prioritize JSON blocks
                                priority = 2 if lang and lang.lower() == 'json' else 1
                                valid_blocks.append((priority, len(content), content))
                        
                        if valid_blocks:
                            # Sort by priority (JSON first), then by length
                            valid_blocks.sort(key=lambda x: (x[0], x[1]), reverse=True)
                            cleaned_text = valid_blocks[0][2].strip()
                except Exception:
                    # If regex fails, continue to fallback
                    pass
            
            # extract JSON object between first { and last }
            # handles cases where there are no closing backticks or incomplete code blocks
            if '{' in cleaned_text and '}' in cleaned_text:
                first_brace = cleaned_text.find('{')
                last_brace = cleaned_text.rfind('}')
                if last_brace > first_brace:
                    # Extract the JSON portion
                    potential_json = cleaned_text[first_brace:last_brace + 1]
                    # Check if it starts with valid JSON structure
                    if potential_json.strip().startswith('{'):
                        cleaned_text = potential_json
            
            cleaned_text = cleaned_text.strip()
            
            # Remove any remaining markdown fences if they weren't caught
            if cleaned_text.startswith('```json'):
                cleaned_text = cleaned_text[7:]
            if cleaned_text.startswith('```'):
                cleaned_text = cleaned_text[3:]
            if cleaned_text.endswith('```'):
                cleaned_text = cleaned_text[:-3]
            cleaned_text = cleaned_text.strip()
            
            # ensure we have a JSON object
            if not cleaned_text.startswith('{'):
                raise ValueError("Extracted text does not start with '{' - invalid JSON format")

            # Try to parse JSON
            try:
                return json.loads(cleaned_text)
            except json.JSONDecodeError as json_err:
                # Try using the validation/fix function from utils first
                #  handles trailing commas, unquoted keys, and other issues
                try:
                    from src.utils import validate_and_fix_json
                    fixed_json = validate_and_fix_json(cleaned_text)
                    if fixed_json:
                        return fixed_json
                except Exception as fix_err:
                    pass
                
                try:
                    # Pattern: find unescaped quotes inside string values
                    # Look for pattern: "text "word" more text" where we need to escape the inner quotes
                    # This regex finds quotes that are inside a string value (between :" and ", or "})
                    fixed_text = cleaned_text
                    # Escape quotes that appear after :" and before ", or before ",
                    pattern = r'(:\s*")((?:(?:[^"\\]|\\.)*?"\s*[,\]}])*?)([^"\\]|(?<!\\))"([^",\]}])'
                    
                    def escape_quotes_in_strings(text):
                        """Escape unescaped quotes inside JSON string values"""
                        result = []
                        i = 0
                        in_string = False
                        after_colon = False
                        
                        while i < len(text):
                            char = text[i]
                            
                            # Check if we're entering a string value (after ":)
                            if i > 1 and text[i-2:i] == ':"':
                                in_string = True
                                after_colon = True
                                result.append(char)
                            elif char == '"' and in_string and i > 0 and text[i-1] != '\\':
                                # Check if this is a closing quote
                                j = i + 1
                                while j < len(text) and text[j] in ' \t\n':
                                    j += 1
                                if j < len(text) and text[j] in ',}]':
                                    # Valid closing quote
                                    in_string = False
                                    result.append(char)
                                else:
                                    # Unescaped quote inside string - escape it
                                    result.append('\\"')
                            elif char == '\\' and in_string:
                                # Handle escape sequences
                                result.append(char)
                                if i + 1 < len(text):
                                    result.append(text[i + 1])
                                    i += 1
                            else:
                                result.append(char)
                            i += 1
                        
                        return ''.join(result)
                    
                    fixed_text = escape_quotes_in_strings(cleaned_text)
                    
                    # Try parsing the fixed text
                    try:
                        return json.loads(fixed_text)
                    except json.JSONDecodeError:
                        pass  # Continue to raise original error
                        
                except Exception:
                    pass  # Continue to raise original error
                
                # If all fixes failed, raise the original error
                raise json_err

    except json.JSONDecodeError as e:
        print(f"Error: Failed to decode JSON from the model's response: {e}")
        print(f"Error location: line {e.lineno if hasattr(e, 'lineno') else 'unknown'}, column {e.colno if hasattr(e, 'colno') else 'unknown'}")
        if 'full_response' in locals():
            print("Raw response:", full_response[:500] + "..." if len(full_response) > 500 else full_response)
        if 'cleaned_text' in locals():
            # Show context around the error location
            error_pos = e.pos if hasattr(e, 'pos') else None
            if error_pos and error_pos < len(cleaned_text):
                start = max(0, error_pos - 100)
                end = min(len(cleaned_text), error_pos + 100)
                print(f"Context around error (pos {error_pos}): ...{cleaned_text[start:end]}...")
        return {"error": "Invalid JSON response from AI."}
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return {"error": str(e)}

def get_hotel_data(hotel_name: str, location: str) -> dict:
    try:
        try:
            loop = asyncio.get_running_loop()
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                future = executor.submit(asyncio.run, get_hotel_data_async(hotel_name, location))
                return future.result()
        except RuntimeError:
            try:
                loop = asyncio.get_event_loop()
                if loop.is_closed():
                    # Loop is closed, create a new one
                    return asyncio.run(get_hotel_data_async(hotel_name, location))
                else:
                    # Use existing loop
                    return loop.run_until_complete(get_hotel_data_async(hotel_name, location))
            except RuntimeError:
                # No event loop at all, create a new one
                return asyncio.run(get_hotel_data_async(hotel_name, location))
    except Exception as e:
        return asyncio.run(get_hotel_data_async(hotel_name, location))
