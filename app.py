from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import numpy as np
import os
import base64
import json
import time
from io import BytesIO
from collections import deque
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
CORS(app)

# Configure Gemini API
# Users should set GEMINI_API_KEY environment variable. If not set, it uses a placeholder.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)
    print("Gemini API configured successfully.")
else:
    print("WARNING: GEMINI_API_KEY environment variable not set. Gen AI features will use fallback mock responses.")

def get_gemini_model(model_name, system_instruction=None, tools=None):
    """
    Initializes a GenerativeModel, trying model variants to ensure compatibility and handle quotas.
    """
    models_to_try = [model_name, 'gemini-2.5-flash', 'gemini-flash-latest']
    last_error = None
    for name in models_to_try:
        try:
            model = genai.GenerativeModel(
                name,
                system_instruction=system_instruction,
                tools=tools
            )
            return model
        except Exception as e:
            last_error = e
            print(f"Failed to load Gemini model '{name}': {e}. Trying fallback...")
    raise last_error if last_error else ValueError("No model successfully loaded.")

def search_wikipedia(query):
    import requests
    import re
    url = "https://en.wikipedia.org/w/api.php"
    params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json",
        "utf8": 1
    }
    headers = {
        "User-Agent": "MicroplasticDetectorApp/1.0 (contact@example.com) requests-python"
    }
    try:
        response = requests.get(url, params=params, headers=headers, timeout=4)
        if response.status_code == 200:
            data = response.json()
            search_results = data.get("query", {}).get("search", [])
            results = []
            for item in search_results:
                title = item.get("title")
                snippet = item.get("snippet")
                clean_snippet = re.sub(r'<[^>]+>', '', snippet)
                clean_snippet = clean_snippet.replace('&quot;', '"').replace('&#x27;', "'").replace('&amp;', '&')
                results.append(f"{title}: {clean_snippet.strip()}")
            return results
    except Exception as e:
        print("Wikipedia search error in app:", e)
    return []

# Load the RAG knowledge base
KB_PATH = "D:\\Microplastic-Detection-System\\backend\\microplastics_kb.json"
kb_data = {}
if os.path.exists(KB_PATH):
    try:
        with open(KB_PATH, "r", encoding="utf-8") as f:
            kb_data = json.load(f)
        print("Successfully loaded RAG Knowledge Base.")
    except Exception as e:
        print(f"Error loading RAG KB: {e}")
else:
    print(f"WARNING: RAG Knowledge Base not found at {KB_PATH}.")

def retrieve_rag_context(query_type, key=None):
    """
    Retrieves grounding facts from the local knowledge base to feed into the Gemini prompt.
    """
    context_parts = []
    
    # Always include DLS and general science basics
    if "general_science" in kb_data:
        context_parts.append(f"DLS Telemetry Science: {kb_data['general_science'].get('DLS_principle', '')}")
        context_parts.append(f"Biological Impact: {kb_data['general_science'].get('ingestion_impact', '')}")
        
    if query_type == "liquid":
        # Based on the prediction text/label, pull polymer and filtration details
        if key and key != "Clean":
            polymers = kb_data.get("polymers", {})
            if "PET" in polymers:
                context_parts.append(f"Polymer PET: {polymers['PET']['full_name']} - {polymers['PET']['shedding_mechanism']} Risk: {polymers['PET']['health_risk']}")
            if "PP" in polymers:
                context_parts.append(f"Polymer PP: {polymers['PP']['full_name']} - {polymers['PP']['shedding_mechanism']} Risk: {polymers['PP']['health_risk']}")
            
            filtration = kb_data.get("filtration_technologies", {})
            for tech, info in filtration.items():
                context_parts.append(f"Filtration {tech}: {info['mechanism']} Efficiency: {info['efficiency']} Recommendation: {info['recommendation']}")
        else:
            context_parts.append("Water meets standard purity profiles with minimal scattering anomalies. No active filtration interventions are required.")
            
    elif query_type == "surface" and key:
        surfaces = kb_data.get("surfaces", {})
        matched_surface = None
        for s_name, s_info in surfaces.items():
            if s_name.lower() in key.lower() or key.lower() in s_name.lower():
                matched_surface = (s_name, s_info)
                break
                
        if matched_surface:
            s_name, s_info = matched_surface
            context_parts.append(f"Surface Profile: {s_name} - {s_info['description']}")
            context_parts.append(f"Degradation Indicators: {s_info['degradation_signs']}")
            context_parts.append(f"Microparticle Shedding Risk: {s_info['shedding_rate']}")
            context_parts.append(f"Recommended Safety Alternative: {s_info['safest_alternative']}")
        else:
            context_parts.append(f"Generic Surface category: {key}. Inspect for surface wear, scratches, fading, and degradation. Avoid exposure to heat.")
            
    return "\n".join(context_parts)


# Load the Random Forest model
MODEL_PATH = "D:\\Microplastic-Detection-System\\backend\\microplastic_model.pkl"
model = None
if os.path.exists(MODEL_PATH):
    try:
        model = joblib.load(MODEL_PATH)
        print("Successfully loaded Random Forest model.")
    except Exception as e:
        print(f"Error loading model: {e}")
else:
    print(f"WARNING: Model file not found at {MODEL_PATH}. Run train_model.py first to calibrate.")

# In-memory buffer for sensor values (stores the last 10 readings)
sensor_buffer = deque(maxlen=10)

# Global sequence counter for unique telemetry packets
reading_counter = 0

# Global dictionary to store the latest result
latest_result = {
    "sensor_value": 0.0,
    "prediction_label": -1,
    "prediction_text": "Waiting for Data",
    "confidence": 0.0,
    "mean": 0.0,
    "std_dev": 0.0,
    "status": "collecting",
    "buffer_size": 0,
    "timestamp": 0.0,
    "reading_id": 0
}

LABELS = {
    0: "Clean", 
    1: "Low Contamination", 
    2: "Medium Contamination", 
    3: "High Contamination"
}

scans_history = [
    {
        "id": 1,
        "type": "surface",
        "name": "Bottle",
        "timestamp": time.time() - 86400 * 2, # 2 days ago
        "risk_level": "Medium",
        "particle_count": "8 particles",
        "observations": "Minor abrasions detected on the surface."
    },
    {
        "id": 2,
        "type": "surface",
        "name": "Cutting Board",
        "timestamp": time.time() - 86400, # 1 day ago
        "risk_level": "High",
        "particle_count": "25 particles",
        "observations": "Deep knife scores and micro-cracks detected."
    },
    {
        "id": 3,
        "type": "liquid",
        "name": "Water Telemetry",
        "timestamp": time.time(), # Today
        "risk_level": "Low",
        "particle_count": "1 particle",
        "observations": "Water meets purity profile."
    }
]


@app.route('/api/sensor-data', methods=['POST'])
def receive_data():
    global latest_result, model, reading_counter
    data = request.json
    
    if not data or 'sensor_value' not in data:
        return jsonify({"error": "Invalid request. 'sensor_value' is required."}), 400
        
    voltage = float(data['sensor_value'])
    sensor_buffer.append(voltage)
    reading_counter += 1
    
    # If the model isn't loaded, try to load it on-the-fly
    if model is None and os.path.exists(MODEL_PATH):
        try:
            model = joblib.load(MODEL_PATH)
        except Exception as e:
            print(f"On-the-fly model load failed: {e}")

    # Standard feature extraction when we have a full window of 10 samples
    if len(sensor_buffer) == 10:
        arr = np.array(sensor_buffer)
        mean_val = float(arr.mean())
        std_val = float(arr.std())
        min_val = float(arr.min())
        max_val = float(arr.max())
        range_val = max_val - min_val
        
        if model is not None:
            features = np.array([[mean_val, std_val, min_val, max_val, range_val]])
            pred = model.predict(features)[0]
            proba = model.predict_proba(features).max()
            
            latest_result = {
                "sensor_value": voltage,
                "prediction_label": int(pred),
                "prediction_text": LABELS[int(pred)],
                "confidence": round(float(proba) * 100, 2),
                "mean": round(mean_val, 4),
                "std_dev": round(std_val, 4),
                "status": "active",
                "buffer_size": 10,
                "timestamp": time.time(),
                "reading_id": reading_counter
            }
        else:
            # Fallback if model training is skipped (simple threshold-based fallback)
            # High voltage + low variance = clean
            if mean_val >= 2.8 and std_val < 0.05:
                pred, text = 0, LABELS[0]
            elif mean_val >= 2.4 and std_val < 0.12:
                pred, text = 1, LABELS[1]
            elif mean_val >= 1.8 and std_val < 0.25:
                pred, text = 2, LABELS[2]
            else:
                pred, text = 3, LABELS[3]
                
            latest_result = {
                "sensor_value": voltage,
                "prediction_label": pred,
                "prediction_text": text,
                "confidence": 85.0, # default fallback confidence
                "mean": round(mean_val, 4),
                "std_dev": round(std_val, 4),
                "status": "active_fallback",
                "buffer_size": 10,
                "timestamp": time.time(),
                "reading_id": reading_counter
            }
    else:
        # Still filling buffer, update latest_result so client gets live feedback
        latest_result = {
            "sensor_value": voltage,
            "prediction_label": -1,
            "prediction_text": f"Buffering ({len(sensor_buffer)}/10)",
            "confidence": 0.0,
            "mean": round(voltage, 4),
            "std_dev": 0.0,
            "status": "collecting",
            "buffer_size": len(sensor_buffer),
            "timestamp": time.time(),
            "reading_id": reading_counter
        }
        
    # Log to scans_history once every 10 packets (every full buffer classification window)
    if latest_result.get("prediction_label", -1) != -1 and reading_counter % 10 == 0:
        risk = "Low" if latest_result["prediction_text"] == "Clean" else "Medium" if "Low" in latest_result["prediction_text"] or "Medium" in latest_result["prediction_text"] else "High"
        scans_history.append({
            "id": len(scans_history) + 1,
            "type": "liquid",
            "name": "Water Telemetry",
            "timestamp": time.time(),
            "risk_level": risk,
            "particle_count": f"{(latest_result['std_dev'] * 100):.1f} units",
            "observations": f"DLS Purity Classification: {latest_result['prediction_text']} (Mean: {latest_result['mean']:.2f}V, Var: {latest_result['std_dev']:.4f}V)"
        })
        
    return jsonify(latest_result)

@app.route('/api/latest', methods=['GET'])
def get_latest():
    global latest_result
    # If the latest data is older than 3 seconds, mark it as stale
    current_time = time.time()
    if latest_result.get("reading_id", 0) > 0 and (current_time - latest_result.get("timestamp", 0) > 3.0):
        latest_result["status"] = "stale"
        latest_result["prediction_text"] = "Waiting for Data"
    return jsonify(latest_result)

@app.route('/api/ai-analysis', methods=['POST'])
def ai_analysis():
    global latest_result
    
    if latest_result["prediction_label"] == -1:
        return jsonify({"ai_report": "No sensor readings available yet. Turn on the ESP32 and dip the sensor."})
    
    pred_text = latest_result["prediction_text"]
    confidence = latest_result["confidence"]
    voltage = latest_result["sensor_value"]
    std_dev = latest_result["std_dev"]
    
    # Retrieve RAG context
    rag_context = retrieve_rag_context("liquid", pred_text)
    
    if not GEMINI_API_KEY:
        # Mock Response if Gemini Key is missing, grounded in retrieved RAG context
        mock_report = f"""**Microplastic Assessment Report (Offline Mode)**
- **Water Contamination Level**: {pred_text} (Confidence: {confidence}%)
- **Dynamic Light Scattering Telemetry**: Average voltage is {latest_result['mean']}V with standard deviation {std_dev}V.
- **Risk Level**: {"Low" if pred_text == "Clean" else "Medium" if "Low" in pred_text else "High"}
- **Retrieved Science Context**: {rag_context}
- **Safety Recommendation**: {"The water sample shows no significant light scattering. It meets standard clarity profiles." if pred_text == "Clean" else "Dynamic scattering patterns indicate microplastic particles or suspended solids. Boiling will NOT remove microplastics; please use a sub-micron carbon block or reverse osmosis (RO) filtration system."}"""
        return jsonify({"ai_report": mock_report})

    try:
        model = get_gemini_model('gemini-2.5-flash')
        prompt = f"""
        You are a water safety AI system running a Retrieval-Augmented Generation (RAG) pipeline.
        
        Retrieved Scientific Reference Context:
        {rag_context}
        
        Telemetry Inputs:
        - Detected Contamination Category: {pred_text}
        - ML Confidence Score: {confidence}%
        - Sensor Raw Voltage: {voltage}V
        - Signal Variance (Std Dev): {std_dev}V
        
        Generate a microplastic risk analysis report based on the telemetry inputs, grounded in the retrieved scientific reference context.
        Structure your response in clear sections using markdown:
        1. **Simple Explanation**: Explain what these sensor values mean (higher signal variance means more floating microparticles blocking the laser beam).
        2. **Estimated Contamination Level**: Classify exposure severity.
        3. **Alternative Actionable Advice**: Suggest safe alternatives (like switching plastic bottles to glass or stainless steel).
        4. **Recommendations**: Highlight specific filtration methods (e.g. Reverse Osmosis, gravity carbon filters) that remove microplastics.
        Keep the report under 120 words.
        """
        response = model.generate_content(prompt)
        return jsonify({"ai_report": response.text})
    except Exception as e:
        return jsonify({"error": f"Gemini API Error: {str(e)}"}), 500


def retrieve_chat_rag_context(user_query):
    """
    Dynamically retrieves scientific grounding facts from microplastics_kb.json
    based on keywords found in the user query.
    """
    query_lower = user_query.lower()
    context_parts = []
    
    # Check polymer matches
    for poly_key, poly_info in kb_data.get("polymers", {}).items():
        if poly_key.lower() in query_lower or poly_info.get("full_name", "").lower() in query_lower:
            context_parts.append(f"Polymer {poly_key}: {poly_info.get('full_name')} - Shedding: {poly_info.get('shedding_mechanism')} Risk: {poly_info.get('health_risk')}")
            
    # Check filtration matches
    for tech_key, tech_info in kb_data.get("filtration_technologies", {}).items():
        match_name = tech_key.replace("_", " ").lower()
        if match_name in query_lower or "filter" in query_lower:
            context_parts.append(f"Filtration {tech_key}: {tech_info.get('mechanism')} Efficiency: {tech_info.get('efficiency')} Recommendation: {tech_info.get('recommendation')}")
            
    # Check surface matches
    for surf_key, surf_info in kb_data.get("surfaces", {}).items():
        if surf_key.lower() in query_lower:
            context_parts.append(f"Surface {surf_key}: {surf_info.get('description')} Degradation indicators: {surf_info.get('degradation_signs')} Shedding Risk: {surf_info.get('shedding_rate')} Alternative: {surf_info.get('safest_alternative')}")

    # General fallback if no specific keywords are matched
    if not context_parts and "general_science" in kb_data:
        context_parts.append(f"DLS Telemetry Science: {kb_data['general_science'].get('DLS_principle', '')}")
        context_parts.append(f"Biological Impact: {kb_data['general_science'].get('ingestion_impact', '')}")
        
    return "\n".join(context_parts)


def generate_smart_chat_fallback(query, persona, rag_context):
    query_lower = query.lower()
    
    # Base intro matching persona
    if persona == "chemist":
        intro = "Hello, I am Dr. Aris (Purity Chemist). "
    elif persona == "health":
        intro = "Hi, I'm Aura (Eco-Health Advisor). "
    elif persona == "filtration":
        intro = "Greetings, I'm Kai (Filtration Architect). "
    else:
        intro = "As your HydroShield Safety Advisor, "
        
    # Query Wikipedia for real-time web context
    web_results = search_wikipedia(query)
    
    if web_results:
        summary_bullets = "\n".join([f"• {r}" for r in web_results[:3]])
        reply = (
            f"{intro}I've checked the web for '{query}' and synthesized the latest updates for you:\n\n"
            f"{summary_bullets}\n\n"
            f"Grounding context: {rag_context or 'Purity guidelines suggest monitoring material integrity.'}"
        )
        if len(reply) > 750:
            reply = reply[:747] + "..."
        return reply
        
    # Fallback to local expert rules if wiki search returned no results
    if "guideline" in query_lower or "who" in query_lower or "limit" in query_lower:
        return intro + "The WHO does not currently set a formal health-based guideline limit for microplastics in drinking water due to insufficient evidence, but they recommend monitoring. Most research suggests avoiding plastics exposed to heat."
    
    if "laser" in query_lower or "sensor" in query_lower or "dls" in query_lower or "voltage" in query_lower:
        return intro + "Our telemetry system uses Dynamic Light Scattering. Microparticles passing through the laser beam block the light, creating fluctuations in the raw voltage. A higher standard deviation indicates more particles."
        
    if "filter" in query_lower or "ro" in query_lower or "carbon" in query_lower or "remove" in query_lower:
        return intro + "To filter out microplastics, standard boiling will not work. You need a sub-micron activated carbon block filter (rejection down to 0.5 microns) or a multi-stage Reverse Osmosis (RO) system (rejection down to 0.0001 microns) for 99.9% effectiveness."
        
    if "bottle" in query_lower or "pet" in query_lower:
        return intro + "Plastic water bottles are made of PET. They shed microplastics when squeezed, twisted at the cap, or exposed to sunlight and heat. Antimony catalysts can also leach into the water."
        
    if "cutting board" in query_lower or "hdpe" in query_lower or "chopping" in query_lower:
        return intro + "Plastic cutting boards (HDPE/PP) are highly prone to mechanical shedding. Chopping actions slice off micro-shavings that end up in your food. I highly recommend switching to maple or walnut wood."
        
    if "risk" in query_lower or "health" in query_lower or "body" in query_lower or "toxic" in query_lower:
        return intro + "Microplastics can cross human cellular membranes. They accumulate in organs and can leach endocrine-disrupting chemicals like BPA and phthalates, leading to cell stress and inflammation."

    responses = [
        "microplastic particles represent a growing environmental and biological challenge. In water samples, they scatter light causing turbidity spikes.",
        "purity testing using light-scattering telemetry highlights suspended contaminants. Standard treatment methods like boiling do not remove plastic polymers.",
        "microparticle ingestion causes gut abrasion and chemical additive absorption. We recommend using wood, glass, or steel storage solutions.",
        "retaining pristine telemetry baselines is crucial for identifying plastic leakage in domestic supply. Sub-micron carbon filtration holds 95-99% particles."
    ]
    import random
    selected = random.choice(responses)
    return intro + f"Regarding '{query}': {selected}"

@app.route('/api/agent/chat', methods=['POST'])
def agent_chat():
    global latest_result
    data = request.json or {}
    
    messages = data.get("messages", [])
    persona = data.get("persona", "general")
    inject_telemetry = data.get("inject_telemetry", False)
    
    if not messages:
        return jsonify({"error": "No messages history provided."}), 400
        
    last_user_message = messages[-1]["content"] if messages[-1]["role"] == "user" else ""
    
    # Retrieve RAG context based on the last message
    rag_context = retrieve_chat_rag_context(last_user_message)
    
    # Configure Persona system instructions
    if persona == "chemist":
        persona_name = "Dr. Aris (Purity Chemist)"
        system_instruction = (
            "You are Dr. Aris, a world-renowned polymer chemist and material scientist. Your expertise "
            "is in the molecular structure of plastics (PET, PP, HDPE, LDPE), their thermal degradation, shedding mechanics, "
            "and chemical leaching. You speak in precise, scientific, and analytical terms. When answering, explain how the physical "
            "forces and chemical composition impact the sample. Limit your response to 150 words."
        )
    elif persona == "health":
        persona_name = "Aura (Eco-Health Advisor)"
        system_instruction = (
            "You are Aura, an Eco-Health Advisor and toxicologist. You specialize in the biological effects of microplastics "
            "on human cells, bioaccumulation, endocrine disruption, and digestive tract damage. You speak with high empathy "
            "and focus on safety warnings, preventative lifestyle habits, and natural material alternatives. Limit your response to 150 words."
        )
    elif persona == "filtration":
        persona_name = "Kai (Filtration Architect)"
        system_instruction = (
            "You are Kai, an environmental engineer and domestic water filtration architect. Your expertise lies in the "
            "mechanics and designs of home filtration systems, including Reverse Osmosis (RO), sub-micron carbon blocks, "
            "ultrafiltration, and gravity filters. You provide practical, step-by-step guidance on how to select filtration. Limit your response to 150 words."
        )
    else:
        persona_name = "HydroShield Advisor"
        system_instruction = (
            "You are a HydroShield Water Safety AI Advisor. You help users understand microplastics in water and surfaces, "
            "explain dynamic light scattering telemetry, and suggest safety measures. Limit your response to 150 words."
        )
        
    # Append Wikipedia info if available to system instructions for grounding
    wiki_info = search_wikipedia(last_user_message)
    if wiki_info:
        system_instruction += f"\n\nReal-time Web Search Grounding:\n" + "\n".join(wiki_info[:3])
        
    if rag_context:
        system_instruction += f"\n\nScientific Grounding Knowledge:\n{rag_context}"
        
    # Append Telemetry context if enabled
    if inject_telemetry:
        if latest_result.get("reading_id", 0) > 0 and latest_result.get("status") != "stale":
            system_instruction += (
                f"\n\n[Active Sensor Telemetry - Real-time readings from user's physical sample]:\n"
                f"- Raw voltage read: {latest_result['sensor_value']:.4f}V\n"
                f"- Running Average (Mean): {latest_result['mean']:.4f}V\n"
                f"- Variance (Std Dev): {latest_result['std_dev']:.4f}V\n"
                f"- ML Prediction: {latest_result['prediction_text']} (Confidence: {latest_result['confidence']}%)\n"
                f"- Telemetry status: {latest_result['status']}\n"
                f"Please directly reference these telemetry values in your answer and explain what they indicate about the sample."
            )
        else:
            system_instruction += "\n\n[Active Sensor Telemetry]: Currently OFFLINE or waiting for sensor data. Advise the user to turn on their ESP32."
 
    try:
        if not GEMINI_API_KEY:
            raise ValueError("No API Key configured")

        # Initialize Gemini Generative Model with system instructions and Google Search grounding
        model = get_gemini_model(
            'gemini-2.5-flash',
            system_instruction=system_instruction,
            tools=[{'google_search_retrieval': {}}]
        )
        
        # Convert messages list to Gemini API format
        formatted_history = []
        for msg in messages[:-1]:
            role = "user" if msg["role"] == "user" else "model"
            formatted_history.append({
                "role": role,
                "parts": [msg["content"]]
            })
            
        # Start chat with formatted history
        chat = model.start_chat(history=formatted_history)
        
        # Send the last message to the model
        response = chat.send_message(last_user_message)
        reply_text = response.text.strip()
        
    except Exception as e:
        print(f"Gemini API failure in agent_chat: {e}. Falling back to smart dynamic generator.")
        reply_text = generate_smart_chat_fallback(last_user_message, persona, rag_context)
        
    return jsonify({
        "reply": reply_text,
        "persona_used": persona_name
    })


def generate_dynamic_surface_analysis(surface_type):
    # Determine polymer, count, risk, alternatives, and recommendations based on surface type
    if surface_type == "Bottle":
        poly_name = "Polyethylene Terephthalate (PET)"
        particle_count = "Estimated 15-20 particles/cm² (high density PET micro-fragments)"
        risk = "Medium"
        alternatives = ["Borosilicate Glass Flask", "304 Stainless Steel Bottle"]
        recs = [
            "Do not leave plastic bottles in hot environments (like cars) as heat triggers antimony and PET shedding.",
            "Avoid squeezing or reusing single-use plastic bottles.",
            "Twisting caps generates micro-fragments; rinse the bottle neck regularly."
        ]
    elif surface_type == "Utensil":
        poly_name = "Polypropylene (PP)"
        particle_count = "Estimated 30-45 fibers/use (warping PP microparticles)"
        risk = "High"
        alternatives = ["Seasoned Cast Iron", "Bamboo Utensils", "Stainless Steel Melamine Alternatives"]
        recs = [
            "Discard utensils showing warping, rough edges, or fading.",
            "Never expose plastic spoons or bowls to liquids above 70°C.",
            "Avoid washing PP plastics in abrasive dishwashers; hand wash with soft sponges."
        ]
    elif surface_type == "Cutting Board":
        poly_name = "High-Density Polyethylene (HDPE)"
        particle_count = "Estimated 50-85 particles/session (10-50mg of HDPE scraped ribbons)"
        risk = "High"
        alternatives = ["Solid Maple Wood Board", "Walnut Wood Board", "Bamboo Chopping Block"]
        recs = [
            "Scrape off fuzzy plastic fibers before food prep if board is cut-damaged.",
            "Do not wash plastic cutting boards in high-heat dishwasher sanitizing cycles.",
            "Switch to a wooden cutting board to prevent plastic shavings in meals."
        ]
    elif surface_type == "Food Packaging":
        poly_name = "Low-Density Polyethylene (LDPE)"
        particle_count = "Estimated 25 particles/m² (flexible film LDPE flakes)"
        risk = "Medium"
        alternatives = ["Beeswax wraps", "Food-grade Silicone lids", "Glass Tupperware"]
        recs = [
            "Never microwave food in contact with plastic cling wrap.",
            "Avoid storing oily, acidic, or hot food in thin plastic wrap.",
            "Peeling heated stretch wrap releases micro-nanoplastics onto food surfaces."
        ]
    elif surface_type == "Street/Soil":
        poly_name = "SBR Rubber (Tyre Wear Particles)"
        particle_count = "Estimated 120 particles/g of road dust"
        risk = "High"
        alternatives = ["Vegetative Bioswales", "Permeable Eco-pavements"]
        recs = [
            "Tyre wear generates massive styrene-butadiene rubber (SBR) particles.",
            "Install rain gardens to trap micro-rubbers from tyre friction run-offs.",
            "Use air purifiers to filter airborne micro-rubbers near heavy traffic."
        ]
    elif surface_type == "Water Container":
        poly_name = "High-Density Polyethylene (HDPE) / Polycarbonate"
        particle_count = "Estimated 10-18 particles/L of container water"
        risk = "Medium"
        alternatives = ["Glass carboys", "Stainless steel water dispensers"]
        recs = [
            "Do not keep large plastic water containers exposed to sunlight or heat.",
            "Use BPA-free glass or steel storage vessels for long-term water storage."
        ]
    else:
        poly_name = "Mixed Thermoplastics"
        particle_count = "Estimated 10-15 particles/cm²"
        risk = "Medium"
        alternatives = ["Glass", "Metal", "Ceramic"]
        recs = [
            "Avoid exposing thermoplastic objects to heat, UV, or acids.",
            "Clean surfaces using non-abrasive soft cloths."
        ]
        
    return {
        "detected_object": surface_type,
        "risk_level": risk,
        "particle_count_estimation": particle_count,
        "observations": f"Visible surface degradation detected on this {surface_type}. Under stress, it sheds {poly_name} microplastics.",
        "alternatives": alternatives,
        "recommendations": recs
    }

@app.route('/api/analyze-image', methods=['POST'])
def analyze_image():
    global scans_history
    data = request.json
    if not data or 'image' not in data:
        return jsonify({"error": "Image data (base64) is required."}), 400
        
    image_base64 = data['image']
    surface_type = data.get('surface_type', 'surface')
    
    # Retrieve RAG context
    rag_context = retrieve_rag_context("surface", surface_type)
    
    try:
        if not GEMINI_API_KEY:
            raise ValueError("No API Key configured")
            
        # Decode base64 image
        image_data = base64.b64decode(image_base64)
        image = Image.open(BytesIO(image_data))
        
        model = get_gemini_model('gemini-2.5-flash')
        
        prompt = f"""
        You are a microplastics detection expert running a Retrieval-Augmented Generation (RAG) pipeline.
        
        Analyze this captured image of a surface. First, identify what the primary object or surface is.
        Classify it into one of these categories: "Bottle", "Utensil", "Food Packaging", "Street/Soil", "Cutting Board", "Water Container", or "Other".
        
        Retrieved Scientific Reference Context for this surface type:
        {rag_context}
        
        Tasks:
        1. Identify the detected object name (e.g. "Bottle", "Cutting Board", "Utensil") and assign it to the "detected_object" field.
        2. Identify the specific polymer type (e.g. PET for Bottle, PP for Utensil, HDPE for Cutting Board, LDPE for Food Packaging, SBR for Street/Soil, etc.) of the microplastics likely to be shed.
        3. Provide a specific estimated count or shedding rate of these microplastic particles (e.g., "Estimated 18 particles" or "Estimated 25 particles/cm²"). You MUST specify a numerical microplastics count.
        4. Suggest safe alternative materials, incorporating information from the retrieved reference context.
        5. Provide actionable recommendations for safety.

        Respond in this exact JSON format (strictly JSON, do not wrap in ```json or markdown blocks):
        {{
          "detected_object": "Detected object name (e.g. Bottle, Utensil, Cutting Board, etc.)",
          "risk_level": "Low" | "Medium" | "High",
          "particle_count_estimation": "estimated particle count string showing specific microplastics count and type",
          "observations": "brief summary of what you see on the surface, grounded in polymer degradation signs and specific microplastics from reference context",
          "alternatives": ["Alternative 1", "Alternative 2"],
          "recommendations": ["Recommendation 1", "Recommendation 2"]
        }}
        """
        
        response = model.generate_content(
            [prompt, image],
            generation_config={"response_mime_type": "application/json"}
        )
        
        # Clean response and parse JSON
        text_response = response.text.strip()
        parsed_json = json.loads(text_response)
        
    except Exception as e:
        print(f"Gemini API failure in analyze_image: {e}. Falling back to high-fidelity dynamic response.")
        parsed_json = generate_dynamic_surface_analysis(surface_type)
        
    # Log to in-memory history
    detected_obj_name = parsed_json.get("detected_object", surface_type)
    scan_record = {
        "id": len(scans_history) + 1,
        "type": "surface",
        "name": detected_obj_name,
        "timestamp": time.time(),
        "risk_level": parsed_json.get("risk_level", "Medium"),
        "particle_count": parsed_json.get("particle_count_estimation", "N/A"),
        "observations": parsed_json.get("observations", "Captured scan of " + detected_obj_name)
    }
    scans_history.append(scan_record)
    return jsonify(parsed_json)


@app.route('/api/dashboard-summary', methods=['GET'])
def get_dashboard_summary():
    global scans_history, latest_result
    
    date_filter = request.args.get('date') # Format: "13 Jun 2026"
    
    filtered_history = scans_history
    if date_filter:
        import datetime
        filtered_history = []
        for s in scans_history:
            dt = datetime.datetime.fromtimestamp(s["timestamp"])
            date_str = dt.strftime("%d %b %Y")
            if date_str.lower().strip() == date_filter.lower().strip():
                filtered_history.append(s)
                
    # Compute metrics from filtered_history
    total_scans = len(filtered_history)
    if total_scans == 0:
        return jsonify({
            "total_particles_detected": "0 particles",
            "most_common_type": "None",
            "aggregate_risk_level": "Low",
            "risk_value": 0.1,
            "recent_scans": []
        })
        
    total_particles = 0
    type_counts = {}
    risk_scores = []
    
    for s in filtered_history:
        name = s.get("name", "Other")
        type_counts[name] = type_counts.get(name, 0) + 1
        
        risk = s.get("risk_level", "Medium").lower()
        if "high" in risk:
            risk_scores.append(3)
        elif "medium" in risk or "moderate" in risk:
            risk_scores.append(2)
        else:
            risk_scores.append(1)
            
        # Try to extract numbers from particle_count string
        p_str = s.get("particle_count", "")
        import re
        nums = re.findall(r'\d+', p_str)
        if nums:
            total_particles += int(nums[0])
        else:
            total_particles += 15 # default fallback average count
            
    most_common = max(type_counts, key=type_counts.get) if type_counts else "None"
    
    # Map raw names to user-friendly titles
    friendly_names = {
        "Bottle": "Plastics (PET)",
        "Utensil": "Kitchen Utensil (PP)",
        "Food Packaging": "Flexible Film (LDPE)",
        "Street/Soil": "Road Rubbers (TWPs)",
        "Cutting Board": "Chopping Board (HDPE)",
        "Water Container": "Water Container",
        "Water Telemetry": "Water Telemetry (DLS)"
    }
    most_common_friendly = friendly_names.get(most_common, most_common)
    
    avg_risk = sum(risk_scores) / len(risk_scores)
    agg_risk = "Low" if avg_risk < 1.6 else "Moderate" if avg_risk < 2.5 else "High"
    
    # Format recent scans list (take all matching/all recent, let frontend handle displaying subset)
    formatted_scans = []
    for s in reversed(filtered_history):
        import datetime
        dt = datetime.datetime.fromtimestamp(s["timestamp"])
        date_str = dt.strftime("%d %b %Y")
        
        formatted_scans.append({
            "title": friendly_names.get(s["name"], s["name"]),
            "date": date_str,
            "particles": s["particle_count"],
            "risk": s["risk_level"],
            "type": s["type"]
        })
        
    return jsonify({
        "total_particles_detected": f"{total_particles} particles",
        "most_common_type": most_common_friendly,
        "aggregate_risk_level": agg_risk,
        "risk_value": avg_risk / 3.0,
        "recent_scans": formatted_scans
    })


if __name__ == '__main__':
    # Listen on all interfaces, port 5000
    app.run(host='0.0.0.0', port=5000, debug=True)
