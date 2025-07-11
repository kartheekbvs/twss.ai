from flask import Flask, request, jsonify
from flask_cors import CORS
import base64
import io
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for production
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import os
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Configure CORS for production
CORS(app, origins=[
    "https://your-netlify-site.netlify.app",  # Update with your Netlify URL
    "https://*.netlify.app",
    "http://localhost:3000",  # For local development
    "http://localhost:5000"   # For local development
])

# Global variables
generator = None
model_loaded = False

def load_model():
    """Load the AI model with error handling"""
    global generator, model_loaded
    
    try:
        logger.info("Loading AI text generation model...")
        
        # Try to import transformers
        from transformers import pipeline
        
        # Load the model
        generator = pipeline(
            'text-generation', 
            model='distilgpt2',
            device=-1,  # Use CPU in production to avoid GPU memory issues
            torch_dtype='auto'
        )
        
        model_loaded = True
        logger.info("Model loaded successfully.")
        
    except ImportError as e:
        logger.error(f"Transformers not available: {e}")
        model_loaded = False
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        model_loaded = False

def generate_sketch_base64(prompt, text):
    """Generate sketch and return as base64 string"""
    try:
        # Use hashes to seed the randomness
        prompt_hash = hash(prompt) % 1000
        text_hash = hash(text) % 1000

        # Create a matplotlib figure
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.set_aspect('equal')
        ax.axis('off')

        # Use the hashes to define colors and shapes
        color_seed = (prompt_hash + text_hash) % 256

        # Simple color scheme based on the combined hash
        bg_r = (color_seed * 1.5) % 1
        bg_g = (color_seed * 0.5) % 1
        bg_b = (color_seed * 2.5) % 1

        # Fill the background
        ax.set_facecolor((bg_r, bg_g, bg_b))

        # Add abstract shapes
        for i in range(12):  # Slightly more shapes for production
            x = (i * 0.1) + (prompt_hash % 100) / 500
            y = (i * 0.08) + (text_hash % 100) / 500
            size = (i % 5) * 0.05 + 0.1

            # Determine shape and color
            shape_type = (prompt_hash + i) % 4  # 4 shape types

            # Dynamic color for shapes
            shape_r = (text_hash * i * 0.01) % 1
            shape_g = (prompt_hash * i * 0.01) % 1
            shape_b = (color_seed * i * 0.01) % 1

            shape_color = (shape_r, shape_g, shape_b, 0.7)

            if shape_type == 0:
                # Rectangle
                rect = patches.Rectangle((x, y), size, size * 0.8, color=shape_color)
                ax.add_patch(rect)
            elif shape_type == 1:
                # Circle
                circ = patches.Circle((x + size/2, y + size/2), size * 0.5, color=shape_color)
                ax.add_patch(circ)
            elif shape_type == 2:
                # Triangle
                triangle = patches.Polygon([[x, y], [x + size, y], [x + size/2, y + size*1.5]], color=shape_color)
                ax.add_patch(triangle)
            else:
                # Ellipse
                ellipse = patches.Ellipse((x + size/2, y + size/2), size * 0.8, size * 0.4, 
                                        angle=i*30, color=shape_color)
                ax.add_patch(ellipse)

        # Save to base64 string
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight', dpi=100, 
                   facecolor='white', edgecolor='none')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        plt.close(fig)  # Important: close the figure to free memory
        
        return image_base64
        
    except Exception as e:
        logger.error(f"Error generating sketch: {e}")
        raise

def generate_text_content(prompt):
    """Generate text using the AI model"""
    global generator, model_loaded
    
    if not model_loaded or generator is None:
        logger.warning("Model not loaded, using fallback text generation")
        return generate_text_fallback(prompt)
    
    try:
        # Generate text using the model
        result = generator(
            prompt,
            max_length=100,
            num_return_sequences=1,
            temperature=0.7,
            do_sample=True,
            pad_token_id=generator.tokenizer.eos_token_id
        )
        
        generated_text = result[0]['generated_text']
        
        # Clean up the generated text
        if generated_text.startswith(prompt):
            generated_text = generated_text[len(prompt):].strip()
        
        # If the generated text is too short, add some fallback content
        if len(generated_text) < 50:
            fallback = generate_text_fallback(prompt)
            generated_text = f"{generated_text} {fallback}"
        
        return generated_text
        
    except Exception as e:
        logger.error(f"Error generating text with model: {e}")
        return generate_text_fallback(prompt)

def generate_text_fallback(prompt):
    """Fallback text generation when AI model is not available"""
    templates = [
        "Once upon a time, {prompt} led to an extraordinary adventure where dreams became reality and imagination knew no bounds.",
        "In a world where {prompt} was the key to unlocking mysteries, ancient secrets whispered through the cosmic winds.",
        "The story begins with {prompt}, a catalyst that transformed the ordinary into the extraordinary, weaving magic through every moment.",
        "Deep within the realm of possibility, {prompt} sparked a journey that would forever change the fabric of existence.",
        "As the stars aligned, {prompt} became the beacon that guided lost souls through the labyrinth of infinite possibilities."
    ]
    
    variations = [
        "Colors danced in harmony, painting emotions across the canvas of time.",
        "Whispers of forgotten melodies echoed through dimensions unknown.",
        "The universe breathed with anticipation, waiting for the next chapter to unfold.",
        "Shadows and light played eternal games, creating stories within stories.",
        "Time itself seemed to pause, allowing magic to weave its eternal spell."
    ]
    
    prompt_hash = hash(prompt)
    selected_template = templates[prompt_hash % len(templates)]
    selected_variation = variations[prompt_hash % len(variations)]
    
    return selected_template.replace('{prompt}', prompt) + " " + selected_variation

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model_loaded,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/generate', methods=['POST'])
def generate_content():
    """Main endpoint for generating text and sketch"""
    try:
        # Get the prompt from the request
        data = request.get_json()
        if not data or 'prompt' not in data:
            return jsonify({
                'success': False,
                'error': 'No prompt provided'
            }), 400
        
        prompt = data['prompt'].strip()
        if not prompt:
            return jsonify({
                'success': False,
                'error': 'Empty prompt provided'
            }), 400
        
        logger.info(f"Generating content for prompt: {prompt}")
        
        # Generate text content
        generated_text = generate_text_content(prompt)
        
        # Generate sketch
        sketch_base64 = generate_sketch_base64(prompt, generated_text)
        
        # Return the results
        return jsonify({
            'success': True,
            'generated_text': generated_text,
            'sketch_image': f"data:image/png;base64,{sketch_base64}",
            'prompt': prompt,
            'timestamp': datetime.now().isoformat(),
            'model_used': 'AI' if model_loaded else 'fallback'
        })
        
    except Exception as e:
        logger.error(f"Error in generate_content: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/', methods=['GET'])
def index():
    """Root endpoint"""
    return jsonify({
        'message': 'MindCanvas AI Backend',
        'version': '1.0.0',
        'endpoints': {
            'health': '/health',
            'generate': '/generate (POST)'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'error': 'Endpoint not found'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'success': False,
        'error': 'Internal server error'
    }), 500

# Initialize the application
if __name__ == '__main__':
    # Load the AI model on startup
    load_model()
    
    # Run the app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting MindCanvas backend on port {port}")
    app.run(host='0.0.0.0', port=port, debug=debug)
