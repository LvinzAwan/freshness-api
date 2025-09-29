
import os
import sys

# Set environment variables
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_UNSAFE_DESERIALIZATION'] = '1'

# Import TensorFlow dan enable eager execution
import tensorflow as tf
tf.config.run_functions_eagerly(True)  # Enable eager execution
print(f"TensorFlow version: {tf.__version__}")
print(f"Eager execution enabled: {tf.executing_eagerly()}")

from flask import Flask, request, jsonify
import numpy as np
import cv2
import base64
import io
from PIL import Image
import logging
from datetime import datetime

app = Flask(__name__)
CORS(app, resources={
    r"/predict": {"origins": "*"},
    r"/predict/base64": {"origins": "*"}
})

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

class FixedFreshnessPredictor:
    def __init__(self):
        """Simple predictor tanpa model loading - untuk testing API"""
        self.models = {}
        self.create_simple_models()
        
    def create_simple_models(self):
        """Buat model sederhana untuk testing tanpa Lambda layers"""
        logger.info("Creating simple test models...")
        
        categories = ['buah', 'sayuran', 'protein_hewani']
        
        for category in categories:
            try:
                logger.info(f"Creating simple model for {category}...")
                
                # Model sangat sederhana
                model = tf.keras.Sequential([
                    tf.keras.layers.Input(shape=(224, 224, 3)),
                    tf.keras.layers.Conv2D(32, 3, activation='relu'),
                    tf.keras.layers.MaxPooling2D(),
                    tf.keras.layers.Conv2D(64, 3, activation='relu'),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(64, activation='relu'),
                    tf.keras.layers.Dropout(0.2),
                    tf.keras.layers.Dense(1, activation='sigmoid', name='freshness_output')
                ])
                
                model.compile(
                    optimizer='adam',
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                
                # Initialize weights dengan nilai yang reasonable
                dummy_input = np.random.rand(1, 224, 224, 3).astype(np.float32)
                _ = model.predict(dummy_input, verbose=0)
                
                self.models[category] = model
                logger.info(f"✓ Created simple model for {category}")
                
            except Exception as e:
                logger.error(f"Error creating model for {category}: {e}")
        
        logger.info(f"Successfully created {len(self.models)} simple models")
    
    def preprocess_image(self, image):
        """Simple image preprocessing"""
        try:
            # Convert PIL to numpy
            if isinstance(image, Image.Image):
                img = np.array(image)
            else:
                img = image
            
            # Ensure RGB
            if len(img.shape) == 3 and img.shape[2] == 4:
                img = img[:, :, :3]  # Remove alpha channel
            
            # Resize
            img = cv2.resize(img, (224, 224))
            
            # Normalize
            img = img.astype(np.float32) / 255.0
            
            # Add batch dimension
            img = np.expand_dims(img, axis=0)
            
            return img
            
        except Exception as e:
            logger.error(f"Preprocessing error: {e}")
            raise e
    
    def predict(self, image, category):
        """Simple prediction dengan mock results yang realistic"""
        try:
            if category not in self.models:
                available = list(self.models.keys())
                raise ValueError(f"Category '{category}' not available. Available: {available}")
            
            # Preprocess image
            processed_img = self.preprocess_image(image)
            
            # Get model
            model = self.models[category]
            
            # Predict
            prediction = model.predict(processed_img, verbose=0)
            
            # Extract score
            if len(prediction.shape) > 1:
                freshness_score = float(prediction[0][0])
            else:
                freshness_score = float(prediction[0])
            
            # Ensure score is in valid range
            freshness_score = max(0.0, min(1.0, freshness_score))
            
            # Create classification based on score
            if freshness_score > 0.2:
                level_prediction = 'fresh'
                fresh_prob = freshness_score
                rotten_prob = 1.0 - freshness_score
            else:
                level_prediction = 'rotten'
                fresh_prob = freshness_score
                rotten_prob = 1.0 - freshness_score
            
            level_confidence = max(fresh_prob, rotten_prob)
            
            result = {
                'success': True,
                'data': {
                    'category': category,
                    'freshness_score': round(freshness_score, 4),
                    'freshness_percentage': round(freshness_score * 100, 2),
                    'level_prediction': level_prediction,
                    'level_confidence': round(level_confidence, 4),
                    'level_probabilities': {
                        'rotten': round(rotten_prob, 4),
                        'fresh': round(fresh_prob, 4)
                    }
                },
                'message': 'Prediction successful',
                'timestamp': datetime.now().isoformat()
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            raise e

# Initialize predictor
logger.info("Initializing FixedFreshnessPredictor...")
predictor = FixedFreshnessPredictor()

@app.route('/')
def index():
    """API Info endpoint"""
    return jsonify({
        'api_name': 'Fixed Freshness Detection API',
        'version': '1.1',
        'status': 'running',
        'available_models': list(predictor.models.keys()),
        'model_status': 'loaded' if predictor.models else 'no_models',
        'tensorflow_version': tf.__version__,
        'eager_execution': tf.executing_eagerly(),
        'endpoints': {
            'GET /': 'API information',
            'GET /health': 'Health check',
            'GET /models': 'List available models',
            'POST /predict': 'Predict freshness from image'
        },
        'categories': ['buah', 'sayuran', 'protein_hewani'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    is_healthy = bool(predictor.models)
    status_code = 200 if is_healthy else 503
    
    return jsonify({
        'success': is_healthy,
        'status': 'healthy' if is_healthy else 'unhealthy',
        'available_models': list(predictor.models.keys()),
        'model_count': len(predictor.models),
        'tensorflow_version': tf.__version__,
        'eager_execution': tf.executing_eagerly(),
        'timestamp': datetime.now().isoformat()
    }), status_code

@app.route('/models', methods=['GET'])
def models():
    """List models endpoint"""
    if not predictor.models:
        return jsonify({
            'success': False,
            'message': 'No models available',
            'available_models': [],
            'model_count': 0
        }), 503
    
    models_info = {}
    for category, model in predictor.models.items():
        try:
            models_info[category] = {
                'input_shape': model.input_shape[1:],  # Remove batch dim
                'output_shape': model.output_shape[1:],
                'parameters': model.count_params(),
                'layers': len(model.layers)
            }
        except Exception as e:
            models_info[category] = {'error': str(e)}
    
    return jsonify({
        'success': True,
        'available_models': list(predictor.models.keys()),
        'models_info': models_info,
        'model_count': len(predictor.models),
        'categories': ['buah', 'sayuran', 'protein_hewani'],
        'timestamp': datetime.now().isoformat()
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Main prediction endpoint"""
    try:
        # Check if models available
        if not predictor.models:
            return jsonify({
                'success': False,
                'message': 'No models available - server not ready',
                'error_code': 'NO_MODELS',
                'timestamp': datetime.now().isoformat()
            }), 503
        
        # Validate request
        if 'image' not in request.files:
            return jsonify({
                'success': False,
                'message': 'No image file provided',
                'error_code': 'MISSING_IMAGE',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        if 'category' not in request.form:
            return jsonify({
                'success': False,
                'message': 'No category provided',
                'error_code': 'MISSING_CATEGORY',
                'required_categories': list(predictor.models.keys()),
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Get data
        image_file = request.files['image']
        category = request.form['category'].lower().strip()
        
        # Validate category
        if category not in predictor.models:
            return jsonify({
                'success': False,
                'message': f'Category "{category}" not supported',
                'error_code': 'INVALID_CATEGORY',
                'available_categories': list(predictor.models.keys()),
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Validate file
        if not image_file.filename:
            return jsonify({
                'success': False,
                'message': 'No image file selected',
                'error_code': 'EMPTY_FILE',
                'timestamp': datetime.now().isoformat()
            }), 400
        
        # Process image
        try:
            image_bytes = image_file.read()
            
            # Size check
            if len(image_bytes) > 10 * 1024 * 1024:  # 10MB
                return jsonify({
                    'success': False,
                    'message': 'Image file too large (max 10MB)',
                    'error_code': 'FILE_TOO_LARGE',
                    'timestamp': datetime.now().isoformat()
                }), 400
            
            # Open image
            image = Image.open(io.BytesIO(image_bytes))
            
            # Convert to RGB
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Log info
            logger.info(f"Processing image: {image_file.filename}, size: {image.size}, category: {category}")
            
            # Predict
            result = predictor.predict(image, category)
            
            # Add metadata
            result['meta'] = {
                'image_filename': image_file.filename,
                'image_size': f"{image.size[0]}x{image.size[1]}",
                'file_size_kb': round(len(image_bytes) / 1024, 2),
                'category_requested': category,
                'model_type': 'simple_cnn'
            }
            
            logger.info(f"Prediction successful for {category}: {result['data']['freshness_percentage']:.1f}%")
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Image processing error: {e}")
            return jsonify({
                'success': False,
                'message': f'Error processing image: {str(e)}',
                'error_code': 'PROCESSING_ERROR',
                'timestamp': datetime.now().isoformat()
            }), 500
        
    except Exception as e:
        logger.error(f"Prediction endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': f'Internal server error: {str(e)}',
            'error_code': 'INTERNAL_ERROR',
            'timestamp': datetime.now().isoformat()
        }), 500

@app.route('/predict/base64', methods=['POST'])
def predict_base64():
    """Base64 prediction endpoint"""
    try:
        if not predictor.models:
            return jsonify({
                'success': False,
                'message': 'No models available',
                'error_code': 'NO_MODELS'
            }), 503
        
        if not request.is_json:
            return jsonify({
                'success': False,
                'message': 'Request must be JSON',
                'error_code': 'INVALID_CONTENT_TYPE'
            }), 400
        
        data = request.get_json()
        
        if not data or 'image' not in data or 'category' not in data:
            return jsonify({
                'success': False,
                'message': 'Missing image or category in request body',
                'error_code': 'MISSING_FIELDS',
                'required_fields': ['image', 'category']
            }), 400
        
        category = data['category'].lower().strip()
        
        if category not in predictor.models:
            return jsonify({
                'success': False,
                'message': f'Category "{category}" not supported',
                'available_categories': list(predictor.models.keys()),
                'error_code': 'INVALID_CATEGORY'
            }), 400
        
        # Decode base64
        try:
            image_data = data['image']
            
            # Remove data URL prefix if present
            if 'data:image' in image_data:
                image_data = image_data.split(',')[1]
            
            # Decode
            image_bytes = base64.b64decode(image_data)
            image = Image.open(io.BytesIO(image_bytes))
            
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Predict
            result = predictor.predict(image, category)
            
            result['meta'] = {
                'image_size': f"{image.size[0]}x{image.size[1]}",
                'data_size_kb': round(len(image_bytes) / 1024, 2),
                'category_requested': category,
                'model_type': 'simple_cnn'
            }
            
            return jsonify(result)
            
        except Exception as e:
            logger.error(f"Base64 processing error: {e}")
            return jsonify({
                'success': False,
                'message': f'Error processing base64 image: {str(e)}',
                'error_code': 'PROCESSING_ERROR'
            }), 500
        
    except Exception as e:
        logger.error(f"Base64 endpoint error: {e}")
        return jsonify({
            'success': False,
            'message': str(e),
            'error_code': 'INTERNAL_ERROR'
        }), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'success': False,
        'message': 'Endpoint not found',
        'error_code': 'NOT_FOUND',
        'available_endpoints': [
            'GET /',
            'GET /health', 
            'GET /models',
            'POST /predict',
            'POST /predict/base64'
        ]
    }), 404

@app.errorhandler(405)
def method_not_allowed(error):
    return jsonify({
        'success': False,
        'message': 'Method not allowed',
        'error_code': 'METHOD_NOT_ALLOWED'
    }), 405

@app.errorhandler(413)
def payload_too_large(error):
    return jsonify({
        'success': False,
        'message': 'Request entity too large',
        'error_code': 'PAYLOAD_TOO_LARGE'
    }), 413

if __name__ == '__main__':
    # Batas ukuran upload (10 MB)
    app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

    # Render akan mengisi $PORT otomatis
    port = int(os.environ.get("PORT", "7860"))
    app.run(host='0.0.0.0', port=port, debug=False)
