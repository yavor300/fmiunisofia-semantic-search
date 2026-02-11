"""
Flask Web Interface for Semantic Search Engine.

Main entry point for the web application.
Allows users to configure and test the search engine through a web UI.
"""

import os
import logging
from flask import Flask, render_template, request, jsonify, session
from flask_cors import CORS

from src.search_engine import ProductSearchEngine

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'semantic-search-secret-key-change-in-production')
CORS(app)

# Global search engine instance
search_engine = None
engine_initialized = False


@app.route('/')
def index():
    """Render main page."""
    return render_template('index.html')


@app.route('/api/config', methods=['GET'])
def get_config():
    """Get current configuration."""
    global search_engine, engine_initialized
    
    config = session.get('config', {
        'use_elasticsearch': True,
        'use_dependency_parsing': False,
        'use_embeddings': False,
        'hybrid_alpha': 0.6,
        'top_k': 10,
        'data_limit': None
    })
    
    return jsonify({
        'config': config,
        'initialized': engine_initialized,
        'statistics': search_engine.get_statistics() if search_engine else {}
    })


@app.route('/api/config', methods=['POST'])
def update_config():
    """Update configuration and reinitialize search engine."""
    global search_engine, engine_initialized
    
    try:
        config = request.json
        
        # Validate configuration
        use_elasticsearch = config.get('use_elasticsearch', True)
        use_dependency_parsing = config.get('use_dependency_parsing', False)
        use_embeddings = config.get('use_embeddings', False)
        hybrid_alpha = float(config.get('hybrid_alpha', 0.6))
        top_k = int(config.get('top_k', 10))
        data_limit = config.get('data_limit')
        
        if data_limit:
            data_limit = int(data_limit)
        
        # Store configuration in session
        session['config'] = {
            'use_elasticsearch': use_elasticsearch,
            'use_dependency_parsing': use_dependency_parsing,
            'use_embeddings': use_embeddings,
            'hybrid_alpha': hybrid_alpha,
            'top_k': top_k,
            'data_limit': data_limit
        }
        
        logger.info(f"Configuration updated: {session['config']}")
        
        return jsonify({
            'success': True,
            'message': 'Configuration updated. Please initialize the search engine.',
            'config': session['config']
        })
        
    except Exception as e:
        logger.error(f"Error updating configuration: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400


@app.route('/api/initialize', methods=['POST'])
def initialize_engine():
    """Initialize search engine with current configuration."""
    global search_engine, engine_initialized
    
    try:
        config = session.get('config', {
            'use_elasticsearch': True,
            'use_dependency_parsing': False,
            'use_embeddings': False,
            'hybrid_alpha': 0.6,
            'top_k': 10,
            'data_limit': None,
            'verbose': True
        })
        
        data_file = request.json.get('data_file', 'data/amazon-products.csv')
        
        logger.info(f"Initializing search engine with config: {config}")
        logger.info(f"Data file: {data_file}")
        
        # Check if data file exists
        if not os.path.exists(data_file):
            return jsonify({
                'success': False,
                'error': f'Data file not found: {data_file}'
            }), 400
        
        # Initialize search engine
        search_engine = ProductSearchEngine(
            use_elasticsearch=config['use_elasticsearch'],
            use_dependency_parsing=config['use_dependency_parsing'],
            use_embeddings=config['use_embeddings'],
            hybrid_alpha=config['hybrid_alpha']
        )
        
        # Load data
        if not search_engine.load_data(data_file, limit=config['data_limit']):
            return jsonify({
                'success': False,
                'error': 'Failed to load data'
            }), 500
        
        # Build index
        if not search_engine.build_index():
            return jsonify({
                'success': False,
                'error': 'Failed to build index'
            }), 500
        
        engine_initialized = True
        
        stats = search_engine.get_statistics()
        
        logger.info(f"Search engine initialized successfully. Stats: {stats}")
        
        return jsonify({
            'success': True,
            'message': 'Search engine initialized successfully',
            'statistics': stats
        })
        
    except Exception as e:
        logger.error(f"Error initializing search engine: {e}", exc_info=True)
        engine_initialized = False
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/search', methods=['POST'])
def search():
    """Perform search with current configuration."""
    global search_engine, engine_initialized
    
    if not engine_initialized or not search_engine:
        return jsonify({
            'success': False,
            'error': 'Search engine not initialized. Please initialize first.'
        }), 400
    
    try:
        data = request.json
        query = data.get('query', '')
        
        if not query:
            return jsonify({
                'success': False,
                'error': 'Query is required'
            }), 400
        
        config = session.get('config', {'top_k': 10})
        top_k = int(data.get('top_k', config.get('top_k', 10)))
        verbose = data.get('verbose', True)
        
        logger.info(f"Searching for: '{query}' (top_k={top_k}, verbose={verbose})")
        
        # Perform search
        results = search_engine.search(
            query=query,
            top_k=top_k,
            verbose=True
        )
        
        logger.info(f"Found {len(results)} results")
        
        return jsonify({
            'success': True,
            'query': query,
            'results': results,
            'count': len(results)
        })
        
    except Exception as e:
        logger.error(f"Error during search: {e}", exc_info=True)
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/api/status', methods=['GET'])
def status():
    """Get system status."""
    global search_engine, engine_initialized
    
    es_status = 'unknown'
    
    if engine_initialized and search_engine and search_engine.use_elasticsearch:
        try:
            # Try to get stats to check ES connection
            stats = search_engine.get_statistics()
            es_status = 'connected'
        except:
            es_status = 'disconnected'
    elif engine_initialized and search_engine and not search_engine.use_elasticsearch:
        es_status = 'not_used'
    
    return jsonify({
        'initialized': engine_initialized,
        'elasticsearch_status': es_status,
        'config': session.get('config', {})
    })


@app.route('/api/reset', methods=['POST'])
def reset():
    """Reset search engine and clear configuration."""
    global search_engine, engine_initialized
    
    try:
        if search_engine:
            search_engine.close()
        
        search_engine = None
        engine_initialized = False
        session.clear()
        
        logger.info("Search engine reset successfully")
        
        return jsonify({
            'success': True,
            'message': 'Search engine reset successfully'
        })
        
    except Exception as e:
        logger.error(f"Error resetting search engine: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


if __name__ == '__main__':
    # Create templates and static directories if they don't exist
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the Flask app
    port = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('DEBUG', 'True').lower() == 'true'
    
    logger.info(f"Starting Flask app on port {port} (debug={debug})")
    app.run(host='0.0.0.0', port=port, debug=debug)
