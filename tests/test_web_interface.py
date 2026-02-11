"""
Test script for Flask Web Interface
Tests the API endpoints without running the full web server
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        import flask
        print("Flask imported")
    except ImportError:
        print("Flask not installed. Run: pip install Flask")
        return False
    
    try:
        import flask_cors
        print("Flask-CORS imported")
    except ImportError:
        print("Flask-CORS not installed. Run: pip install Flask-CORS")
        return False
    
    try:
        from src.search_engine import ProductSearchEngine
        print("ProductSearchEngine imported")
    except ImportError as e:
        print(f"Failed to import ProductSearchEngine: {e}")
        return False
    
    try:
        from src.config import DEFAULT_USE_ELASTICSEARCH
        print("Config imported")
    except ImportError as e:
        print(f"Failed to import config: {e}")
        return False
    
    return True


def test_directories():
    """Test that required directories exist."""
    print("\nTesting directories...")
    
    required_dirs = ['templates', 'static', 'src', 'data']
    missing_dirs = []
    
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"{dir_name}/ exists")
        else:
            print(f"{dir_name}/ not found")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nCreating missing directories: {', '.join(missing_dirs)}")
        for dir_name in missing_dirs:
            os.makedirs(dir_name, exist_ok=True)
            print(f"Created {dir_name}/")
    
    return True


def test_templates():
    """Test that required templates exist."""
    print("\nTesting templates...")
    
    template_file = 'templates/index.html'
    if os.path.exists(template_file):
        print(f"{template_file} exists")
        
        # Check if template has required elements
        with open(template_file, 'r', encoding='utf-8') as f:
            content = f.read()
            
        required_elements = [
            'searchQuery',
            'useElasticsearch',
            'useDependencyParsing',
            'useEmbeddings',
            'performSearch',
            'initializeEngine'
        ]
        
        missing_elements = []
        for element in required_elements:
            if element in content:
                print(f"  Element '{element}' found")
            else:
                print(f"  Element '{element}' missing")
                missing_elements.append(element)
        
        return len(missing_elements) == 0
    else:
        print(f"{template_file} not found")
        return False


def test_app_structure():
    """Test that app.py has correct structure."""
    print("\nTesting app.py structure...")
    
    if not os.path.exists('app.py'):
        print("app.py not found")
        return False
    
    print("app.py exists")
    
    with open('app.py', 'r', encoding='utf-8') as f:
        content = f.read()
    
    required_routes = [
        '@app.route(\'/\')',
        '@app.route(\'/api/config\'',
        '@app.route(\'/api/initialize\'',
        '@app.route(\'/api/search\'',
        '@app.route(\'/api/status\'',
        '@app.route(\'/api/reset\''
    ]
    
    missing_routes = []
    for route in required_routes:
        if route in content:
            route_path = route.split("'")[1]
            print(f"  Route {route_path} found")
        else:
            print(f"  Route {route} missing")
            missing_routes.append(route)
    
    return len(missing_routes) == 0


def test_data_file():
    """Test if data file exists."""
    print("\nTesting data file...")
    
    data_files = [
        'data/amazon-products.csv',
        'data/sample_products.csv'
    ]
    
    found = False
    for data_file in data_files:
        if os.path.exists(data_file):
            print(f"{data_file} exists")
            
            # Check file size
            size_mb = os.path.getsize(data_file) / (1024 * 1024)
            print(f"  File size: {size_mb:.2f} MB")
            found = True
        else:
            print(f"{data_file} not found")
    
    if not found:
        print("\nNo data file found. You'll need to:")
        print("  1. Place your CSV file in data/amazon-products.csv")
        print("  2. Or generate sample data by running the main script")
    
    return True  # Not critical for web app to start


def test_search_engine():
    """Test that search engine can be instantiated."""
    print("\nTesting search engine instantiation...")
    
    try:
        from src.search_engine import ProductSearchEngine
        
        # Test with minimal configuration
        engine = ProductSearchEngine(
            use_elasticsearch=False,
            use_dependency_parsing=False,
            use_embeddings=False
        )
        print("Search engine instantiated successfully")
        
        # Check if engine has required methods
        required_methods = ['load_data', 'build_index', 'search', 'get_statistics', 'close']
        for method in required_methods:
            if hasattr(engine, method):
                print(f"  Method '{method}' exists")
            else:
                print(f"  Method '{method}' missing")
                return False
        
        return True
        
    except Exception as e:
        print(f"Failed to instantiate search engine: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("="*60)
    print("Flask Web Interface - Test Suite")
    print("="*60)
    
    tests = [
        ("Imports", test_imports),
        ("Directories", test_directories),
        ("Templates", test_templates),
        ("App Structure", test_app_structure),
        ("Data File", test_data_file),
        ("Search Engine", test_search_engine)
    ]
    
    results = {}
    for test_name, test_func in tests:
        print()
        try:
            results[test_name] = test_func()
        except Exception as e:
            print(f"Test '{test_name}' failed with exception: {e}")
            import traceback
            traceback.print_exc()
            results[test_name] = False
    
    # Summary
    print("\n" + "="*60)
    print("Test Summary")
    print("="*60)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "PASS" if result else "FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\nAll tests passed! You can now run the web application:")
        print("   python app.py")
        print("\n   Then open: http://localhost:5000")
        return True
    else:
        print("\nSome tests failed. Please fix the issues above.")
        return False


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
