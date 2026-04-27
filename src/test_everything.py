#!/usr/bin/env python3
"""
Complete System Test - Tests all components
"""
import sys
import os

def test_imports():
    """Test all Python imports"""
    print("=" * 60)
    print("Testing Python Imports...")
    print("=" * 60)
    
    try:
        # Test main imports
        print("Testing main.py imports...")
        import main
        print("✅ main.py imports successfully")
        
        # Test security
        print("Testing security.py imports...")
        import security
        print("✅ security.py imports successfully")
        
        # Test models
        print("Testing model imports...")
        from finsight_models_production import (
            HealthModel, WasteModel, GoalModel, ClusterModel,
            score_behavioral_answers, explain_health, explain_waste, explain_goal
        )
        print("✅ All models import successfully")
        
        # Test model initialization
        print("\nTesting model initialization...")
        health = HealthModel()
        waste = WasteModel()
        goal = GoalModel()
        cluster = ClusterModel()
        print("✅ All models initialize successfully")
        
        return True
    except Exception as e:
        print(f"❌ Import test failed: {e}")
        return False

def test_fastapi_app():
    """Test FastAPI app creation"""
    print("\n" + "=" * 60)
    print("Testing FastAPI Application...")
    print("=" * 60)
    
    try:
        from main import app
        print("✅ FastAPI app created successfully")
        
        # Check routes
        routes = [getattr(route, 'path', '') for route in app.routes]
        print(f"✅ Found {len(routes)} routes")
        
        # Check critical endpoints
        critical_endpoints = [
            '/auth/login', '/auth/signup', '/predict/health', 
            '/predict/waste', '/predict/goal', '/predict/cluster', '/predict/behavioral-scores'
        ]
        for endpoint in critical_endpoints:
            if endpoint in routes:
                print(f"✅ {endpoint} endpoint exists")
            else:
                print(f"⚠️  {endpoint} endpoint not found")
        
        return True
    except Exception as e:
        print(f"❌ FastAPI test failed: {e}")
        return False

def test_database():
    """Test database setup"""
    print("\n" + "=" * 60)
    print("Testing Database...")
    print("=" * 60)
    
    try:
        import sqlite3
        
        # Check if database file exists
        if os.path.exists('finsight.db'):
            print("✅ Database file exists")
            
            # Test connection
            conn = sqlite3.connect('finsight.db')
            c = conn.cursor()
            
            # Check tables
            c.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = c.fetchall()
            print(f"✅ Found {len(tables)} tables: {[t[0] for t in tables]}")
            
            conn.close()
        else:
            print("⚠️  Database file doesn't exist (will be created on first run)")
        
        return True
    except Exception as e:
        print(f"❌ Database test failed: {e}")
        return False

def test_environment():
    """Test environment configuration"""
    print("\n" + "=" * 60)
    print("Testing Environment Configuration...")
    print("=" * 60)
    
    try:
        from dotenv import load_dotenv
        load_dotenv()
        
        # Check for .env file
        if os.path.exists('.env'):
            print("✅ .env file exists")
        else:
            print("⚠️  .env file not found (using defaults)")
        
        # Check environment variables
        jwt_secret = os.getenv('JWT_SECRET')
        if jwt_secret:
            print("✅ JWT_SECRET is set")
        else:
            print("⚠️  JWT_SECRET not set (using default)")
        
        gemini_key = os.getenv('GEMINI_API_KEY')
        if gemini_key:
            print("✅ GEMINI_API_KEY is set")
        else:
            print("⚠️  GEMINI_API_KEY not set")
        
        return True
    except Exception as e:
        print(f"❌ Environment test failed: {e}")
        return False

def test_data_files():
    """Test if data files exist"""
    print("\n" + "=" * 60)
    print("Testing Data Files...")
    print("=" * 60)
    
    data_dir = 'src/finsight_models_production/data'
    required_files = [
        'health_score_dataset_v3.csv',
        'waste_recovery_dataset_v6.csv',
        'goal_intelligence_dataset_v2.csv',
        'behavioral_clusters_dataset.csv',
        'health_weights.json'
    ]
    
    all_exist = True
    for file in required_files:
        filepath = os.path.join(data_dir, file)
        if os.path.exists(filepath):
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - MISSING")
            all_exist = False
    
    return all_exist

def main():
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "FinSight AI - Complete System Test" + " " * 14 + "║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    results = {
        'imports': test_imports(),
        'fastapi': test_fastapi_app(),
        'database': test_database(),
        'environment': test_environment(),
        'data_files': test_data_files(),
    }
    
    print("\n" + "=" * 60)
    print("FINAL RESULTS")
    print("=" * 60)
    
    for test_name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{test_name.upper()}: {status}")
    
    all_passed = all(results.values())
    
    print("\n" + "=" * 60)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("\nYour FinSight AI system is ready to run!")
        print("\nTo start the backend:")
        print("  python main.py")
        print("\nTo start the frontend:")
        print("  npm run dev")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("\nPlease fix the issues above before running the application.")
    print("=" * 60)
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())
