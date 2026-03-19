import requests
import json

BASE_URL = "http://localhost:8001"

def test_health():
    """Test health check endpoint"""
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print("=== HEALTH CHECK ===")
        print(f"Status: {result['status']}")
        print(f"Environment: {result['environment']}")
        print(f"Gemini Available: {result.get('gemini_available', False)}")
        print(f"Gemini Configured: {result.get('gemini_configured', False)}")
        print(f"Playwright Available: {result.get('playwright_available', False)}")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_audit_matrices():
    """Test AI analysis endpoint (returns AI analysis only, no raw matrices)"""
    url = f"{BASE_URL}/audit/matrices"
    payload = {
        "url": "https://linkrunner.io/blog/top-6-appsflyer-alternatives-for-indian-mobile-marketers-in-2025",
        "product_name": "Linkrunner",
        "user_id": "3c451d93-1287-4b20-9d08-a0eaa8f953e9",
        "product_id": "b2429888-4c5d-492d-81ca-d9163319a0f4"
    }
    # payload = {
    #     "url": "https://godseyes.world/",
    #     "product_name": "GodsEye"
    # }
    
    print("\n=== TEST: /audit/matrices (AI Analysis Only) ===")
    print(f"URL: {payload['url']}")
    print("Note: This endpoint returns AI analysis only (no raw matrices)")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(response.text)
            
    elif response.status_code == 501:
        print("Error: Playwright not installed. Run: pip install playwright && playwright install chromium")
    else:
        print(f"Error: {response.status_code} - {response.text}")

def test_full_audit():
    """Test full audit with Gemini AI analysis"""
    url = f"{BASE_URL}/audit"
    payload = {
        "url": "https://linkrunner.io/blog/top-6-appsflyer-alternatives-for-indian-mobile-marketers-in-2025",
        "product_name": "Linkrunner",
        "user_id": "3c451d93-1287-4b20-9d08-a0eaa8f953e9",
        "product_id": "b2429888-4c5d-492d-81ca-d9163319a0f4"
    }
    # payload = {
    #     "url": "https://godseyes.world/",
    #     "product_name": "GodsEye"
    # }
    
    print("\n=== TEST: /audit (AI Analysis Only) ===")
    print(f"URL: {payload['url']}")
    print("Note: This endpoint calls Gemini 2.5 Flash - may take 10-30 seconds")
    response = requests.post(url, json=payload)
    
    if response.status_code == 200:
        print(response.text)
            
    elif response.status_code == 501:
        print("Error: Playwright not installed. Run: pip install playwright && playwright install chromium")
    else:
        print(f"Error: {response.status_code} - {response.text}")

if __name__ == "__main__":
    # test_health()
    test_full_audit()
