#!/usr/bin/env python3
"""
Test the SSL session creation
"""

import requests
import ssl
import os
from requests.adapters import HTTPAdapter
from urllib3.util.ssl_ import create_urllib3_context

def create_secure_session():
    """Create a requests session with enterprise SSL certificate configuration"""
    session = requests.Session()
    
    # Use final complete certificate bundle with all certificates in chain
    cert_bundle_path = "/Users/a0144076/sample-streamlit-rag-langchain/corp-bundle-final-complete.pem"
    
    # Create SSL context with enterprise certificates
    ctx = create_urllib3_context()
    ctx.check_hostname = True
    ctx.verify_mode = ssl.CERT_REQUIRED
    
    # Load enterprise certificate bundle and system certificates
    if os.path.exists(cert_bundle_path):
        ctx.load_verify_locations(cert_bundle_path)
        # Also load system keychain certificates (macOS)
        ctx.load_default_certs()
        print(f"Using enterprise certificate bundle + system keychain: {cert_bundle_path}")
    else:
        print("Certificate bundle not found, using default")
        return None
    
    # Custom adapter
    class SSLAdapter(HTTPAdapter):
        def init_poolmanager(self, *args, **kwargs):
            kwargs['ssl_context'] = ctx
            return super().init_poolmanager(*args, **kwargs)
    
    session.mount('https://', SSLAdapter())
    return session

def test_session():
    print("Testing secure session...")
    
    try:
        session = create_secure_session()
        if not session:
            print("Failed to create session")
            return False
            
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/86.0.4240.111 Safari/537.36"
        }
        
        url = "https://langfuse.com/guides/cookbook/evaluation_of_rag_with_ragas"
        print(f"Testing URL: {url}")
        
        response = session.get(url, headers=headers, timeout=30)
        print(f"Status: {response.status_code}")
        print("SUCCESS: SSL session works!")
        return True
        
    except Exception as e:
        print(f"FAILED: {e}")
        return False

if __name__ == "__main__":
    test_session()