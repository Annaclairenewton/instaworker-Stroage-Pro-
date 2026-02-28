"""
InstaWorker™ Pro — Entry Point
==============================
Run this file to start the warehouse management system.

Usage:
    streamlit run run.py

Structure:
    instaworker/
    ├── run.py              ← Start here
    ├── backend/
    │   └── core.py         ← Data, logic, AI engine, RAG
    ├── frontend/
    │   └── app.py          ← All Streamlit UI views
    └── vision_agent.py     ← Camera + on-device AI agent (separate)
"""

import sys
import os

# Add paths
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'backend'))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'frontend'))

# Launch frontend
from frontend.app import *
