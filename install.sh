#!/bin/bash
# Quick install: copy automation.py to backend/
echo "Installing new features..."
cp backend/automation.py ~/Desktop/instaworker_v7/backend/automation.py 2>/dev/null
cp frontend/app.py ~/Desktop/instaworker_v7/frontend/app.py 2>/dev/null
cp backend/core.py ~/Desktop/instaworker_v7/backend/core.py 2>/dev/null
echo "Done! Run: cd ~/Desktop/instaworker_v7 && streamlit run frontend/app.py"
