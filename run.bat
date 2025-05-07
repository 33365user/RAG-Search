@echo off

start "" "frontend.html"
uvicorn rag_at2:app --reload
