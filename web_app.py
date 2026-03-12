import os
import shutil
import uuid
import argparse
import asyncio
import json
import threading
from typing import List, Dict
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

# --- Deep-Live-Cam Imports ---
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import modules.globals
import modules.core
from modules.utilities import is_image, is_video

# --- FastAPI App ---
app = FastAPI(title="Deep-Live-Cam Web Interface")

# Directories
TEMP_DIR = os.path.join(BASE_DIR, "web_temp")
UPLOADS_DIR = os.path.join(TEMP_DIR, "uploads")
OUTPUTS_DIR = os.path.join(TEMP_DIR, "outputs")
STATIC_DIR = os.path.join(BASE_DIR, "web", "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "web", "templates")

for d in [UPLOADS_DIR, OUTPUTS_DIR]:
    os.makedirs(d, exist_ok=True)

app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# active_websockets: Dict[str, WebSocket] = {}
websocket_pool = []

class ProcessOptions(BaseModel):
    session_id: str
    source: str
    target: str
    enhancer: bool
    poisson: bool
    opacity: float

# store main event loop for background task communication
main_loop = None

@app.on_event("startup")
async def startup_event():
    global main_loop
    main_loop = asyncio.get_event_loop()

@app.get("/", response_class=HTMLResponse)
async def read_item():
    with open(os.path.join(TEMPLATES_DIR, "index.html"), "r") as f:
        return f.read()

@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    session_id = str(uuid.uuid4())
    ext = os.path.splitext(file.filename)[1].lower()
    filename = f"{session_id}{ext}"
    filepath = os.path.join(UPLOADS_DIR, filename)
    
    with open(filepath, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    
    return {"session_id": session_id, "filename": filename}

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    websocket_pool.append(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        websocket_pool.remove(websocket)

async def notify_clients(message: dict):
    for ws in websocket_pool:
        try:
            await ws.send_json(message)
        except:
            pass

# Monkey-patching modules.core.update_status to send WS updates
original_update_status = modules.core.update_status

def web_update_status(message: str, scope: str = 'DLC.CORE'):
    original_update_status(message, scope)
    if main_loop and main_loop.is_running():
        asyncio.run_coroutine_threadsafe(
            notify_clients({"type": "progress", "status": message, "value": 50}),
            main_loop
        )

modules.core.update_status = web_update_status

def run_processing_task(options: ProcessOptions):
    # Set Globals
    modules.globals.source_path = os.path.join(UPLOADS_DIR, options.source)
    modules.globals.target_path = os.path.join(UPLOADS_DIR, options.target)
    
    ext = os.path.splitext(options.target)[1].lower()
    output_filename = f"result_{options.session_id}{ext}"
    modules.globals.output_path = os.path.join(OUTPUTS_DIR, output_filename)
    
    modules.globals.headless = True
    modules.globals.frame_processors = ["face_swapper"]
    if options.enhancer:
        modules.globals.frame_processors.append("face_enhancer")
    
    modules.globals.poisson_blend = options.poisson
    modules.globals.opacity = options.opacity
    
    # Run Core
    try:
        modules.core.start()
        
        # Notify completion
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                notify_clients({
                    "type": "complete", 
                    "url": f"/download/{output_filename}"
                }),
                main_loop
            )
    except Exception as e:
        if main_loop and main_loop.is_running():
            asyncio.run_coroutine_threadsafe(
                notify_clients({"type": "progress", "status": f"Error: {str(e)}", "value": 0}),
                main_loop
            )

@app.post("/process")
async def process_task(options: ProcessOptions, background_tasks: BackgroundTasks):
    background_tasks.add_task(run_processing_task, options)
    return {"status": "started"}

@app.get("/download/{filename}")
async def download_result(filename: str):
    filepath = os.path.join(OUTPUTS_DIR, filename)
    if os.path.exists(filepath):
        return FileResponse(filepath)
    return {"error": "File not found"}

if __name__ == "__main__":
    import uvicorn
    # Pre-select best execution providers for web mode
    modules.globals.execution_providers = modules.core.decode_execution_providers(
        modules.core.suggest_execution_providers()
    )
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=9000)
    args = parser.parse_args()
    
    uvicorn.run(app, host="0.0.0.0", port=args.port)
