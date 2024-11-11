import os
import uvicorn
from fastapi import FastAPI, Body, Depends
from pydantic import BaseModel
from fastapi.responses import FileResponse
from melo.api import TTS
from dotenv import load_dotenv
import tempfile
import threading
import time

load_dotenv()
DEFAULT_SPEED = float(os.getenv("DEFAULT_SPEED"))
DEFAULT_LANGUAGE = os.getenv("DEFAULT_LANGUAGE")
DEFAULT_SPEAKER_ID = os.getenv("DEFAULT_SPEAKER_ID")
# Idle timeout for unloading the model (in seconds), defaulting to -1. If the timeout is -1 the model won't be unloaded.
MODEL_IDLE_TIMEOUT = int(os.getenv("MODEL_IDLE_TIMEOUT", -1))
device = "auto"  # Will automatically use GPU if available

class TextModel(BaseModel):
    text: str
    speed: float = DEFAULT_SPEED
    language: str = DEFAULT_LANGUAGE
    speaker_id: str = DEFAULT_SPEAKER_ID

app = FastAPI()

class ModelManager:
    def __init__(self):
        self.model = None
        self.last_used = time.time()
        self.lock = threading.Lock()
        if MODEL_IDLE_TIMEOUT != -1:
            self._start_cleanup_thread()

    def _start_cleanup_thread(self):
        # Start a background thread that will periodically check and unload the model if idle
        def cleanup():
            while True:
                with self.lock:
                    if self.model and (time.time() - self.last_used) > MODEL_IDLE_TIMEOUT:
                        print("Unloading model due to inactivity...")
                        self.model = None
                time.sleep(60)  # Check every minute

        thread = threading.Thread(target=cleanup)
        thread.start()

    def get_model(self, language):
        with self.lock:
            if not self.model:
                print("Loading TTS model...")
                self.model = TTS(language=language, device=device)
            self.last_used = time.time()
            return self.model

# Instantiate the model manager
model_manager = ModelManager()

def get_tts_model(body: TextModel):
    return TTS(language=body.language, device=device)


@app.post("/convert/tts")
async def create_upload_file(
    body: TextModel = Body(...), model: TTS = Depends(get_tts_model)
):
    speaker_ids = model.hps.data.spk2id

    print(os.path.basename(body.text))

    # Use a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        output_path = tmp.name
        model.tts_to_file(
            body.text, speaker_ids[body.speaker_id], output_path, speed=body.speed
        )

        # Return the audio file, ensure the file is not deleted until after the response is sent
        response = FileResponse(
            output_path, media_type="audio/mpeg", filename=os.path.basename(output_path)
        )

    return response


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)
