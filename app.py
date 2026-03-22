import os
import uvicorn
os.environ["GRADIO_SERVER_PORT"] = "7860"
os.environ["GRADIO_SERVER_NAME"] = "0.0.0.0"
os.environ["GRADIO_ROOT_PATH"] = "/"

from api.main import app

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(app, host="0.0.0.0", port=port)