from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import torch
if not hasattr(torch, "xpu"):
    torch.xpi = type("MockXPU", (), {"empty_cache": lambda: None})
from diffusers import StableDiffusionPipeline, EulerAncestralDiscreteScheduler
from peft import PeftModel
from io import BytesIO
import base64
from PIL import Image
import os

app = FastAPI(title = "Indian Art Generator API")

#CORS Middleware frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["*"],
    allow_credentials = True,
    allow_methods = ["*"],
    allow_headers = ["*"],
)

pipe = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BASE_MODEL = "runway/stable-diffusion-v1-5"
LORA_WEIGHTS_PATH = "./lora_weights"

class PromptRequest(BaseModel):
    prompt: str
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    seed: int = None

class GenerationResponse(BaseModel):
    image_base64: str
    prompt: str
    seed: int

@app.on_event("startup")
async def load_model():
    global pipe

    print("="*100)
    print("Loading Stable Diffusion with LoRA weights")
    print("="*100)

    pipe = StableDiffusionPipeline.from_pretrained(
        BASE_MODEL,
        torch_dtype = torch.float16 if DEVICE == "cuda" else torch.float32,
        safety_checker = None
    )

    if os.path.exists(LORA_WEIGHTS_PATH):
        print(f"Loading LoRA from: {LORA_WEIGHTS_PATH}")
        pipe.unet = PeftModel.from_pretrained(pipe.unet, LORA_WEIGHTS_PATH)
    else:
        print(f"WARNING: LoRa weights not plugged in")
        print("Using base model only")

    pipe = pipe.to(DEVICE)
    pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(pipe.scheduler.config)

    print("Model loaded successfully")
    print(f"Device: {DEVICE}")
    print("="*100)

@app.get("/")
async def root():
    #checkup endpoint
    return {
        "status": "online",
        "model": "Indian Art Generator",
        "device": DEVICE,
        "lora_loaded": os.path.exists(LORA_WEIGHTS_PATH)
    }

@app.post("/generate", response_model = GenerationResponse)
async def generate_image(request: PromptRequest):
    #Generating Image

    if pipe is None:
        raise HTTPException(status_code = 503, detail = "Model not loaded")
    
    try:
        generator = None
        if request.seed is not None:
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        else:
            request.seed = torch.randint(0, 2**32, (1,)).item()
            generator = torch.Generator(device=DEVICE).manual_seed(request.seed)
        
        with torch.autocaset(DEVICE):
            result = pipe(
                prompt = request.prompt,
                negative_prompt = "blurry, low quality, distorted, ugly, bad anatomy, watermark",
                num_inference_steps = request.num_inference_steps,
                guidance_scale = request.guidance_scale,
                width = 512,
                height = 512,
                generator = generator
            )
        image = result.images[0]

        buffered = BytesIO()
        image.save(buffered, format = "PNG")
        img_str = base64.b64encode(buffered.getvalue()).decode()

        return GenerationResponse(
            image_base64 = img_str,
            prompt = request.prompt,
            seed = request.seed
        )
    except Exception as e:
        raise HTTPException(status_code = 500, detail=str(e))


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": pipe is not None,
        "device": DEVICE,
        "cuda_available": torch.cuda.is_available()
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host = "0.0.0.0", port = 8000)