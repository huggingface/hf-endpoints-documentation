# Deploy with your own container

If the model you're looking to deploy isn't supported by any of the high-performance inference engines, or you have *custom inference logic*, need *specific Python dependencies*, you can deploy a **custom Docker container** on **Inference Endpoints**.

This requires more upfront work & understanding of running models in production but gives you full control over the hardware and server.

We'll walk you through a simple guide on how to:
- build a FastAPI server to run [`HuggingFaceTB/SmolLM3-3B`](https://huggingface.co/HuggingFaceTB/SmolLM3-3B)
- containerize the server
- deploy the container on Inference Endpoints 

Let's get to it!

## 1. Create the inference server

Start by creating a new diretory and initializing a uv project by running:
```bash
mkdir inference-server & cd inference-server & uv init
```

> We'll be using `uv` to build this project but using pip or conda works as well, just adjust the commands accordingly

The `main.py` file will:

* load the model from `/repository`,
* start a FastAPI app,
* expose a `/health` route and a `/generate` route.

> 🚨 **Important**: Inference Endpoints has a way to download model artifacts super fast, so 
> ideally our code doesn't download anything related to the model.
> The model you select when creating the endpoint will be mounted at `/repository`.
> **So always load your model from `/repository`**, not directly from the Hugging Face Hub.

Before getting to the code, let's also install out dependencies like so:
```bash
uv add transformers torch "fastapi[standard]"
```

And let's build the code step by step. We'll start by adding all imports now (don't worry, we'll use them all in due time)
and also declare a few global variables. Here we set DEVICE and DTYPE to work nicely on a GPU but also to allow us to test it the server on CPU. 

Nothing too complicated 👍

```python
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------
# Config
# ------------------------------------------------------
MODEL_ID = "/repository"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS=512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
```

We’ll also follow a few best practices:

1. **ModelManager**
  Avoid keeping raw global model/tokenizer objects without lifecycle control.
  A small `ModelManager` class lets us:
    * eagerly **load** the model onto the accelerator, and
    * safely **unload** it and free memory when the server shuts down.

The benefit we get here is that we can control the server's behaviour based on the state of the model and tokenizer.
We want to server to start --> load the model & tokenizer --> then signal that the server is ready for requests.

For convenience we also create a small `ModelNotLoadedError` class to be able to communicate more clearly when the model & tokenizer aren't loaded. 

```python
class ModelNotLoadedError(RuntimeError):
    """Raised when attempting to use the model before it is loaded."""


class ModelManager:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    async def load(self):
        """Load model + tokenizer if not already loaded."""
        if self.model is not None and self.tokenizer is not None:
            return

        start = time.perf_counter()
        logger.info("Loading tokenizer and model for %s", self.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
            )
            .to(self.device)
            .eval()
        )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("Finished loading %s in %.2fms", self.model_id, duration_ms)

    async def unload(self):
        """Free model + tokenizer and clear CUDA cache."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self):
        """Return the loaded model + tokenizer or raise if not ready."""
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("Model not loaded")
        return self.model, self.tokenizer

model_manager = ModelManager(MODEL_ID, DEVICE, DTYPE)

```

2. **FastAPI lifespan**
  Use FastAPI’s `lifespan` to:
    * load the model on app startup,
    * unload the model on app shutdown.
  This keeps your server’s memory usage clean and predictable.


```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load()
    try:
        yield
    finally:
        await model_manager.unload()


app = FastAPI(lifespan=lifespan)
```

Now that we have the lifesyce in place we can start building the core logic of the server itself. We'll start by defining the request and response types, so that we know exactly what type of data we can pass in to the server and what type of data we can expect back.

By default the `max_new_tokens` will be 128 and the max is 512. This is a practical way of capping the max memory a request can take. 

```python
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Plain-text prompt")
    max_new_tokens: int = Field(
        128,
        ge=1,
        le=MAX_NEW_TOKENS,
        description="Upper bound on generated tokens",
    )

class GenerateResponse(BaseModel):
    response: str
    input_token_count: int
    output_token_count: int
```
Feel free to extend these to include `temperature`, `top_p` and other configurations supported by the model.

Moving on to creating the routes for the server, let's start with the `/health` route. Here we're finally using the model manager to know if the model and tokenizer are ready to go. If the model manager returns a `ModelNotLoadedError` also return an error with the statuscode of 503.

On Inference Endpoints (and most other platforms), a readiness probe will ping an endpoint every second on its health route, to check that everything is okay. Using this pattern we can clearly signal that the server isn't ready before the models and tokenizer are fully initialized.

```python
@app.get("/health")
def health():
    try:
        model_manager.get()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"message": "API is running."}
```

And finally the most interesting section: the `/generate` route. This is the route that we want to call to actually use the model for text generation.

- It's starts with a similar guard as the `/health` route, we check that the model and tokenizer are loaded, and if not return a 503 error.
- We assume that the model supprots `apply_chat_template`, but fallback to the passing the prompt directly without chat templating
- We encode the text to tokens and call `model.generate()` 
- Lastly, we gather the outputs, decode the tokens to text and return the response


```python
@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    start_time = time.perf_counter()
    try:
        model, tokenizer = model_manager.get()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": request.prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = request.prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    except RuntimeError as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    input_token_count = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_token_count = generated_ids.shape[0]
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "generate prompt_tokens=%d new_tokens=%d max_new_tokens=%d duration_ms=%.2f",
        input_token_count,
        output_token_count,
        request.max_new_tokens,
        duration_ms,
    )

    return GenerateResponse(
        response=generated_text,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
    )
```

You can now run your server locally with:
```bash
uv run uvicorn main:app
```
And go to `http://127.0.0.1:8000/docs` to see the automatic documentation that FastAPI provides. Well done 🙌

<details>
<summary>If you want to copy & paste the full code you'll find it here:</summary>

```python
import logging
import time
from contextlib import asynccontextmanager
from typing import Optional

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from transformers import AutoModelForCausalLM, AutoTokenizer


# ------------------------------------------------------
# Config
# ------------------------------------------------------
MODEL_ID = "./repository"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DTYPE = torch.bfloat16 if torch.cuda.is_available() else torch.float32
MAX_NEW_TOKENS=512

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ------------------------------------------------------
# Model Manager
# ------------------------------------------------------
class ModelNotLoadedError(RuntimeError):
    """Raised when attempting to use the model before it is loaded."""


class ModelManager:
    def __init__(self, model_id: str, device: str, dtype: torch.dtype):
        self.model_id = model_id
        self.device = device
        self.dtype = dtype

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None

    async def load(self):
        """Load model + tokenizer if not already loaded."""
        if self.model is not None and self.tokenizer is not None:
            return

        start = time.perf_counter()
        logger.info("Loading tokenizer and model for %s", self.model_id)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
        )
        self.model = (
            AutoModelForCausalLM.from_pretrained(
                self.model_id,
                dtype=self.dtype,
            )
            .to(self.device)
            .eval()
        )
        duration_ms = (time.perf_counter() - start) * 1000
        logger.info("Finished loading %s in %.2fms", self.model_id, duration_ms)

    async def unload(self):
        """Free model + tokenizer and clear CUDA cache."""
        if self.model is not None:
            self.model.to("cpu")
            del self.model
            self.model = None

        if self.tokenizer is not None:
            del self.tokenizer
            self.tokenizer = None

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get(self):
        """Return the loaded model + tokenizer or raise if not ready."""
        if self.model is None or self.tokenizer is None:
            raise ModelNotLoadedError("Model not loaded")
        return self.model, self.tokenizer


model_manager = ModelManager(MODEL_ID, DEVICE, DTYPE)


# ------------------------------------------------------
# Lifespan (startup + shutdown)
# ------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    await model_manager.load()
    try:
        yield
    finally:
        await model_manager.unload()


app = FastAPI(lifespan=lifespan)


# ------------------------------------------------------
# Schemas
# ------------------------------------------------------
class GenerateRequest(BaseModel):
    prompt: str = Field(..., min_length=1, description="Plain-text prompt")
    max_new_tokens: int = Field(
        128,
        ge=1,
        le=MAX_NEW_TOKENS,
        description="Upper bound on generated tokens",
    )

class GenerateResponse(BaseModel):
    response: str
    input_token_count: int
    output_token_count: int

# ------------------------------------------------------
# Routes
# ------------------------------------------------------
@app.get("/health")
def health():
    try:
        model_manager.get()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    return {"message": "API is running."}


@app.post("/generate", response_model=GenerateResponse)
def generate(request: GenerateRequest) -> GenerateResponse:
    start_time = time.perf_counter()
    try:
        model, tokenizer = model_manager.get()
    except ModelNotLoadedError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    if getattr(tokenizer, "chat_template", None):
        messages = [{"role": "user", "content": request.prompt}]
        input_text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        input_text = request.prompt

    inputs = tokenizer(input_text, return_tensors="pt").to(DEVICE)

    try:
        with torch.inference_mode():
            outputs = model.generate(**inputs, max_new_tokens=request.max_new_tokens)
    except RuntimeError as exc:
        logger.exception("Generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}") from exc

    input_token_count = inputs.input_ids.shape[1]
    generated_ids = outputs[0][input_token_count:]
    generated_text = tokenizer.decode(generated_ids, skip_special_tokens=True)
    output_token_count = generated_ids.shape[0]
    duration_ms = (time.perf_counter() - start_time) * 1000

    logger.info(
        "generate prompt_tokens=%d new_tokens=%d max_new_tokens=%d duration_ms=%.2f",
        input_token_count,
        output_token_count,
        request.max_new_tokens,
        duration_ms,
    )

    return GenerateResponse(
        response=generated_text,
        input_token_count=input_token_count,
        output_token_count=output_token_count,
    )
```
</details>

------


## 2. Build the Docker image

Now let's create a `Dockerfile` to package our server into a container.

> 💡 **Remember: model weights shouldn't be baked into the image**: Inference Endpoints will mount
> the selected model at `/repository`, so the image only needs your **code** and **dependencies**.

We’ll also avoid running as `root` inside the container by creating a non-root user and granting it access to `/app`.

First if your uv project doesn't have a lockfile, which is common if you just created it, we can manually tell uv to make one for us by running:
```bash
uv lock
```

Our Dockerfile will otherwise be very standard:
1. We us the base pytorch image with CUDA and cuDNN
2. We copy the uv binary
3. Make sure that we're not running things as a priviledged user
4. Install the depencencies with uv
5. Make sure that we expose the correct port
6. Run the server

```Dockerfile
FROM pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime

# Install uv by copying the static binary from the distroless image.
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv
COPY --from=ghcr.io/astral-sh/uv:latest /uvx /bin/uvx

ENV USER=appuser HOME=/home/appuser
RUN useradd -m -s /bin/bash $USER

WORKDIR /app

# Ensure uv uses the bundled venv
ENV VIRTUAL_ENV=/app/.venv
ENV PATH="/app/.venv/bin:${PATH}"

# Copy project metadata first (better caching)
COPY pyproject.toml uv.lock ./

# Create the venv up front and sync dependencies (no source yet for better caching)
RUN uv venv ${VIRTUAL_ENV} \
    && uv sync --frozen --no-dev --no-install-project

# Copy the rest of the application code
COPY . .

# Re-sync to capture the project itself inside the venv
RUN uv sync --frozen --no-dev

RUN chown -R $USER:$USER /app

USER $USER

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
```

<details>
<summary>
It's also a good idea to include a `.dockerignore` file to make sure we're not copy pasting irrelevant file. Since it's quite verbose we won't go in detail through all the parts there. But please copy it into your working directory.
</summary>
</details>

## 3. Build and Push the Image

Once your `Dockerfile` and `main.py` are ready, build the container and push it
to a registry that Inference Endpoints can access (Docker Hub, Amazon ECR, Azure ACR, or Google GCR).

```bash
docker build -t your-username/smollm-endpoint:v0.1.0 . --platform linux/amd64
docker push your-username/smollm-endpoint:v0.1.0
```
> 🤔 **Why `--platform linux/amd64`?**: if you're building this image on a Mac, it will automatically be built for an arm64 machine, which is not what's the architecture the machines in Inference Endpoints have. That's why we need this flag to tell that we're targeting x86.
> If you're on a x86 machine already, you can ignore this flag.

## 4. Create the Endpoint

Now switch to the Inference Endpoints UI and deploy your custom container.

1. Open the [Inference Endpoints dashboard](https://endpoints.huggingface.co/) and click **"+ New"**.
  ![endpoint-new.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/endpoint-new.png)

2. Select **`HuggingFaceTB/SmolLM3-3B`** as the model repository (this will be
  mounted at `/repository` inside the container).
  ![choose-smollm.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/choose-smollm.png)

3. Click **“Configure”** to proceed with the deployment setup.
  ![configure.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/configure.png)

4. This is the configuration page where you’ll define compute, networking, and container settings.
  ![home.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/home.png)

5. Choose the hardware.
  For this example, a **T4 GPU** is sufficient.
  ![authenticated.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/authenticated.png)

6. Under **Custom Container**, enter:
  * your image URL (e.g., `your-username/smollm-endpoint:v0.1.0`)
  * the port exposed by your container (e.g., `8000` or whatever you used in `CMD`)
    ![custom.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/custom.png)

7. Click **“Create Endpoint”**.
  The platform will:

  * pull your container image
  * mount the model at `/repository`
  * start your FastAPI server
    ![initializing.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/initializing.png)

8. After a short initialization period, the status will change to **Running**.
  Your custom container is now serving requests.
  ![running.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/running.png)

Once deployed, your endpoint will be available at a URL like:

```bash
https://random-number.region.endpoints.huggingface.cloud/
```

Below is a minimal Python client you can use to test it:

```python
from huggingface_hub import get_token
import requests

url = "https://random-number.region.endpoints.huggingface.cloud/generate"

prompt = "What is an Inference Endpoint?"
data = {"prompt": prompt, "max_new_tokens": 512}


response = requests.post(
    url=url,
    json=data,
    headers={
        "Authorization": f"Bearer {get_token()}",
        "Content-Type": "application/json",
    },
).json()

print(f"Input:\n{prompt}\n\nOutput:\n{response['response']}")

```

If you open the **Logs** tab of your endpoint, you should see the incoming POST request and the model’s response.

![post.png](https://raw.githubusercontent.com/huggingface/hf-endpoints-documentation/main/assets/custom_container/post.png)

```
Input:
What is an Inference Endpoint?

Output:
<think>
Okay, so I need to ...
```

Congratulations for making it until the end 🎉 Good ideas to extend this demo would be to test out with a completely different model, say an audio model or image generaion one, happy hacking 🙌