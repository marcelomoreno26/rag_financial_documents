import uvicorn
from argparse import ArgumentParser
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from json import JSONDecodeError
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


app = FastAPI()

DEFAULT_PARAMS = {
            "max_tokens":512, 
            "top_p":0.8, 
            "temperature":0.1, 
            "min_p":0.1
        }




@app.get("/api_active/")
async def is_api_active():
    return {"active": True}


@app.post("/generate/")
async def generate_prediction(request: Request):
    """
    Handles text generation requests.

    Parameters
    ----------
    request : Request
        Incoming HTTP request containing:
        - "messages" (list): Chat messages for generation.
        - "sampling_params" (dict, optional): Parameters for text generation.

    Returns
    -------
    dict
        JSON response with success status and generated predictions or an error message.
    """
    try:
        data = await request.json()
        messages = data.get("messages")
        if not messages:
            return JSONResponse(content={"success": False, "error": "Missing 'messages' key."}, status_code=400)
        
            
        messages = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        if data.get("sampling_params"):
            sampling_params = SamplingParams(**data.get("sampling_params"))
        else:
            sampling_params = SamplingParams(**DEFAULT_PARAMS)

        outputs = model.generate(messages, sampling_params=sampling_params)
        
        if not outputs:
            return JSONResponse(content={"success": False, "error": "No output from model."}, status_code=500)

        predictions = [output.outputs[0].text for output in outputs if output.outputs]

        return {"success": True, "predictions": predictions}

    except JSONDecodeError:
        return JSONResponse(content={"success": False, "error": "Invalid JSON in request."}, status_code=400)

    except KeyError as e:
        return JSONResponse(content={"success": False, "error": f"Missing key: {str(e)}."}, status_code=400)

    except RuntimeError as e:
        return JSONResponse(content={"success": False, "error": f"Text generation error: {str(e)}."}, status_code=500)

    except Exception as e:
        return JSONResponse(content={"success": False, "error": f"Unexpected error: {str(e)}."}, status_code=500)




if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True, help="Hugging Face model name or path")
    parser.add_argument("--tokenizer", type=str, required=False, help="Hugging Face model name or path for the tokenizer in case it is different than the model used.")
    parser.add_argument("--port", type=int, required=False, default=8000, help="Port in which to run the API")
    parser.add_argument("--max_model_len", type=int, required=False, default=2048, help="Max model token length.")
    parser.add_argument("--gpu_memory_utilization", type=float, required=False, default=0.9, help="GPU memory utilization with vLLM. Useful if multiple models are needed to balance load.")

    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer if args.tokenizer else args.model_name)
    model = LLM(args.model_name, max_model_len=args.max_model_len, gpu_memory_utilization=args.gpu_memory_utilization)
    uvicorn.run(app, host="0.0.0.0", port=args.port)