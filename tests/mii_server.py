import mii

MODEL_NAME = "google/gemma-3-27b-it"
DEPLOYMENT_NAME = "gemma-3-27b-it"
PORT = 8000

if __name__ == "__main__":
    print(f"[MII SERVER] Launching {MODEL_NAME} on port {PORT}...")
    mii.serve(
        model_name_or_path=MODEL_NAME,  # <-- FIXED ARGUMENT NAME
        deployment_name=DEPLOYMENT_NAME,
        host="0.0.0.0",
        port=PORT
    )
    # mii.serve is blocking and will keep the server alive until interrupted