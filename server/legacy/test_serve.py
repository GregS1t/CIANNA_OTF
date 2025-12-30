from flask import Flask, send_from_directory, abort
import os

app = Flask(__name__)

@app.route("/models/<path:filename>")
def serve_models(filename):
    models_dir = os.path.abspath("server/models")
    full_path = os.path.join(models_dir, filename)

    print("[DEBUG] full_path:", full_path)
    if not os.path.exists(full_path):
        print("[ERROR] File not found.")
        return abort(404)

    return send_from_directory(models_dir, filename)

if __name__ == "__main__":
    app.run(port=5050, debug=True)

