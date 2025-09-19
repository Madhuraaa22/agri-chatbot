from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

@app.route("/")
def home():
    return "Server is running!"

@app.route("/chat", methods=["POST"])
def chat():
    data = request.get_json(force=True)  # ✅ force parsing JSON
    if not data:
        return jsonify({"response": "Invalid request"}), 400

    user_message = data.get("message", "")
    bot_reply = f"You said: {user_message}"  # replace with model
    return jsonify({"response": bot_reply})

if __name__ == "__main__":
   app.run(host="0.0.0.0", port=5000, debug=True)

