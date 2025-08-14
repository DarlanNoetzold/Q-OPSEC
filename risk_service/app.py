from flask import Flask, jsonify
from controllers.risk_controller import risk_bp

def create_app():
    app = Flask(__name__)
    app.register_blueprint(risk_bp)

    @app.get("/health")
    def health():
        return jsonify({"status": "ok"}), 200

    return app

app = create_app()

if __name__ == "__main__":
    # Run: python app.py
    app.run(host="0.0.0.0", port=8082, debug=True)