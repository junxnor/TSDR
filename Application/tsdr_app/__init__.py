from flask import Flask

def create_app() -> Flask:
    """Create and configure the Flask application."""
    app = Flask(__name__, template_folder="../templates", static_folder="../static")

    from .routes import web

    app.register_blueprint(web)
    return app