from flask import Flask
from app.routes import main


def create_app():
    app = Flask(__name__, static_folder='static', template_folder='templates')
    app.secret_key = 'dev'  # needed for flashing messages

    app.register_blueprint(main)

    return app
