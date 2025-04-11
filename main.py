import sys
import os
# sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from app import create_app
from app.routes import main  # add this line

app = create_app()
# app.register_blueprint(main)

if __name__ == "__main__":
    app.run(debug=True)
