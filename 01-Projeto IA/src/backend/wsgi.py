"""WSGI entrypoint for the Flask app.

Keeps compatibility with the current root-level app module while the
modular refactor is in progress.
"""

from app import app


if __name__ == "__main__":
    app.run(debug=True)
