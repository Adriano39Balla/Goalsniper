import logging
from app import app

if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s'
    )
    app.run(host='0.0.0.0', port=5000, debug=app.config.get("DEBUG", False))
