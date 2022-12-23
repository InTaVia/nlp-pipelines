from flask import Flask
from routes import en_blueprint, naf_converter, srl_predictor

app = Flask(__name__)
app.register_blueprint(en_blueprint)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
