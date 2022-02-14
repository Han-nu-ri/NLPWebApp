import os
from flask import Flask, jsonify, request
import analyzer


def get_prediction(sentence):
    sentiment_analyzer = analyzer.SentimentAnalyzer()
    # TODO: test 종료 후 삭제하기
    print(sentence)
    return sentiment_analyzer.predict([sentence])[0]



def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'app.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

    @app.route('/predict', methods=['POST'])
    def predict():
        if request.method == 'POST':
            params = request.get_json()
            sentence = params['sentence']
            # TODO: test 종료 후 삭제하기
            print(sentence)
            sentiment_result = get_prediction(sentence)
            # TODO: test 종료 후 삭제하기
            print(sentiment_result)
            return jsonify({'sentence': sentence, 'sentiment_result': sentiment_result})

    return app

app = create_app(test_config=None)

if __name__ == '__main__':
    app.run()
