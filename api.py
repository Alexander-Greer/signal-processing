import flask
from flask import request, jsonify
from flask_cors import CORS

app = flask.Flask(__name__)
cors = CORS(app, resources={r"/api/*": {"origins": "*"}})
app.config["DEBUG"] = True
app.config["TEMPLATES_AUTO_RELOAD"] = True

# Create some test data for our catalog in the form of a list of dictionaries.
books = [
    {'id': 0,
     'title': 'A Fire Upon the Deep',
     'author': 'Vernor Vinge',
     'first_sentence': 'The coldsleep itself was dreamless.',
     'year_published': '1992'}
]


@app.route('/', methods=['GET'])
def home():
    return '''<h1>dAnK FoRdieR TrabSfoRm ApI</p>'''


# Send the API the original audio file and process the file
@app.route('/api/send/', methods=['POST'])
def api_process():
    print(request.form)
    return ""


app.run()
