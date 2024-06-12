import sys
# sys.path.append("/Users/user/question-retrieval-KIPerWeb/")

from utils import *
from flask import Flask, jsonify, request, Response, make_response, send_from_directory
from flask_cors import CORS, cross_origin
from flask_httpauth import HTTPBasicAuth
from flask_sqlalchemy import SQLAlchemy, session
from sentence_transformers import SentenceTransformer
import pandas as pd
from werkzeug.security import generate_password_hash, check_password_hash
import jwt
from functools import wraps
import uuid # for public id
from datetime import datetime, timedelta
from base64 import b64decode
import pytz
import json
import configparser
import os


# -----------  Configuration  ----------------
current_path = '/Users/user/EdTec-QBuilder'
config = configparser.ConfigParser()
ini_path = current_path+"/config.ini"
# ini_path = os.path.join(os.getcwd(),'config.ini')# this is without VSCODE
print(ini_path)
config.read(ini_path)


app = Flask(__name__, static_url_path='/static', instance_path=current_path)
app.app_context().push()
cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type: application/json; charset=utf-8'
# app.config['CORS_HEADERS'] = 'Content-Type'

@app.route('/login.html')
def login_page():
    return send_from_directory(app.static_folder, 'login.html')


@app.route('/index.html')
def index_page():
    return send_from_directory(app.static_folder, 'index.html')

@app.route('/signup.html')
def signup_page():
    return send_from_directory(app.static_folder, 'signup.html')


# -----------  Data Base  ----------------
# configuration
# NEVER HARDCODE YOUR CONFIGURATION IN YOUR CODE
# INSTEAD CREATE A .env FILE AND STORE IN IT
#TODO make sure that all the data is not leaked
app.config['SECRET_KEY'] = 'your secret key'
# database name
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///Database.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = True
# creates SQLALCHEMY object
db = SQLAlchemy(app)



# Database ORMs
class User(db.Model):
    id = db.Column(db.Integer, primary_key = True)
    public_id = db.Column(db.String(50), unique = True)
    name = db.Column(db.String(100))
    email = db.Column(db.String(70), unique = True)
    password = db.Column(db.String(80))
    interactions = db.Column(db.String(1000))
  
with app.app_context():
    db.create_all()

# -----------  Authentification  ----------------

# decorator for verifying the JWT
def token_required(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        token = None
        # jwt is passed in the request header
        print("--> X-Access-Token:", request.headers['X-Access-Token'])
        if 'X-Access-Token' in request.headers:
            token = request.headers['X-Access-Token']
            # print('->token', token)
        # return 401 if token is not passed
        if not token:
            return jsonify({'message' : 'Token is missing !!'}), 403
        try:
            # decoding the payload to fetch the stored details
            print("token", token)
            print("SECRET KEY", app.config['SECRET_KEY'])
            data = jwt.decode(token, app.config['SECRET_KEY'], algorithms=["HS256"])
            print(data)
            current_user = User.query\
                .filter_by(public_id = data['public_id'])\
                .first()
            
            
        except:
            return jsonify({
                'message' : 'Token is invalid !!'
            }), 403
        # returns the current logged in users context to the routes
        return  f(current_user, *args, **kwargs)
  
    return decorated

# route for logging user in
@app.route('/login',methods=['GET','POST'])
@cross_origin()
def login():
    # creates dictionary of form data
    auth = request.get_json()
    # print(auth)
    if not auth or not auth.get('email') or not auth.get('password'):
        # returns 401 if any email or / and password is missing
        return make_response(
            'Could not verify',
            403,
            {'WWW-Authenticate' : 'Basic realm ="Login required !!"'}
        )
  
    user = User.query\
        .filter_by(email = auth.get('email'))\
        .first()
    
    if not user:
        # returns 401 if user does not exist
        return make_response(
            'Could not verify',
            403,
            {'WWW-Authenticate' : 'Basic realm ="User does not exist !!"'}
        )
  
    if check_password_hash(user.password, auth.get('password')):
        # generates the JWT Token
        token = jwt.encode({
            'public_id': user.public_id,
            'exp' : datetime.utcnow() + timedelta(minutes = 4320)
        }, app.config['SECRET_KEY'], algorithm="HS256")
  
        return make_response(jsonify({'token' : token, 'user_id':user.public_id}), 201)
    # returns 403 if password is wrong
    return make_response(
        'Could not verify',
        403,
        {'WWW-Authenticate' : 'Basic realm ="Wrong Password !!"'}
    )
  
# signup route
@app.route('/signup', methods=['GET','POST'])
@cross_origin()
def signup():
    # creates a dictionary of the form data
    data = request.get_json()
    # print(data)
    # gets name, email and password
    name = data['name']
    email = data['email']
    password = data['password']
    # checking for existing user
    user = User.query\
        .filter_by(email = email)\
        .first()
    if not user:
        # database ORM object
        user = User(
            public_id = str(uuid.uuid4()),
            name = name,
            email = email,
            password = generate_password_hash(password)
        )
        # insert user
        db.session.add(user)
        db.session.commit()
  
        return make_response('Successfully registered.', 200)
    else:
        # returns 202 if user already exists
        return make_response('User already exists. Please Log in.', 202)


# User Database Route
# this route sends back list of users
@app.route('/user', methods =['GET'])
@token_required
@cross_origin()
def get_all_users(current_user):
    # querying the database
    # for all the entries in it
    users = User.query.all()
    print(users)
    # converting the query objects
    # to list of jsons
    output = []
    for user in users:
        # appending the user data json
        # to the response list
        output.append({
            'public_id': user.public_id,
            'name' : user.name,
            'email' : user.email
        })
  
    return jsonify({'users': output})




# -----------  Config Files  -------------------

# print(list(config))
# data = config.get('SEARCH_UTILITIES','data')
# nmslib_index_path = config.get('SEARCH_UTILITIES','nmslib_index')
# lang_model_path = config.get('SEARCH_UTILITIES', 'lang_model')

# df = preprocess_json_data(data)
# index = load_nmslib_index(nmslib_index_path)
# lang_model = SentenceTransformer(lang_model_path)

data = config.get('SEARCH_UTILITIES','data')
nmslib_index_path = config.get('SEARCH_UTILITIES','nmslib_index')

# print(nmslib_index_path)
lang_model_path = config.get('SEARCH_UTILITIES', 'lang_model')

# df = preprocess_json_data(data)
df = pd.read_csv(data).fillna("N/A")
df['points'] = df['points'].fillna('N/A')
df['onlinetest_title'] = df['onlinetest_title'].fillna('N/A')
df['question_type_id'] = df['question_type_id'].fillna('N/A')
df['question_type_name'] = df['question_type_name'].fillna('N/A')
df['variable'] = df['variable'].fillna('N/A')
df['correct_answers_temp'] = df['correct_answers_temp'].fillna('N/A')



# print(list(df.columns))

index = load_nmslib_index(nmslib_index_path)
lang_model = SentenceTransformer(lang_model_path)

from pprint import pprint
# -----------  API Functionality  -------------------
# TODO: when add the token requiemrents
@app.route('/retrieve',  methods=['POST'])
@cross_origin()
@token_required
def retrieve(current_user):
    raw_req = request.get_json()
    query = raw_req['q'].lower()
    search_output = search(query, index, lang_model, df, k=10)
    # print(search_output)
    response = jsonify(search_output)
    return response


@app.route('/response', methods=['POST'])
@cross_origin()
@token_required
def get_response(current_user):
    user_response = request.get_json()
    print('Recieved from client: {}'.format(user_response))
    response = jsonify(user_response)
    tz = pytz.timezone('Europe/Berlin')
    timestamp = datetime.now(tz)
    #TODO: check if theres a better way of adding rows to the DB

    users = User.query.all()

    for user in users:
        print("public_id",user.public_id)
    
    
    #TODO: here I would need to create another table with this schema:
    # db.session.add({"id":user.public_id, "timestamp": [timestamp.strftime("%Y-%b-%d, %A %I:%M:%S")], "session":[request.get_json()]})
    # db.session.add({"id":user.public_id, "timestamp": [timestamp.strftime("%Y-%b-%d, %A %I:%M:%S")], "session":[request.get_json()]})

    #TODO: improve the timestamp format, would be nice to have date hashes
    pd.DataFrame({"id":user.public_id, "timestamp": [timestamp.strftime("%Y-%b-%d, %A %I:%M:%S")],"session":[request.get_json()]}).\
        to_csv("./data/sessions_log.csv", index=False, mode='a', header=False)
    
    return response

# TODO: implement
# @app.route('/visualize', methods=['POST'])
# @cross_origin()
# @token_required
# def visualize(current_user):
#     #user_response = request.get_json()
#     # R will send the ids of the questions
#     # R will send the query
#     # from the original DF, we filter the ids, and encode them into a doc embedding
#     document_set_embeddings = [model.encode(e) for e in df['content'].tolist()]

#     print("Implement") # if receive the viz request send embeddings

if __name__ == "__main__":
    # app.run(port=8000, debug=True)
    app.run(port=8000, debug=True)
