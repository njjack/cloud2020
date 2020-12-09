from flask import Flask, render_template, request, jsonify, session, redirect
import json
import pymongo
from functools import wraps

app = Flask(__name__)
app.secret_key = b'\xa4tt\x94\x7f&\x9d>C\xe4\xbc\x8cU\xdc-\xc2'

client = pymongo.MongoClient('localhost',27017 )
db = client.user_login_system

def login_required(f):
  @wraps(f)
  def wrap(*args, **kwargs):
    if 'logged_in' in session:
      return f(*args, **kwargs)
    else:
      return redirect('/')
  
  return wrap

from user import routes

@app.route('/')
def home():
  return render_template('home.html')

@app.route('/dashboard/')
@login_required
def dashboard():
  return render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
