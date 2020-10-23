from flask import Flask, render_template, request, jsonify
import json
from crawl import crawl

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('test.html')

@app.route('/do', methods=["GET", "POST"])
def do():
    target = int(request.form.get("twa"))
    result = dict()
    twa = []
    title = []
    url = []
    twa, title, url = crawl(target)
    result["twa"] = twa
    result["title"] = title
    result["url"] = url
    print(result)
    return jsonify(result)

if __name__ == '__main__':
    app.run()
#render_template將會找尋html檔案傳送給使用者
