from flask import Flask, render_template ,request
import eva as e

app = Flask(__name__)

@app.route('/')
def login():
    return render_template('login.html')

@app.route('/index/')
def index():
    return render_template('index.html')
    
@app.route('/task/')
def task():
    return render_template('task.html')
    
@app.route('/competitions/')
def competitions():
    return render_template('competitions.html')

@app.route('/results/')
def results():
    return render_template('results.html')


@app.route('/eve',methods = ['POST', 'GET'])
def eve():
    if request.method == 'POST':
      txt = request.form['name']
      return render_template('results.html',output=e.run(txt))

@app.route('/practice/')
def practice():
    return render_template('practice.html')


if __name__ == '__main__':
    app.run(debug=True)