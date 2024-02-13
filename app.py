from flask import Flask, render_template

app = Flask("FRAMS")

@app.route('/')
def index():
    return render_template('start_screen.html')


@app.route('/register_screen')
def register_screen():
    return render_template('register_screen.html')

@app.route('/attendence_screen')
def attendence_screen():
    return render_template('attendence_screen.html')

@app.route('/open-webcam')
def open_webcam():
    
    return 'Webcam opened successfully.'

if __name__ == '__main__':
    app.run(debug=True)


    