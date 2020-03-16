import os
from flask import Flask, flash, request, redirect, url_for, render_template
# from flask import Session
from flask import *  
from flask_session.__init__ import Session
from werkzeug.utils import secure_filename

UPLOAD_FOLDER = ' '
ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

sess = Session()
@app.route('/upload')
def upload():
   return render_template('index.html')

@app.route('/uploader', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            flash('No file part')
            # return redirect(request.url)
            return "no file part"
        file = request.files['file']
        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            flash('No selected file')
            return "no selected file"
        if file:
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
            # return redirect(url_for('uploaded_file', filename=filename))
            return "uploaded"
    return "only POST"   
		
if __name__ == '__main__':
    app.secret_key = 'super secret key'
    app.config['SESSION_TYPE'] = 'filesystem'

    # sess.init_app(app)
    app.run(debug = True)