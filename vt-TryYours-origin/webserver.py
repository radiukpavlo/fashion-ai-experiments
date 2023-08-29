"""
webserver.py
"""
from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
import os

app = Flask(__name__)
run_with_ngrok(app)

data_list = []


@app.route('/', methods=['GET', 'POST'])
def main():
    """
    :return:
    """
    return render_template('main.html')


@app.route('/fileUpload', methods=['GET', 'POST'])
def file_upload():
    """
    :return:
    """
    if request.method == 'POST':
        f = request.files['file']
        f_src = 'static/origin_web.jpg'

        f.save(f_src)
        return render_template('fileUpload.html')


@app.route('/fileUpload_cloth', methods=['GET', 'POST'])
def fileUpload_cloth():
    """
    :return:
    """
    if request.method == 'POST':
        f = request.files['file']
        f_src = 'static/cloth_web.jpg'

        f.save(f_src)
        return render_template('fileUpload_cloth.html')


@app.route('/view', methods=['GET', 'POST'])
def view():
    """
    :return:
    """
    print("inference start")

    terminnal_command = "python main.py"
    os.system(terminnal_command)

    print("inference end")
    # Rendering html and transferring values from the database
    return render_template('view.html', data_list=data_list)


if __name__ == '__main__':
    app.run()
