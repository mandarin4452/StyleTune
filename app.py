from flask import Flask, render_template, request, Response, redirect
from werkzeug import secure_filename
import os
import time
import requests
import json


#IMAGE_FOLDER = os.path.join('static','style')

app = Flask(__name__)
file_name = ""
style = ""

@app.route('/')
def index():
    title = "Main"
    detail = "Select one of your images to transfer"
    return render_template("index.html",page_title = title, page_info = detail)

@app.route('/style', methods =['GET','POST'])
def styler():
    title = "Style"
    detail = "Choose one of the styles given below"

    if request.method == "POST":
        f = request.files['file']
        print(f)
        source = "../static/Images/" + str(f.filename)
        if f:
            f.save("./static/Images/" + secure_filename(f.filename))
            print("Image saved")
            global file_name
            file_name = f.filename
            return render_template('style.html',page_title = title, page_info = detail, name = f.filename)
        else:
            return redirect("/")

@app.route('/transfer/<style_type>')
def transfer(style_type):
    global style
    style = str(style_type)
    fileName = str(file_name)
    title = "Transfer"
    detail = "Image transferring"
    return render_template('transfer.html', page_title = title, page_info = detail)
        

@app.route('/progress')
def progress():
    def generate():
        x = 0
        process = ""
        image_url = ""
        temp  = ""
        while x <= 100:
            if x is 0:
                process = "Training Content Image."
            elif x is 10:
                process = "Training Content Image.."
            elif x is 20 :
                process = "Training Content Image..."
            elif x is 30:
                
                r = requests.post(
                    "https://api.deepai.org/api/fast-style-transfer",
                    files = {
                        'content' : open('./static/Images/' + str(file_name), 'rb'),
                        'style':open('./static/style_preset/' + str(style) + ".jpg", 'rb') 
                    },
                    headers={'api-key': '1ef3aab4-cb2d-4153-902b-8bebb1b14844'}
                )
                print(r.json())
                temp = r.json()["output_url"]
                

                process = "Transfer..."
                x += 30
            elif x is 70:
                process = "Your Image will be ready soon."
            elif x is 80:
                process = "Your Image will be ready soon.."
            elif x is 90:
                process = "Your Image will be ready soon..."
            else:
                process = "Done!"
                image_url = temp
            
            yield "data:" + str(x) + "*" + process + "*" +image_url + "\n\n" 
            x += 10
            time.sleep(1)

    return Response(generate(), mimetype = 'text/event-stream')
if __name__ == "__main__":
    app.run(host = '0.0.0.0', port = 5000, debug = True)
