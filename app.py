from flask import Flask,render_template,request, redirect, Response
from werkzeug import secure_filename
import requests
import time

app = Flask(__name__)
file_name = ""
style = ""
@app.route('/')
def index():
    return render_template('index.html',nav_home = "active")

@app.route('/information')
def info():
    return render_template('intro.html',nav_info = "active")

@app.route('/transfer')
def transfer_home():
    return render_template('transfer.html',nav_transfer = "active")

@app.route('/transfer/style', methods =['GET','POST'])
def style():
    if request.method == "POST":
        f = request.files['file']
        print(f.filename)
        if f:
            f.save("./static/images/" + secure_filename(f.filename))
            print("Image saved")
            global file_name
            file_name = f.filename
            return render_template('style.html',nav_transfer = "active")
        else:
            return redirect("/transfer")

@app.route('/transfer/style/<style_type>')
def transfer(style_type):
    global style
    style = str(style_type)
    fileName = str(file_name)
    return render_template('transfer_final.html', nav_transfer = "active")


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
    app.run(host = '0.0.0.0')