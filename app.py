from flask import Flask,request,render_template,jsonify,send_from_directory
import os
import face_recognition
import pickle

with open('face_encodings.pkl', 'rb') as f:
    stored_encodings = pickle.load(f)
    
app=Flask(__name__)
img_folder="photo/"

app.config["img_folder"]=img_folder

if not os.path.exists(img_folder):
       
    os.makedirs("photo",exist_ok=True)
 
 

def process_image(image):
    matches=''
    img=face_recognition.load_image_file(image)
    new_face_encoding=face_recognition.face_encodings(img)
    if new_face_encoding:
        new_face_encoding=new_face_encoding[0]
        match_found=False
        for photo_file,encodings in stored_encodings.items():
            results=face_recognition.compare_faces(encodings,new_face_encoding,tolerance=0.6)
            if any(results):
                match_found=True
                
                matches+=f"match found in stored photo: {photo_file}\n"
                
        if not match_found:
            return "No matching face found in any stored photo."
    return matches

    
    
    
    
   

@app.route('/upload',methods=["GET","POST"])
def home():
    if not 'image' in request.files:
        return jsonify({"message":"upload a valid file"})
    
    if 'image' in request.files:
        file=request.files['image']
        image_path=os.path.join(app.config["img_folder"],file.filename)
        file.save(image_path)
        matches=process_image(image_path)
        print(matches)
        os.remove(image_path)
        
        return jsonify({"message":matches})
    return jsonify({"messsage":"please upload image"})
if __name__=="__main__":
    app.run(debug=True,host="0.0.0.0")