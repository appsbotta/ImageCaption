import streamlit as st
import streamlit.components.v1 as components
import tensorflow as tf
from keras.preprocessing.text import Tokenizer 
from keras.utils import pad_sequences
import pickle
import numpy as np
from keras.models import Model,load_model
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import load_img, img_to_array


tokenizer = pickle.load(open("tokens.pkl","rb"))
vocab_size = len(tokenizer.word_index) + 1
model = load_model("best_model.h5")
max_length = 35



def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length):
    # add start tag for generation process
    in_text = 'startseq'
    # iterate over the max length of sequence
    for i in range(max_length):
        # encode input sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        # pad the sequence
        sequence = pad_sequences([sequence], max_length)
        # predict next word
        yhat = model.predict([image, sequence], verbose=0)
        
        # get index with high probability
        yhat = np.argmax(yhat)
        # convert index to word
        word = idx_to_word(yhat, tokenizer)
        # stop if word not found
        if word is None:
            break
        # append word as input for generating next word
        in_text += " " + word
        # stop if we reach end tag
        if word == 'endseq':
            break
      
    return in_text



def homepage():
   
    # st.markdown("<h1 style='text-align: center; color: White;'>Image Caption Generator</h1>", unsafe_allow_html=True)
    components.html(
    """
        <!DOCTYPE html>
        <html>
        <head>
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
        * {box-sizing: border-box;}
        body {
            font-family: Verdana, sans-serif;
            margin:none;
        }
        .mySlides {display: none;}
        img {vertical-align: middle;}

        /* Slideshow container */
        .slideshow-container {
        max-width: 500px;
        position: relative;
        margin: auto;
        }

        /* Caption text */
        .text {
        color: white;
        font-size: 15px;
        padding: 8px 12px;
        position: absolute;
        bottom: 8px;
        width: 100%;
        text-align: center;
        }

        /* Number text (1/3 etc) */
        .numbertext {
        color: #f2f2f2;
        font-size: 12px;
        padding: 8px 12px;
        position: absolute;
        top: 0;
        }

        /* The dots/bullets/indicators */
        .dot {
        height: 15px;
        width: 15px;
        margin: 0 2px;
        background-color: white;
        border-radius: 50%;
        display: inline-block;
        transition: background-color 0.6s ease;
        }

        .active {
        background-color: #717171;
        }

        /* Fading animation */
        .fade {
        animation-name: fade;
        animation-duration: 1.5s;
        }

        @keyframes fade {
        from {opacity: .4} 
        to {opacity: 1}
        }

        /* On smaller screens, decrease text size */
        @media only screen and (max-width: 300px) {
        .text {font-size: 11px}
        }
        </style>
        </head>
        <body>
        <b><h1 style="text-align: center; color: White;font-family: Noto Sans and Noto CJK;">Image Caption Generator</h1></b>
        <p style="color:white;margin:0 0 0 0;">Examples of Image Caption generator </p>
        <div class="slideshow-container">

        <div class="mySlides fade">
            <div class="numbertext">1 / 3</div>
            <img class="image" src="https://unsplash.com/photos/2w5SEiEImJc/download?force=true&w=1920" style="width:110% ">
            <div class="text">man lays on the bench while leashed dog on the ground</div>
        </div>

        <div class="mySlides fade">
            <div class="numbertext">2 / 3</div>
            <img src="https://unsplash.com/photos/iH6uKNdT2vw/download?force=true&w=1920" style="width:110%">
            <div class="text">bird is running through field</div>
        </div>

        <div class="mySlides fade">
            <div class="numbertext">3 / 3</div>
            <img src="https://unsplash.com/photos/bEcC0nyIp2g/download?force=true&w=1920" style="width:110%">
            <div class="text">group of people fly on the sand</div>
        </div>


        </div>
        <br>

        <div style="text-align:center">
            <span class="dot"></span> 
            <span class="dot"></span> 
            <span class="dot"></span> 
        </div>

        <script>
        let slideIndex = 0;
        showSlides();

        function showSlides() {
        let i;
        let slides = document.getElementsByClassName("mySlides");
        let dots = document.getElementsByClassName("dot");
        for (i = 0; i < slides.length; i++) {
            slides[i].style.display = "none";  
        }
        slideIndex++;
        if (slideIndex > slides.length) {slideIndex = 1}    
        for (i = 0; i < dots.length; i++) {
            dots[i].className = dots[i].className.replace(" active", "");
        }
        slides[slideIndex-1].style.display = "block";  
        dots[slideIndex-1].className += " active";
        setTimeout(showSlides, 2000); // Change image every 2 seconds
        }
        </script>
        </body>
        </html> 

            """,
            height=600,
        )



def main():
    nav = st.sidebar.radio("Navigation",["Home","Caption prediction"])
    if nav=="Home":
        homepage()
    elif nav == "Caption prediction":
        img = st.file_uploader("Upload Image",)
        if img:
            st.image(img)
            vgg_model = VGG16()
            vgg_model = Model(inputs=vgg_model.inputs,outputs = vgg_model.layers[-2].output)
            img = load_img(img,target_size=(224,224))
            img = img_to_array(img)
            img = np.expand_dims(img,axis=0)
            img = preprocess_input(img)
            feature = vgg_model.predict(img,verbose=0)
            ans = predict_caption(model,feature,tokenizer,max_length)
            ans = ans.split(" ")
            l = len(ans)
            ans = ans[1:l-1]
            ans = " ".join(ans)
            st.success(ans)
        


if __name__ == '__main__':
    main()