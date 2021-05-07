import streamlit as st
from PIL import Image
from classfier import predict

st.title("Image Classification on CIFAR-10")
file = st.file_uploader("Upload an image...")

if file is not None:
    image = Image.open(file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("")
    st.write("Predicting the class...")
    label = predict(file)
    st.write(label)
    # st.write('%s (%.2f%%)' % (label[1], label[2]*100))