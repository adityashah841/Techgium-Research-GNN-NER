import streamlit as st
import os
from copy import deepcopy
from PIL import Image
import pickle
from model import AttributeMapper

if "id" not in st.session_state:
    st.session_state.id = 0
    st.session_state.img_path = ''

@st.cache
def load_image(image_file):
    img = Image.open(image_file)
    return img

def save_uploadedfile(uploadedfile):
    with open(os.path.join("static/documents", uploadedfile.name), "wb") as f:
        f.write(uploadedfile.getbuffer())
    return st.success("Saved File:{} to Data".format(uploadedfile.name))
model = pickle.load(open("extractattri.pkl", "rb"))
st.set_page_config(page_title="Knowledge Graph", page_icon=":book:", layout="wide")
st.subheader("Welcome to Knowledge Graph Creation using NLP & GNN")
st.title("You can upload your file below and select the parameters as you want")
document = st.file_uploader("Choose a file", type=[])
if document is not None:
    file_details = {"FileName":document.name,"FileType":document.type}
    doc = save_uploadedfile(document)
attribute_similarity_ratio = st.number_input('Insert the percentage of attributes you would the the sub-entity & entity to have in common: ', max_value=100, min_value=0)/100
max_depth = st.number_input("Enter the max depth till which subentities can be found: ", min_value=1)
path = "documents/car-manual.pdf"
result = {}
if st.button("Create Knowledge Graph"):
    if document is not None:
        model.set_params(str("D:/Techgium/static/documents/" + document.name), attribute_similarity_ratio, max_depth)
        st.session_state.attributes = model.extract_attributes()
        st.session_state.subentities = model.extract_subentities(deepcopy(st.session_state.attributes))
        model.get_tensors()
        model.get_graph()
        st.session_state.img_path = f'D:\Techgium\static\images\{str(document.name).split(".")[0]}.png'
if st.session_state.img_path != '':
    img = Image.open(st.session_state.img_path)
    st.image(img,caption='Knowledge Graph',width=800)
    def next_ent():
        st.session_state.id = st.session_state.id + 1

    def prev_ent():
        st.session_state.id = st.session_state.id - 1
    try:
        st.write(st.session_state.attributes[st.session_state.id]['entity'])
    except:
        st.write("Click on Previous entity. No more entities to be found.")
    st.button("Next Entity", on_click=next_ent)
    if st.session_state.id >=0:
        st.button("Previous Entity", on_click=prev_ent)
    try:
        st.table(st.session_state.attributes[st.session_state.id]['attributes'])
        st.json(st.session_state.subentities)
    except:
        pass