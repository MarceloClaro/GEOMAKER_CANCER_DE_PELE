import streamlit as st
import cv2
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model
from skimage.transform import resize
from skimage.io import imread

def enhance(img):
    # reshape(1, 256, 256, 1)
    #sub = (model.predict(img.reshape(1,256,256,3))).flatten()
    img = img.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    sub = (model.predict(img)).flatten()

    for i in range(len(sub)):
        if sub[i] > 0.5:
            sub[i] = 1
        else:
            sub[i] = 0
    return sub

def applyMask(img):
    sub = img.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    #sub = np.array(img.reshape(256, 256), dtype=np.uint8)
    mask = np.array(enhance(sub).reshape(256, 256), dtype=np.uint8)
    sub2 = img.reshape(256, 256, 3)
    #sub2 = np.array(img.reshape(256, 256, 3), dtype=np.uint8)
    res = cv2.bitwise_and(sub2, sub2, mask = mask)

    return res

##############
# Model Load #
##############
@st.cache
def load():
    return load_model('ResU_net.h5')
model = load()


##############
# Barra Lateral   #
##############
with st.sidebar.header('Faça o upload da sua imagem de pele'):
    upload_file = st.sidebar.file_uploader('Escolha sua imagem de pele', type=['jpg', 'jpeg', 'png'])


##############
# Título da Página #
##############
st.write('# 🧠Segmentação de Lesões na Pele🧠')
st.write('Este site foi criado por Crinex. O código do site e da segmentação está no Github. Se você quiser usar este código, por favor faça um Fork e use-o.🤩🤩')
st.write('📕 Github: https://github.com/crinex/Skin-Lesion-Segmentation-Streamlit 📕')


###############
# Tela Principal #
###############
col1, col2, col3 = st.beta_columns(3)
with col1:
    st.write('### Imagem Original')
    img = imread(upload_file)
    img = resize(img, (256, 256))
    preview_img = resize(img, (256, 256))
    st.image(preview_img)

col2.write('### Botão')
clicked = col2.button('Segmentar!!')
clicked2 = col2.button('Prever Imagem')

if clicked:
    x = img
    #x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    #x = x.reshape((1, 256, 256, 3)).astype(np.float32) / 255.
    x = np.reshape(x, (256, 256, 3))
    #x = resize(x, (256, 256, 3))
    #pred = model.predict(x).squeeze()
    col3.write('### Imagem Segmentada')
    mask_img = applyMask(x)
    col3.image(mask_img)

if clicked2:
    x = img
    #x = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    x = np.reshape(x, (256, 256, 3))
    #x = resize(x, (256, 256, 1))
    enhance_img = enhance(x).reshape(256, 256)
    col3.write('### Imagem de Predição')
    col3.image(enhance_img)
