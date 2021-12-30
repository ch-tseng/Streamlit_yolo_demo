import streamlit as st
import numpy as np
from PIL import Image
from configparser import ConfigParser
import ast
from LIB.libDNN import opencvYOLO
import pandas as pd

def chart_data(dataList):
    clist = list(set(dataList))
    rtn = []
    for cname in clist:
        count = dataList.count(cname)
        rtn.append({cname:count})

    return rtn

def main():
    #st.set_page_config(layout="wide")

    logoCOL1, logoCOL2 = st.columns(2)
    logoImg = Image.open("Images/logo.png")
    logoCOL1.image(logoImg, width=160)
    logoCOL2.title( cfg.get("PAGE", "TITLE") )
    logoCOL2.markdown("<b><p2><font color='blue'>{}</font></p2></b>".format(cfg.get("PAGE", "MODEL_SUBKECT")), unsafe_allow_html=True)


    img_array = upload_image_ui()

    model_size = ast.literal_eval(cfg.get("MODEL", "model_size"))
    path_objname = cfg.get("MODEL", "path_objname")
    path_weights = cfg.get("MODEL", "path_weights")
    path_darknetcfg = cfg.get("MODEL", "path_darknetcfg")
    score = float(cfg.get("MODEL", "score"))
    nms = float(cfg.get("MODEL", "nms"))
    gpu = cfg.getboolean("MODEL", "gpu")

    path_objname = path_objname.replace('\\', '/')
    path_weights = path_weights.replace('\\', '/')
    path_darknetcfg = path_darknetcfg.replace('\\', '/')

    model = opencvYOLO( \
            imgsize=model_size, \
            objnames=path_objname, \
            weights=path_weights, \
            darknetcfg=path_darknetcfg, score=score, nms=nms, gpu=gpu)

    if isinstance(img_array, np.ndarray):
        image = model.getObject(img_array, score, nms, drawBox=True, char_type='Chinese')

        if model.labelNames is not None:
            detCOL1, detCOL2 = st.columns(2)
            refs = []
            total = []
            for id, name in enumerate(model.labelNames):
                box = model.bbox[id]
                bbox = "({},{},{},{})".format(box[0],box[1],box[2],box[3])
                det = { 'Class Name':name, 'Class ID':model.classIds[id], 'Confidence':round(model.scores[id],3), 'BBOX':bbox }
                refs.append(det)
                total.append(name)

            dataChart = pd.DataFrame(chart_data(total))
            detCOL1.bar_chart(dataChart)

            dataREF = pd.DataFrame(refs)
            detCOL2.write(dataREF)

        st.image(image, width=None)

def upload_image_ui():
    uploaded_image = st.file_uploader("Please choose an image file", type=["png", "jpg", "jpeg"])
    if uploaded_image is not None:
        try:
            image = Image.open(uploaded_image)
        except Exception:
            st.error("Error: Invalid image")
        else:
            img_array = np.array(image)
            return img_array

if __name__ == '__main__':
    cfg = ConfigParser()
    cfg.read("config.ini",encoding="utf-8")

    main()

