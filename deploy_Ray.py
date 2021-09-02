from flask import Flask, jsonify, request, send_file

import uuid
from flask_cors import CORS, cross_origin
import os
from collections import OrderedDict
from torch.autograd import Variable
from options.test_options import TestOptions
from data.data_loader import CreateDataLoader
from models.models import create_model
import util.util as util
from util.visualizer import Visualizer
from util import html
import torch
from torchvision import transforms
from PIL import Image
import numpy as np
from io import BytesIO
import ShadowMaker2
import TreePlant
import ray
import cv2

ray.init()

app = Flask(__name__)
CORS(app)
app.config["FLASK_DEBUG"] = False

# query_params = request.values[0]
# body_form_data = request.values[1]
# body_raw_json = request.json

# def serve_pil_image(inputImage,pil_img):
#     img_io = BytesIO()
#     shadow = ShadowMaker.shadow4Building(inputImage,45,120)
#     pil_img.paste(shadow,(0,0),shadow)
#     pil_img.save(img_io,'JPEG',quality=100)
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/jpeg')

# def serveImage(npInput,pil_img):
#     img_io = BytesIO()
#     shadow = ShadowMaker2.drawShadow(npInput,18,120,120)
#     pil_img.paste(shadow,(0,0),shadow)
#     pil_img.save(img_io,'PNG')
#     img_io.seek(0)
#     return send_file(img_io, mimetype='image/png')

@ray.remote
def taskBuildingShadow(label_img_load):
    shadow = ShadowMaker2.drawShadow(label_img_load,18,120,120)
    return shadow

@ray.remote
def taskTreePlant(Imagesize):
    Blank = np.zeros((Imagesize[1],Imagesize[0],4),np.uint8)
    tree = cv2.imread("TreePlantS.png",-1)
    posList = [[0,0],
        [130,130],
        [130,100],
        [100,700],
        [100,900],
        [300,900],
        [500,900],
        [100,1000]]
    TreeImage = TreePlant.plantTreeCV(tree,posList,Blank,120)
    TreeImage = cv2.cvtColor(TreeImage, cv2.COLOR_BGRA2RGBA)

    return Image.fromarray(TreeImage).convert("RGBA")

def Rendering(label_img_load):
    data=dict()
    inst_img_load = label_img_load
    data['inst'] = data['label'] = torch.tensor([[label_img_load]])
    data['feat'] = data['image'] = torch.tensor([0])
    # data['path'] =  ['./temp/0.jpg']
    if opt.data_type == 16:
        data['label'] = data['label'].half()
        data['inst'] = data['inst'].half()
    elif opt.data_type == 8:
        data['label'] = data['label'].uint8()
        data['inst'] = data['inst'].uint8()
    if opt.export_onnx:
        print("Exporting to ONNX: ", opt.export_onnx)
        assert opt.export_onnx.endswith("onnx"), "Export model file should end with .onnx"
        torch.onnx.export(model, [data['label'], data['inst']],
                            opt.export_onnx, verbose=True)
        exit(0)
    minibatch = 1
    if opt.engine:
        generated = run_trt_engine(opt.engine, minibatch, [data['label'], data['inst']])
    elif opt.onnx:
        generated = run_onnx(opt.onnx, opt.data_type, minibatch, [data['label'], data['inst']])
    else:
        generated = model.inference(data['label'], data['inst'], data['image'])

    visuals = OrderedDict([('input_label', util.tensor2label(data['label'][0], opt.label_nc)),
                            ('synthesized_image', util.tensor2im(generated.data[0]))])
    image_pil = Image.fromarray(util.tensor2im(generated.data[0]))
    return image_pil

@app.route('/mask/', methods=['POST'])
@cross_origin()
def post_mask():
    if request.method == 'POST':
        
        label_img = request.files['label']
        label_pil = Image.open(label_img.stream).convert('L')
        label_img_load = np.asarray(label_pil)

        task1 = taskBuildingShadow.remote(label_img_load)
        task2 = taskTreePlant.remote(label_pil.size)

        BuildShadow = ray.get(task1)
        TreeImage = ray.get(task2)

        while True:
            if len(ray.wait([task1,task2])[1])==0:
                break
            

        RendImage = Rendering.remote(label_img_load)

        RendImage.paste(TreeImage,(0,0),TreeImage)
        RendImage.paste(BuildShadow,(0,0),BuildShadow)
        
        img_io = BytesIO()
        RendImage.save(img_io,'PNG')
        img_io.seek(0)
        return send_file(img_io, mimetype='image/png')

        
if __name__ == '__main__':
    opt = TestOptions().parse(save=False)
    opt.nThreads = 1  # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip

    data_loader = CreateDataLoader(opt)
    dataset = data_loader.load_data()
    visualizer = Visualizer(opt)
    # create website
    web_dir = os.path.join(opt.results_dir, opt.name, '%s_%s' % (opt.phase, opt.which_epoch))
    webpage = html.HTML(web_dir, 'Experiment = %s, Phase = %s, Epoch = %s' % (opt.name, opt.phase, opt.which_epoch))

    # test
    if not opt.engine and not opt.onnx:
        model = create_model(opt)
        if opt.data_type == 16:
            model.half()
        elif opt.data_type == 8:
            model.type(torch.uint8)

        if opt.verbose:
            print(model)
    else:
        from run_engine import run_trt_engine, run_onnx
    app.run(host='0.0.0.0')

