import paddle
import cv2
import numpy as np
import os
import datetime
from models.model import FaceSwap, l2_norm
from models.arcface import IRBlock, ResNet
from utils.align_face import back_matrix, dealign, align_img
from utils.util import paddle2cv, cv2paddle
from utils.prepare_data import LandmarkModel
from gfpgan import GFPGANer
from PIL import Image

# User inputs a couple image and outputs a new couple image.

def couple2couple(background_img_path, source_img_path):
    COMBINATED_DIR = 'combinated_image'
    os.makedirs(COMBINATED_DIR, exist_ok=True)
    landmarkModel = LandmarkModel(name='landmarks')
    landmarkModel.prepare(ctx_id= 0, det_thresh=0.6, det_size=(640,640))

    source_aligned_img =  faces_align(landmarkModel, source_img_path)
    background_aligned_img = faces_align(landmarkModel, background_img_path)

    paddle.set_device("cpu")
    faceswap_model = FaceSwap("cpu")

    id_net = ResNet(block=IRBlock, layers=[3, 4, 23, 3])
    id_net.set_dict(paddle.load('./checkpoints/arcface.pdparams'))

    id_net.eval()

    weight = paddle.load('./checkpoints/MobileFaceSwap_224.pdparams')

    start_idx = background_img_path.rfind('/')
    if start_idx > 0:
        background_name = background_img_path[background_img_path.rfind('/'):]
    else:
        background_name = background_img_path
    origin_att_img = cv2.imread(background_img_path)

    for idx, background in enumerate(background_aligned_img):
        print('checking background couple gender, idx: ', idx, ' gender: ', background[2])
        if(background[2] == 0):
            for idx, source in enumerate(source_aligned_img):
                print('checking if source is a lady, idx: ', idx, ' gender: ', source[2])
                if(source[2] == 0):
                    id_emb, id_feature = get_id_emb_from_image(id_net, source[0])


                    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
                    faceswap_model.eval()

                    att_img = cv2paddle(background[0])
                    res, mask = faceswap_model(att_img)
                    res = paddle2cv(res)

                    back_matrix = background[1]
                    mask = np.transpose(mask[0].numpy(), (1, 2, 0))

                    origin_att_img = dealign(res, origin_att_img, back_matrix, mask)
                    print('faceswap for a lady completed !')

        elif(background[2] == 1):
            for idx, source in enumerate(source_aligned_img):
                print('checking if source is a man, idx: ', idx, ' gender: ', source[2])
                if(source[2] == 1):
                    id_emb, id_feature = get_id_emb_from_image(id_net, source[0])


                    faceswap_model.set_model_param(id_emb, id_feature, model_weight=weight)
                    faceswap_model.eval()

                    att_img = cv2paddle(background[0])
                    res, mask = faceswap_model(att_img)
                    res = paddle2cv(res)

                    back_matrix = background[1]
                    mask = np.transpose(mask[0].numpy(), (1, 2, 0))

                    origin_att_img = dealign(res, origin_att_img, back_matrix, mask)
                    print('faceswap for a man completed !')

    # cv2.imshow('result', origin_att_img)
    # cv2.waitKey(0)

    cv2.imwrite(os.path.join(COMBINATED_DIR, os.path.basename(background_name.format(idx))), origin_att_img)
    result_img = os.path.join(COMBINATED_DIR, os.path.basename(background_name.format(idx)))
    gfpgan_gogo(result_img)

def get_id_emb_from_image(id_net, id_img):
    id_img = cv2.resize(id_img, (112, 112))
    id_img = cv2paddle(id_img)
    mean = paddle.to_tensor([[0.485, 0.456, 0.406]]).reshape((1, 3, 1, 1))
    std = paddle.to_tensor([[0.229, 0.224, 0.225]]).reshape((1, 3, 1, 1))
    id_img = (id_img - mean) / std
    id_emb, id_feature = id_net(id_img)
    id_emb = l2_norm(id_emb)

    return id_emb, id_feature

def faces_align(landmarkModel, image_path, image_size=224):
    aligned_imgs =[]
    if os.path.isfile(image_path):
        img_list = [image_path]
    else:
        img_list = [os.path.join(image_path, x) for x in os.listdir(image_path) if x.endswith('png') or x.endswith('jpg') or x.endswith('jpeg')]

    for path in img_list:
        img = cv2.imread(path)
        face_info = landmarkModel.get_faces(img)
        
        for fi in face_info:
            if fi['kps'] is not None:
                face_gender = fi['gender']
                aligned_img, back_matrix = align_img(img, fi['kps'], image_size)
                aligned_imgs.append([aligned_img, back_matrix, face_gender])

    return aligned_imgs

def gfpgan_gogo(img):
    img = Image.open(img)
    original_img = img.copy()
    np_img = np.array(img)

    device = 'cpu'
    model = GFPGANer(model_path='./models/GFPGANv1.4.pth', upscale=1, arch='clean', channel_multiplier=2, bg_upsampler=None, device=device)
    np_img_bgr = np_img[:, :, ::-1]
    _, _, gfpgan_output_bgr = model.enhance(np_img_bgr, has_aligned=False, only_center_face=False, paste_back=True)
    np_img = gfpgan_output_bgr[:, :, ::-1]

    restored_img = Image.fromarray(np_img)
    result_img = Image.blend(
        original_img, restored_img, 1
    )

    base_path = './results/'
    result_img_np = np.array(result_img)
    result_img_rgb = result_img_np[:, :, ::-1]

    suffix = datetime.datetime.now().strftime('%y%m%d_%H%M%S')
    fileName = suffix + '.png'

    cv2.imwrite(base_path + fileName, result_img_rgb)

    print('couple2couple completed !')

if __name__ == '__main__':
    # background image, couple image
    couple2couple('data/test_couple2.jpg', 'data/test_couple.jpg')