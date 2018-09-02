import sys
sys.path.append('/media/disk1/gaoyuan/OpenFacePytorch/')
from loadOpenFace import *
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
import pickle
import os
from os.path import join

img_parent_path = 'Img/img_align_celeba'
anno_path = './Anno/identity_CelebA.txt'
img_list = []
label_list = []
emb_list = []

gen_emb = prepareOpenFace()
gen_emb = gen_emb.eval()
idx = 0
with open(anno_path, 'r') as f:
    for line in f:
        img_name, label = line.strip().split(' ')
        img_path = join(img_parent_path, img_name)
        img = readAndAlignImage(img_path)
        if img is None:
            continue
        idx += 1
        print('{}: processing img_path={}'.format(idx, img_path))
        img_t = img2Tensor(img)
        img_emb_128, img_emb_736 = gen_emb(img_t)
        emb = img_emb_128.detach().cpu().numpy()[0]
        img = cv2.resize(img, (32, 32))
        import ipdb
        ipdb.set_trace()

        img_list.append(img)
        label_list.append(label)
        emb_list.append(emb)

img_arr = np.array(img_list)
label_arr = np.array(label_list)
emb_arr = np.array(emb_list)



a = train_test_split(img_arr, label_arr, test_size=0.2, random_state=2018)
b = train_test_split(emb_arr, label_arr, test_size=0.2, random_state=2018)

pickle.dump({'img':a[0], 'emb':b[0], 'label':a[2]}, open('celeba_train_set_128.p', 'wb'))
pickle.dump({'img':a[1], 'emb':b[1], 'label':a[3]}, open('celeba_test_set_128.p', 'wb'))
        





#img_output_path = 'lfw_align_img/'
#emb_128_output_path = 'lfw_align_emb_128/'
#emb_736_output_path = 'lfw_align_emb_736/'
#
#if not os.path.exists(img_output_path):
#    os.makedirs(img_output_path)
#if not os.path.exists(emb_128_output_path):
#    os.makedirs(emb_128_output_path)
#if not os.path.exists(emb_736_output_path):
#    os.makedirs(emb_736_output_path)
#
#img_path = []
#for name in os.listdir('lfw/'):
#    for img_name in os.listdir('lfw/'+name):
#        img_path.append('lfw/' + name + '/' + img_name)
##imgs = [readAndAlignImage(path) for path in img_path[:3]]
#
#
#nof = prepareOpenFace()
#nof = nof.eval()
#idx = 0
#for i in range(len(img_path)):
#    img = readAndAlignImage(img_path[i])
#    if img is not None:
#        idx += 1
#        print('{}: processing img_path={}'.format(idx, img_path[i]))
#        img_name = img_path[i][img_path[i].rfind('/')+1:]
#        img_t = img2Tensor(img)
#        img_emb_128, img_emb_736 = nof(img_t)
#
#        cv2.imwrite(img_output_path + img_name, img)
#        np.save(emb_128_output_path + img_name[:img_name.rfind('.')] + '.npy', img_emb_128.detach().cpu().numpy())
#        np.save(emb_736_output_path + img_name[:img_name.rfind('.')] + '.npy', img_emb_736.detach().cpu().numpy())
#
#
#imgs = []
#embs = []
#labels = []
#for name in os.listdir('lfw_align_img/'):
#
#    img = cv2.imread('lfw_align_img/' + name)
#    img = cv2.resize(img, (32, 32))
#    imgs.append(img[...,::-1])
#    emb = np.load('lfw_align_emb_128/' + name[:name.find('.')] + '.npy')
#    embs.append(emb[0])
#    label = name[:name.rfind('_')]
#    labels.append(label)
#imgs = np.array(imgs)
#embs = np.array(embs)
#
#label_n2s = {}
#label_s2n = {}
#for num, name in enumerate(np.unique(labels)):
#    label_n2s[num] = name
#    label_s2n[name] = num
#np.save('label_num2str.npy', label_n2s) 
#np.save('label_str2num.npy', label_s2n) # tmp = np.load('label_str2num.npy').item()
#exit()
#
#labels_num = []
#for label in labels:
#    labels_num.append(label_s2n[label])
#
#a = train_test_split(imgs, labels_num, test_size=0.1, random_state=2018)
#b = train_test_split(embs, labels_num, test_size=0.1, random_state=2018)
#
#pickle.dump({'img':a[0], 'emb':b[0], 'label':a[2]}, open('lfw_train_set_128.p', 'wb'))
#pickle.dump({'img':a[1], 'emb':b[1], 'label':a[3]}, open('lfw_test_set_128.p', 'wb'))
