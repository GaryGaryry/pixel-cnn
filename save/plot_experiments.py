import sys
import pickle
import cv2
import numpy as np
import argparse
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('MacOSX')

parser = argparse.ArgumentParser()
# data I/O
parser.add_argument('-i', '--input_path', type=str, default=None, help='Location for the dataset')
parser.add_argument('-o', '--output_name', type=str, default=None, help='Location for parameter checkpoints and samples')
parser.add_argument('-uo', '--origin', dest='origin', action='store_true', help='using origin img?')
parser.add_argument('-ch', '--cond_h', dest='cond_h', action='store_true', help='using conditional h?')
args = parser.parse_args()

def img_stretch(img):
    img = img.astype(float)
    img -= np.min(img)
    img /= np.max(img)+1e-12
    return img

def plot_a(sample, img, label, subtitle='', name=None):
    resizesize = (128, 128)
    fig = plt.figure(figsize=(15, 10))
    fig.suptitle("experiment a " + subtitle, fontsize=10, verticalalignment='bottom')
    h, w = len(sample), sample[0].shape[0]+1
    import matplotlib.gridspec as gridspec
    gs = gridspec.GridSpec(h, w)
    gs.update(wspace=0.025, hspace=0.1)
    idx = 0
    for i in range(h):
        for j in range(w):
            ax = plt.subplot(h, w, idx+1)
            #ax = plt.subplot(gs[idx])
            ax.axis('off')
            if j == 0:
                face = cv2.resize(img[i][j], resizesize)
                ax.set_title("origin " + label[i][0], size=5)
            else:
                face = cv2.resize(img_stretch(sample[i][j-1]), resizesize)
                ax.set_title("gen " + label[i][j-1], size=5)
            #ax.set_sapect(1.0)
            ax.imshow(face, aspect='equal')
            idx += 1
    plt.tight_layout()
    #fig.subplots_adjust(right=0.02)
    plt.subplots_adjust(wspace=0, hspace=0)
    if name is not None:
        plt.savefig(filepath[:filepath.rfind('/')+1] + name.replace(' ', '_')+'.png')
    else:
        plt.show()

if not args.cond_h:
    filepath = args.input_path
    img_name = args.output_name if args.output_name else filepath[filepath.rfind('/') : filepath.rfind('.')]
    print("plot " + filepath)

    a = pickle.load(open(filepath, 'rb'))
    sample = a['sample']
    img = a['img']
    label = a['label']
    plot_a(sample, img, label, subtitle='origin', name=img_name)
else:
    def plot_b(sample, img, label, cond_img, cond_label, subtitle='', name=None):
        resizesize = (128, 128)
        fig = plt.figure(figsize=(15, 10))
        fig.suptitle("experiment a " + subtitle, fontsize=10, verticalalignment='bottom')
        h, w = len(sample), sample[0].shape[0] + 2
        import matplotlib.gridspec as gridspec
        gs = gridspec.GridSpec(h, w)
        gs.update(wspace=0.025, hspace=0.1)
        idx = 0
        for i in range(h):
            for j in range(w):
                ax = plt.subplot(h, w, idx + 1)
                # ax = plt.subplot(gs[idx])
                ax.axis('off')
                if j == 0:
                    face = cv2.resize(img[i][j], resizesize)
                    ax.set_title("origin " + label[i][0], size=5)
                elif j == w - 1:
                    face = cv2.resize(cond_img, resizesize)
                    ax.set_title("origin " + cond_label, size=5)
                else:
                    face = cv2.resize(img_stretch(sample[i][j - 1]), resizesize)
                    ax.set_title("gen " + label[i][j - 1], size=5)
                # ax.set_sapect(1.0)
                ax.imshow(face, aspect='equal')
                idx += 1
        plt.tight_layout()
        # fig.subplots_adjust(right=0.02)
        plt.subplots_adjust(wspace=0, hspace=0)
        if name is not None:
            plt.savefig(filepath[:filepath.rfind('/') + 1] + name.replace(' ', '_') + '.png')
        else:
            plt.show()

    filepath = args.input_path
    img_name = args.output_name if args.output_name else filepath[filepath.rfind('/') : filepath.rfind('.')]
    p
    print("plot " + filepath)
    a = pickle.load(open(filepath, 'rb'))
    sample = a['sample']
    img = a['img']
    label = a['label']
    cond_img = a['cond_img']
    cond_label = a['cond_label']
    plot_b(sample, img, label, cond_img, cond_label, subtitle='origin', name=img_name)
