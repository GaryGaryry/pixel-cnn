CUDA_VISIBLE_DEVICES=0 nohup python -u gen_embedding_facenet.py /media/disk1/gaoyuan/pixel-cnn/data/celeba/img_align_mtcnn/img/ /media/disk1/gaoyuan/facenet/pretrained_models/20180402-114759/ --use_fixed_image_standardization > ../../run_logs/celeba_pickle_facenet_dataset.log 2>&1 &
