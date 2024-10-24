
python train.py --name=clip_vitl14 --wang2020_data_path=datasets/ --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone



python train.py --name=clip_vitl14 --wang2020_data_path=/home/fanzheming/zm/NPR-DeepfakeDetection/dataset --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone


CUDA_VISIBLE_DEVICES=2 python train.py --name=clip_vitl14 --wang2020_data_path=/home/fanzheming/zm/NPR-DeepfakeDetection/dataset --data_mode=wang2020  --arch=CLIP:ViT-L/14  --fix_backbone


CUDA_VISIBLE_DEVICES=0 python3 validate.py --arch=CLIP:ViT-L/14  --ckpt=pretrained_weights/fc_weights.pth --result_folder=clip_vitl14 



CUDA_VISIBLE_DEVICES=1 python3 validate.py --arch=CLIP:ViT-L/14  --ckpt=pretrained_weights/fc_weights.pth --result_folder=clip_vitl14 --jpeg_quality 75


CUDA_VISIBLE_DEVICES=0 python3 validate.py --arch=CLIP:ViT-L/14  --ckpt=pretrained_weights/fc_weights.pth --result_folder=clip_vitl14 --gaussian_sigma 1






