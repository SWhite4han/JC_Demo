
#python -u verification.py --gpu 0 --data-dir /opt/jiaguo/faces_vgg_112x112 --image-size 112,112 --model '../../model/softmax1010d3-r101-p0_0_96_112_0,21|22|32' --target agedb_30
python -u verification.py --gpu 0 --data-dir /home/user/Downloads/faces_emore --model '/home/user/PycharmProjects/Demo_site/JC_Demo/fr_module/model_alignt_person/model,792' --target lfw --batch-size 128
