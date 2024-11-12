
python src\align_dataset_mtcnn.py data\raw data\processed --image_size 160 --margin 32 --random_order --gpu_memory_fraction 0.25

python src\classifier.py TRAIN data\processed Models\20180402-114759.pb Models\facemodel.pkl --batch_size 1000