# Design and implementation of remote video recognition system based on convolutional neural network
Inception V3 transfer learning in Hmdb51.referring someone and some creation
this project is for beginner in deep-learning and Can't be applied to a real scenario.
environment: windows10 + pycharm2019 + Anaconda2.4 + tensorflow_gpu1.13.0 + cuda10.0 + cudnn7.5 + Nvidia geforce 940mx 
# download datasets and generate documents
http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/#Downloads

run /data/generateClassTXT.py .  we can get a file with classification in HMDB51.
run /data/combine_test_split.py . we can get train.txt and test.txt.
run /data/move_file.py .we can move videos into train/test folders. 
run /data/extract_file.py . some pictures are extracted from videos and a data file is created for training and testing later
# train model
run train.py.  It takes 3.hours in first training and 4.5hours in second training.
# start server
run server.py
# start client
run client.py
