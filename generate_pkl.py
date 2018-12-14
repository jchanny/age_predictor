import os
import numpy as np
from shutil import copyfile
import cv2

if __name__ == '__main__':
    training_files = [f for f in os.listdir(os.getcwd() + "/training") if os.path.isfile(os.getcwd() + "/training/" + f)]
    testing_files = [f for f in os.listdir(os.getcwd() + "/testing") if os.path.isfile(os.getcwd() + "/testing/" + f)]
    training_vectors = []
    testing_vectors = []
    for f in training_files:
        src = cv2.imread(os.getcwd() + "/training/" + f,0)
        target_size = (128,128)
        if src is not None:
            dst = cv2.resize(src, target_size)
            age = int(f.split("_")[0])
            training_vectors.append((dst, age))
    for f in testing_files:
        src = cv2.imread(os.getcwd() + "/testing/" + f,0)
        target_size = (128,128)
        if src is not None:
            dst = cv2.resize(src, target_size)
            age = int(f.split("_")[0])
            testing_vectors.append((dst,age))
            
    print(training_vectors[0][0].shape)
    np.save('training_vecs.npy', training_vectors)
    np.save('testing_vecs.npy', testing_vectors)
