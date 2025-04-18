import warnings
warnings.filterwarnings('ignore')

import sys
import os
import time
import pickle
import torch
import numpy as np
import random

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from main import Model

SEED = 23
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

pre_file_path = 'datas/datas_pre.pkl'
fit_file_path = 'datas/datas_fit.pkl'

with open(pre_file_path, 'rb') as f:
    datas_pre = pickle.load(f)

#if os.path.exists(fit_file_path):
#    with open(fit_file_path, 'rb') as f:
#        datas_fit = pickle.load(f)
#else:
datas_fit = []

MINUTES = [5,30]
LENGTHS = [1,2]

for i in range(0,len(datas_pre)):
    print(f"Processing dataset {i+1}/{len(datas_pre)}")
    for Minute in MINUTES:
        for Length in LENGTHS:
            Data = datas_pre[i]
            Center = 31
            Step = Minute // Data.TemporalResolution
            Transport = Model.Transport(Data, Center=Center, Step=Step, Length=Length)
            print(f"  Fitting with step size: {Minute} minutes and length: {Length} ")
                
            start_time = time.time()
            Transport.TrainMLE(Method="NeuralFlow", Epochs=50, SubSampleSize=2000)
            end_time = time.time()
            Transport.Time = end_time - start_time
            datas_fit.append(Transport)

    with open(fit_file_path, 'wb') as f:
        pickle.dump(datas_fit, f)

print("Fitting process completed.")