## Corresponding Paper

This project corresponds to the paper

[Lianmeng Jiao, Feng Wang, Zhun-Ga Liu, and Quan Pan, "**TDEC: Evidential Clustering Based on Transfer Learning and Deep Autoencoder**]

If you have issues, please email: fengwang@mail.nwpu.edu.cn


## Dependency
Now, codes of DEC implemented by pytorch is available: 
- pytorch-1.8.0
- numpy
- scikit-learn 


## Brief Introduction
- TDEC.py: the main source code of TDEC.
- data_loader.py: load source data and target data from matlab files (*.mat). 
- metric.py: codes for evaluation of clustering results. 
-mmd: calculate mmd loss



Samples to run the code is given as follows
```python
   source_data = loader.load_Xs_1()
   target_data, target_labels = loader.load_Xt_1()
   tdekm = TDECM(source_data, target_data, target_labels, [target_data.shape[0], 1024,                       512], gamma=0.1, sigma=0.1, lam1=1, lam2=1, target_batch_size=128,                    source_batch_size=384,  lr=10**-4)
   tdekm.run()
 ```


