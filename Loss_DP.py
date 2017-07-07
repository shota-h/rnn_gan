import numpy as np
import itertools
import os

file_dir = os.path.abspath(os.path.dirname(__file__))
os.mkdir(file_dir+'/rnn_gan_parallel_train_lossdp/layer{0}_cell{1}'.format(layer_num, cell_num))
file_path = file_dir+'/rnn_gan_parallel_train_lossdp/layer{0}_cell{1}


def loss_dp(y_true,y_pred):
    lodd_vec=[]
    for k in range(y_true.shape[0]):
        dp_mat=np.zeros([y_true.shape[1],y_pred.shape[1]])
# norm matrix
        for i,j in itertools.product(range(dp_mat.shape[0]),range(dp_mat.shape[1])):
                dp_mat[i,j]=np.linalg.norm(y_true[k,i,0]-y_pred[k,j,0])
                if j==0 and i>0:
                    dp_mat[i,j]=float('inf')
# DP argorithm
            for j,i in itertools.product(range(1,dp_mat.shape[1]),range(dp_mat.shape[0])):
                if i==0:
                    dp_mat[i,j]+=dp_mat[i,j-1]
                elif i==1:
                    # print(np.argmin(dp_mat_copy[i+1:i-1:-1,j-1]))
                    dp_mat[i,j]+=np.min(dp_mat[i-1:i+1,j-1])
                else:
                    dp_mat[i,j]+=np.min(dp_mat[i-2:i+1:,j-1])
        loss_vec.append(dp_mat[-1,-1])
    return loss_vec
