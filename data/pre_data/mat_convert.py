import scipy.io
import numpy as np

# 读取mat文件转化成txt数据
mat1 = scipy.io.loadmat('DST_80SOC.mat')  
mat2 = scipy.io.loadmat('pfit.mat') 
 
current = mat1['current'] 
voltage = mat1['voltage']     
pfit = mat2['pfit']


np.savetxt('current.txt', current, delimiter='\t', fmt='%s')  
np.savetxt('voltage.txt', voltage, delimiter='\t', fmt='%s')  
np.savetxt('pfit.txt', pfit.flatten(), delimiter='\t', fmt='%s')  
