import numpy as np
import os
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt

#导入pandas库
import pandas as pd
#生成一个Series
s=pd.Series([1,3,3,4], index=list('ABCD'))

#括号内不指定图表类型，则默认生成直线图
s.plot()
plt.show()
print('pandas_picture_show！！！')
