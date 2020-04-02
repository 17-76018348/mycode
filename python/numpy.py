# Numpy where
import numpy as np
def where():
    a = np.array([1, 2, 3, 10, 20, 30, 0.1, 0.2])
    
    np.where(a<1)
    # array([6, 7])
    # 해당 조건을 만족하는 인덱스를 return 하게 된다
    
    # value 즉 값을 구하고 싶다면 슬라이싱이 가능하다
    a[np.where(a<1)]
    #array([0.1, 0.2])
    
    np.where(a>=10, 0, a)
    # 1.조건 2.조건만족시 변경할값 3.
    # 
    
  