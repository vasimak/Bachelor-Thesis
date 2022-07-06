import numpy as np
import math
import time
import matplotlib.pyplot as plt
start_time = time.time()
N=10**5
k=0.79
g=3
filename="./Latex/LateX images/log/q/g" +str(g)+".jpg"
x =np.zeros(len(range(0, N)))
y = np.zeros(len(range(0, N)))
X = np.zeros(len(range(0, N)))
Y =np.zeros(len(range(0, N)))
M=np.zeros(len(range(0, N)))
theta=np.zeros(len(range(0, N)))

#chaotic maps
# def rs(x1,y1):
#     #choose parameters for the two chaotic maps used

#     x[0]=x1
#     y[0]=y1
#     #choose parameters for the robot
#     # X[1]=0
#     #Y[1]=0
#     #theta[1]=0
#     L= 0.15
#     h=0.2
#     q=-1.4
#     for i in range(1,N):
#         x[i] =k *((1 + x[i - 1]) *(1 + x[i - 1])) * (2 - x[i - 1]) +q
#         y[i] =k *((1 + y[i - 1]) *(1 + y[i - 1]))  * (2 - y[i - 1]) +q
        
# #robot coordinates
#         X[i]=X[i-1]+h*math.cos(theta[i-1])*(x[i-1]+y[i-1])/2
#         Y[i]=Y[i-1]+h*math.sin(theta[i-1])*(x[i-1]+y[i-1])/2
#         theta[i]=theta[i-1]+h*(x[i-1]-y[i-1])/L
# #keep the robot inside the boundaries
#         if X[i]>=40 or X[i] <=0:
#             X[i]=X[i-1]-h*math.cos(theta[i-1])*(x[i-1]+y[i-1])/2
#         if Y[i]>=40 or Y[i] <=0 :
#             Y[i]=Y[i-1]-h*math.sin(theta[i-1])*(x[i-1]+y[i-1])/2
#     return(X,Y)


#chaotic maps
def rs2(q):
    #choose parameters for the two chaotic maps used
    N=10**5  
    x[0]=0
    y[0]=0.1
    #choose parameters for the robot
    X[0]=0
    Y[0]=0
    #theta[1]=0
    L= 0.15
    h=0.2
    q=q
    for i in range(1,N):
        x[i] =k *((1 + x[i - 1]) *(1 + x[i - 1])) * (2 - x[i - 1]) +q
        y[i] =k *((1 + y[i - 1]) *(1 + y[i - 1]))  * (2 - y[i - 1]) +q
        
#robot coordinates
        X[i]=X[i-1]+h*math.cos(theta[i-1])*(x[i-1]+y[i-1])/2
        Y[i]=Y[i-1]+h*math.sin(theta[i-1])*(x[i-1]+y[i-1])/2
        theta[i]=theta[i-1]+h*(x[i-1]-y[i-1])/L
#keep the robot inside the boundaries
        if X[i]>=40 or X[i] <=0:
            X[i]=X[i-1]-h*math.cos(theta[i-1])*(x[i-1]+y[i-1])/2
        if Y[i]>=40 or Y[i] <=0 :
            Y[i]=Y[i-1]-h*math.sin(theta[i-1])*(x[i-1]+y[i-1])/2
    return(X,Y)


#chaotic maps
# def rs2(X1,Y1):
#     #choose parameters for the two chaotic maps used    
#     x[0]=0
#     y[0]=0.1
#     #choose parameters for the robot
#     X[0]=X1
#     Y[0]=Y1
#     #theta[1]=0
#     L= 0.15
#     h=0.2
#     q=-1.9
#     for i in range(1,N):
#         x[i] =k *((1 + x[i - 1]) *(1 + x[i - 1])) * (2 - x[i - 1]) +q
#         y[i] =k *((1 + y[i - 1]) *(1 + y[i - 1]))  * (2 - y[i - 1]) +q
        
#robot coordinates
#         X[i]=X[i-1]+h*math.cos(theta[i-1])*(x[i-1]+y[i-1])/2
#         Y[i]=Y[i-1]+h*math.sin(theta[i-1])*(x[i-1]+y[i-1])/2
#         theta[i]=theta[i-1]+h*(x[i-1]-y[i-1])/L
# #keep the robot inside the boundaries
#         if X[i]>=40 or X[i] <=0:
#             X[i]=X[i-1]-h*math.cos(theta[i-1])*(x[i-1]+y[i-1])/2
#         if Y[i]>=40 or Y[i] <=0 :
#             Y[i]=Y[i-1]-h*math.sin(theta[i-1])*(x[i-1]+y[i-1])/2
#     return(X,Y)

plt.figure()
font = {'size': 45}
plt.rc('font', **font)
plt.plot(X,Y ,'--b',alpha=0.8,ms=1) 
plt.rcParams.update({'text.usetex' : True})
plt.rcParams['agg.path.chunksize'] = 10000
plt.xlim(0,40)
plt.ylim(0,40)
plt.xlabel("X")
plt.ylabel("Y")

rs1=rs2(-1.6)
plt.plot(X,Y ,'--k',alpha=1,ms=1.2)
rs1=rs2(-1.4)
plt.plot(X,Y ,'--r',alpha=0.8,ms=0.8)
# rs1=rs2(8,30)
# plt.plot(X,Y ,'--b',alpha=0.8,ms=0.6)
# rs1=rs2(36,6)
# plt.plot(X,Y ,'--y',alpha=0.5,ms=0.2)
# rs1=rs(0.8,1.2)
# plt.plot(X,Y ,'--y',alpha=0.2,ms=0.2)

# rs1=rs(-2.1)
# plt.plot(X,Y ,'--y',alpha=0.8,ms=1)
# rs2=np.array(rs1, dtype=np.int)
figure = plt.gcf()  # get current figure
figure.set_size_inches(1920/40, 1080/40)
plt.savefig(filename,dpi=40)
print("--- %s seconds ---" % (time.time() - start_time))
plt.show()