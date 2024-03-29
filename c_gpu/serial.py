import math,sys,os
#import matplotlib.pyplot as plt no need for graphing in our case
#import matplotlib.cm as cm
#from matplotlib.colors import Normalize
import numpy as np
import time

try:
    sys.path.append('libs/') #add the libs to the system path so we can import them
    #import matrix_solver as ms #add custom modules to this line
    import setup_2d_PDE as pd
except ModuleNotFoundError:
    print('Please only run this from the PRH_HW3 directory.')
    raise ModuleNotFoundError
    sys.exit()

#be aware that grabbing from the config file and setting up module-level variables
#happens inside setup_2d_PDE.py
setupStart = time.time()
A,b = pd.buildCartStencil(pd.NX,pd.NY)
setupEnd = time.time()
setupTime = setupEnd - setupStart
psi_vector = pd.solvePsi(A, b, pd.SOLVER, pd.TOL, pd.MAXITERS)
solveEnd = time.time()
solveTime = solveEnd - setupEnd
psi = pd.wrapPsi(psi_vector) #stream function psi(x,y) for delsquared = 0
wrapEnd = time.time()
wrapTime = wrapEnd - solveEnd

totalTime = wrapEnd - setupStart
percentSetup = setupTime/totalTime*100
percentSolve = solveTime/totalTime*100
percentWrap = wrapTime/totalTime*100

print("Time taken for setup: {}\n".format(setupEnd - setupStart), "Percentage: {}\n".format(percentSetup))
print("Time taken for solution: {}\n".format(solveEnd - setupEnd), "Percentage: {}\n".format(percentSolve))
print("Time taken for wrapping: {}\n".format(wrapEnd - solveEnd), "Percentage: {}\n".format(percentWrap))
print("Total time (excluding graphing): {}\n".format(totalTime))

##U,V = pd.vectorField(psi, pd.DX, pd.DY) #vector field for psi
##
###now it's time to graph
##xpsi, ypsi = np.meshgrid(pd.X,pd.Y) #x and y matrices for pylab graphing for psi
##xfield, yfield = np.meshgrid(pd.X[1:-1],pd.Y[1:-1]) #ditto for the vector field; we're cutting off the edges because this is a central different approximation
##
##plt.figure(1)
##plt.pcolor(xpsi, ypsi, psi)
##plt.colorbar().set_label('Value of Psi at x and y')
##plt.title('Psi(x,y), Q = {3} rho={0}, dx={1}, dy={2}'.format(pd.RHO,pd.DX,pd.DY,pd.Q))
##
###don't clobber existing files - this can mean a lot of copies
##copynum = 0
##psipath = 'pics/psi/psi{0}.png'.format(copynum)
##while os.path.isfile(psipath):
##    copynum +=1
##    psipath = 'pics/psi/psi{0}.png'.format(copynum)
##plt.savefig(psipath)
##
##plt.figure(2)
##
##Q = plt.quiver(xfield,yfield,U,V,)
##
##plt.title('Vector field, u = (U,V), rho={0}, dx={1}, dy={2}'.format(pd.RHO,pd.DX,pd.DY))
##
##fieldpath = 'pics/field/field{0}.png'.format(copynum)
##copynum = 0
##while os.path.isfile(fieldpath):
##    copynum +=1
##    fieldpath = 'pics/field/field{0}.png'.format(copynum)
##plt.savefig(fieldpath)
##
##plt.show()
