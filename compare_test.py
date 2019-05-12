import math,sys,os,time,csv
#import matplotlib.pyplot as plt no need for graphing in our case
#import matplotlib.cm as cm
#from matplotlib.colors import Normalize
import numpy as np
import statistics as stats

try:
    sys.path.append('test-script-libs/') #add the libs to the system path so we can import them
    #import matrix_solver as ms #add custom modules to this line
    import orig_setup_2d_PDE as orig_pd
    import serial_setup_2d_PDE as ser_pd
    import gpu_setup_2d_PDE as gpu_pd
except ModuleNotFoundError:
    print('Incorrect run directory, please do not move.')
    raise ModuleNotFoundError
    sys.exit()

#be aware that grabbing from the config file and setting up module-level variables
#happens inside setup_2d_PDE.py

#
#original, nested python lists as dense matrices
#
origSetupTimes = []
origSetupPercs = []
origSolveTimes = []
origSolvePercs = []
origWrapTimes = []
origWrapPercs = []
origTotalTimes = []
origTotalPercs = []

for i in range(10):    
    setupStart = time.time()
    A,b = orig_pd.buildCartStencil(orig_pd.NX,orig_pd.NY)
    setupEnd = time.time()
    setupTime = setupEnd - setupStart
    psi_vector = orig_pd.solvePsi(A, b, orig_pd.SOLVER, orig_pd.TOL, orig_pd.MAXITERS)
    solveEnd = time.time()
    solveTime = solveEnd - setupEnd
    psi = orig_pd.wrapPsi(psi_vector) #stream function psi(x,y) for delsquared = 0
    wrapEnd = time.time()
    wrapTime = wrapEnd - solveEnd

    totalTime = wrapEnd - setupStart
    percentSetup = setupTime/totalTime*100
    percentSolve = solveTime/totalTime*100
    percentWrap = wrapTime/totalTime*100

    origSetupTimes.append(setupTime)
    origSetupPercs.append(percentSetup)
    origSolveTimes.append(setupTime)
    origSolvePercs.append(percentSolve)
    origWrapTimes.append(wrapTime)
    origWrapPercs.append(percentWrap)
    origTotalTimes.append(totalTime)

#    print("Time taken for setup: {}\n".format(setupEnd - setupStart), "Percentage: {}\n".format(percentSetup))
#    print("Time taken for solution: {}\n".format(solveEnd - setupEnd), "Percentage: {}\n".format(percentSolve))
#    print("Time taken for wrapping: {}\n".format(wrapEnd - solveEnd), "Percentage: {}\n".format(percentWrap))
#    print("Total time: {}\n".format(totalTime))


#serial, optimized np sparse arrays
serSetupTimes = []
serSetupPercs = []
serSolveTimes = []
serSolvePercs = []
serWrapTimes = []
serWrapPercs = []
serTotalTimes = []
serTotalPercs = []

for i in range(10):    
    setupStart = time.time()
    A,b = ser_pd.buildCartStencil(ser_pd.NX,ser_pd.NY)
    setupEnd = time.time()
    setupTime = setupEnd - setupStart
    psi_vector = ser_pd.solvePsi(A, b, ser_pd.SOLVER, ser_pd.TOL, ser_pd.MAXITERS)
    solveEnd = time.time()
    solveTime = solveEnd - setupEnd
    psi = ser_pd.wrapPsi(psi_vector) #stream function psi(x,y) for delsquared = 0
    wrapEnd = time.time()
    wrapTime = wrapEnd - solveEnd

    totalTime = wrapEnd - setupStart
    percentSetup = setupTime/totalTime*100
    percentSolve = solveTime/totalTime*100
    percentWrap = wrapTime/totalTime*100

    serSetupTimes.append(setupTime)
    serSetupPercs.append(percentSetup)
    serSolveTimes.append(setupTime)
    serSolvePercs.append(percentSolve)
    serWrapTimes.append(wrapTime)
    serWrapPercs.append(percentWrap)
    serTotalTimes.append(totalTime)

#gpu, sparse arrays via CUDA/CUPy backend
gpuSetupTimes = []
gpuSetupPercs = []
gpuSolveTimes = []
gpuSolvePercs = []
gpuWrapTimes = []
gpuWrapPercs = []
gpuTotalTimes = []
gpuTotalPercs = []

for i in range(10):    
    setupStart = time.time()
    A,b = gpu_pd.buildCartStencil(gpu_pd.NX,gpu_pd.NY)
    setupEnd = time.time()
    setupTime = setupEnd - setupStart
    psi_vector = gpu_pd.solvePsi(A, b, gpu_pd.SOLVER, gpu_pd.TOL, gpu_pd.MAXITERS)
    solveEnd = time.time()
    solveTime = solveEnd - setupEnd
    psi = gpu_pd.wrapPsi(psi_vector) #stream function psi(x,y) for delsquared = 0
    wrapEnd = time.time()
    wrapTime = wrapEnd - solveEnd

    totalTime = wrapEnd - setupStart
    percentSetup = setupTime/totalTime*100
    percentSolve = solveTime/totalTime*100
    percentWrap = wrapTime/totalTime*100

    gpuSetupTimes.append(setupTime)
    gpuSetupPercs.append(percentSetup)
    gpuSolveTimes.append(setupTime)
    gpuSolvePercs.append(percentSolve)
    gpuWrapTimes.append(wrapTime)
    gpuWrapPercs.append(percentWrap)
    gpuTotalTimes.append(totalTime)

#
#stats for comparison
#

#originals
minOrigSetupTime = min(origSetupTimes) #don't care about percentages except solve
maxOrigSetupTime = max(origSetupTimes) #can change later if we like
avgOrigSetupTime = stats.mean(origSetupTimes)

minOrigSolveTime = min(origSolveTimes) 
maxOrigSolveTime = max(origSolveTimes)
avgOrigSolveTime = stats.mean(origSolveTimes)

minOrigSolvePerc = min(origSolvePercs) 
maxOrigSolvePerc = max(origSolvePercs)
avgOrigSolvePerc = stats.mean(origSolvePercs)

minOrigWrapTime = min(origWrapTimes) 
maxOrigWrapTime = max(origWrapTimes)
avgOrigWrapTime = stats.mean(origWrapTimes)

minOrigTotalTime = min(origTotalTimes) 
maxOrigTotalTime = max(origTotalTimes)
avgOrigTotalTime = stats.mean(origTotalTimes)

#serial
minSerSetupTime = min(serSetupTimes) #don't care about percentages except solve
maxSerSetupTime = max(serSetupTimes) #can change later if we like
avgSerSetupTime = stats.mean(serSetupTimes)

minSerSolveTime = min(serSolveTimes) 
maxSerSolveTime = max(serSolveTimes)
avgSerSolveTime = stats.mean(serSolveTimes)

minSerSolvePerc = min(serSolvePercs) 
maxSerSolvePerc = max(serSolvePercs)
avgSerSolvePerc = stats.mean(serSolvePercs)

minSerWrapTime = min(serWrapTimes) 
maxSerWrapTime = max(serWrapTimes)
avgSerWrapTime = stats.mean(serWrapTimes)

minSerTotalTime = min(serTotalTimes) 
maxSerTotalTime = max(serTotalTimes)
avgSerTotalTime = stats.mean(serTotalTimes)

#gpu
minGPUSetupTime = min(gpuSetupTimes) #don't care about percentages except solve
maxGPUSetupTime = max(gpuSetupTimes) #can change later if we like
avgGPUSetupTime = stats.mean(gpuSetupTimes)

minGPUSolveTime = min(gpuSolveTimes) 
maxGPUSolveTime = max(gpuSolveTimes)
avgGPUSolveTime = stats.mean(gpuSolveTimes)

minGPUSolvePerc = min(gpuSolvePercs) 
maxGPUSolvePerc = max(gpuSolvePercs)
avgGPUSolvePerc = stats.mean(gpuSolvePercs)

minGPUWrapTime = min(gpuWrapTimes) 
maxGPUWrapTime = max(gpuWrapTimes)
avgGPUWrapTime = stats.mean(gpuWrapTimes)

minGPUTotalTime = min(gpuTotalTimes) 
maxGPUTotalTime = max(gpuTotalTimes)
avgGPUTotalTime = stats.mean(gpuTotalTimes)

#
#save our test results
#
testdir = "test-results/"
filenum = 0
filename = testdir + "comparisontest_{:03d}.csv".format(filenum)
nodes = str(len(psi)*len(psi[0]))
print("Nodes: " + nodes)
while os.path.isfile(filename):
    filenum +=1
    filename = testdir + "comparisontest_{:03d}.csv".format(filenum)

with open(filename, 'w', newline='') as file:
    resultswriter = csv.writer(file, delimiter='|', quoting=csv.QUOTE_MINIMAL)

    #write original results
    resultswriter.writerow(["Original Algorithm", "Time (sec)", "Nodes: " + nodes])
    resultswriter.writerow(["Setup times", origSetupTimes])
    resultswriter.writerow(["Solve times", origSolveTimes])
    resultswriter.writerow(["Solve time %", origSolvePercs])
    resultswriter.writerow(["Wrap times", origWrapTimes])
    resultswriter.writerow(["Total times", origTotalTimes])

    #write serial results
    resultswriter.writerow(["Serial Algorithm", "Time (sec)", "Nodes: " + nodes])
    resultswriter.writerow(["Setup times", serSetupTimes])
    resultswriter.writerow(["Solve times", serSolveTimes])
    resultswriter.writerow(["Solve time %", serSolvePercs])
    resultswriter.writerow(["Wrap times", serWrapTimes])
    resultswriter.writerow(["Total times", serTotalTimes])

    #write gpu results
    resultswriter.writerow(["GPU Algorithm", "Time (sec)", "Nodes: " + nodes])
    resultswriter.writerow(["Setup times", gpuSetupTimes])
    resultswriter.writerow(["Solve times", gpuSolveTimes])
    resultswriter.writerow(["Solve time %", gpuSolvePercs])
    resultswriter.writerow(["Wrap times", gpuWrapTimes])
    resultswriter.writerow(["Total times", gpuTotalTimes])
