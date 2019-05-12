#include <cublas.h>
#include <cusparse.h>
#include <cstdlib>
#include <ctime>

/*library with routines to run custom conjugate gradient solver on the GPU using cuSparse and related. Called in a Python script
 * Owes inspiration to the packaged CUDA samples including the Conjugate Gradient samples, which can be found here: https://github.com/NVIDIA/cuda-samples/tree/master/Samples/conjugateGradientCudaGraphs
 */

double gpuCGSolve(const int &len, const int &nnz, double *Arow, double *Acol, double *Adat, double *b, double *x) {
	/*Given matrix A and vector b in CSR format (and the length of the vector len) solve Ax = b for x using conjugate gradient
	/with a CUDA-capable GPU. Does no preconditioning.*/
	
	
	const int max_iter = 1000;
	const int nz = len - nnz;
	int k;
	//		int k, M = 0, N = 0, nz = 0, *I = NULL, *J = NULL;
	int *d_col, *d_row;
	//		int qatest = 0;
	const float tol = 1e-10f;
	//		double *x, *rhs;
	double r0, r1, alpha, beta;
	double *d_val, *d_x;
	//double *d_zm1, *d_zm2, *d_rm2;
	double *d_b, *d_p, *d_omega, *d_y;

	//		float *d_valsILU0;
	//		float *valsILU0;
	double rsum, diff, err = 0.0;
	//		float qaerr1, qaerr2 = 0.0;
	double dot, numerator, denominator, nalpha;
	const double floatone = 1.0; //??
	const double floatzero = 0.0;

	
	
	std::clock_t startTime = clock();
	
//	/* This will pick the best possible CUDA capable device */
//	cudaDeviceProp deviceProp;
////	int devID = findCudaDevice(argc, (const char **)argv);
////	printf("GPU selected Device ID = %d \n", devID);
//
//	if (devID < 0)
//	{
//	printf("Invalid GPU device %d selected, exiting...\n", devID);
//	exit(EXIT_SUCCESS);
//	}
//
//	checkCudaErrors(cudaGetDeviceProperties(&deviceProp, devID));
//
//	/* Statistics about the GPU device */
//	printf("> GPU device has %d Multi-Processors, SM %d.%d compute capabilities\n\n",
//	deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor);
//
//	int version = (deviceProp.major * 0x10 + deviceProp.minor);
//
//	if (version < 0x11)
//	{
//	printf("%s: requires a minimum CUDA compute 1.1 capability\n", sSDKname);
//	cudaDeviceReset();
//	exit(EXIT_SUCCESS);
//	}
	
	/* Create CUBLAS context */
	cublasHandle_t cublasHandle;
	cublasStatus_t cublasStatus;
	cublasStatus = cublasCreate(&cublasHandle);

//	checkCudaErrors(cublasStatus); //probably unnecessary
	
	/* Create CUSPARSE context */
	cusparseHandle_t cusparseHandle;
	cusparseStatus_t cusparseStatus;
	cusparseStatus = cusparseCreate(&cusparseHandle);

//	checkCudaErrors(cusparseStatus);

	/* Description of the A matrix*/
	cusparseMatDescr_t desc;
	cusparseStatus = cusparseCreateMatDescr(&desc);
	
	/* Define the properties of the matrix */
	cusparseSetMatType(desc,CUSPARSE_MATRIX_TYPE_GENERAL);
	cusparseSetMatIndexBase(desc,CUSPARSE_INDEX_BASE_ZERO);
	
	/* Allocate required memory */
	// M, N are ints for matrix dim, correspond to my len
	//nz is int for number of zeros, correspond to nz (len - len(Adat))
	//I, J are arrays for row, column in CSR format (len(I) = N+1) correspond to  Arow, Acol
	//above notes are for conversion, remove when done
	cudaMalloc((void **)&d_col, nz*sizeof(int));
	cudaMalloc((void **)&d_row, (len+1)*sizeof(int));
	cudaMalloc((void **)&d_val, nz*sizeof(float));
	cudaMalloc((void **)&d_x, len*sizeof(float));
	cudaMalloc((void **)&d_y, len*sizeof(float));
	cudaMalloc((void **)&d_b, len*sizeof(float));
	cudaMalloc((void **)&d_p, len*sizeof(float));
	cudaMalloc((void **)&d_omega, len*sizeof(float));

	cudaMemcpy(d_col, Acol, nz*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_row, Arow, (len+1)*sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(d_val, Adat, nz*sizeof(float), cudaMemcpyHostToDevice); 
	cudaMemcpy(d_x, x, len*sizeof(float), cudaMemcpyHostToDevice);
	cudaMemcpy(d_b, b, len*sizeof(float), cudaMemcpyHostToDevice); //check for rhs, b -> d_b
	
	
	/* Conjugate gradient without preconditioning.
------------------------------------------
Follows the description by Golub & Van Loan, "Matrix Computations 3rd ed.", Section 10.2.6

Alternatively, see http://adl.stanford.edu/aa222/Lecture_Notes_files/CG_lecture.pdf */
	

	//printf("Convergence of conjugate gradient without preconditioning: \n");
	k = 0;
	r0 = 0;
	cublasDdot(cublasHandle, len, d_b, 1, d_b, 1, &r1);

	while (r1 > tol*tol && k <= max_iter)
	{
		k++;

		if (k == 1)
		{
			cublasDcopy(cublasHandle, len, d_b, 1, d_p, 1); //first iteration with first guess, find p vectors
		}
		else
		{
			beta = r1/r0;
			cublasDscal(cublasHandle, len, &beta, d_p, 1);
			cublasDaxpy(cublasHandle, len, &floatone, d_b, 1, d_p, 1); //find p vectors this way
		}

		cusparseDcsrmv(cusparseHandle,CUSPARSE_OPERATION_NON_TRANSPOSE, len, len, nz, &floatone, descr, d_val, d_row, d_col, d_p, &floatzero, d_omega);
		cublasDdot(cublasHandle, len, d_p, 1, d_omega, 1, &dot);
		alpha = r1/dot;
		cublasDaxpy(cublasHandle, len, &alpha, d_p, 1, d_x, 1);
		nalpha = -alpha;
		cublasDaxpy(cublasHandle, len, &nalpha, d_omega, 1, d_b, 1);
		r0 = r1;
		cublasDdot(cublasHandle, len, d_b, 1, d_b, 1, &r1);
	}

	printf(" iteration = %3d, residual = %e \n", k, sqrt(r1));
	cudaThreadSynchronize();//probably unnecessary; I think this is handled by the API, since I'm not explicitly creating threads anyway
	cudaMemcpy(x, d_x, len*sizeof(float), cudaMemcpyDeviceToHost);

	double duration = double( clock() - startTime ) / (double)CLOCKS_PER_SEC;
	std::cout << "time to solve Ax=b: " << duration << std::endl; //change to drop to file, or just extract to python time.time

	/* check result */
	err = 0.0;

	for (int i = 0; i < len; i++)
	{
		rsum = 0.0;

		for (int j = Arow[i]; j < Arow[i+1]; j++)
		{
			rsum += Adat[j]*x[Acol[j]];
		}

		diff = fabs(rsum - b[i]);

		if (diff > err)
		{
			err = diff;
		}
	}
	
	//clean up memory
	
	
	// Destroy parameters 
		cusparseDestroySolveAnalysisInfo(infoA);
		cusparseDestroySolveAnalysisInfo(info_u);

		// Destroy contexts
		cusparseDestroy(cusparseHandle);
		cublasDestroy(cublasHandle);

		// Free device memory 
		cudaFree(d_col);
		cudaFree(d_row);
		cudaFree(d_val);
		cudaFree(d_x);
		cudaFree(d_y);
		cudaFree(d_b);
		cudaFree(d_p);
		cudaFree(d_omega);
//		cudaFree(d_zm1);
//		cudaFree(d_zm2);
//		cudaFree(d_rm2);

		cudaDeviceReset();
	
	
	return(err); //in case we want to do some checking


}