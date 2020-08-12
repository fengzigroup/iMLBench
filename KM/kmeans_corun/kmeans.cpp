#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include <string>
#include "kmeans.h"

#ifdef WIN
#include <windows.h>
#else
#include <pthread.h>
#include <sys/time.h>
double gettime()
{
    struct timeval t;
    gettimeofday(&t, NULL);
    return t.tv_sec + t.tv_usec * 1e-6;
}
#endif

#ifdef NV
#include <oclUtils.h>
#else
#include <CL/cl.h>
#endif

#ifndef FLT_MAX
#define FLT_MAX 3.40282347e+38
#endif

#ifdef RD_WG_SIZE_0_0
#define BLOCK_SIZE RD_WG_SIZE_0_0
#elif defined(RD_WG_SIZE_0)
#define BLOCK_SIZE RD_WG_SIZE_0
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE RD_WG_SIZE
#else
#define BLOCK_SIZE 256
#endif

#ifdef RD_WG_SIZE_1_0
#define BLOCK_SIZE2 RD_WG_SIZE_1_0
#elif defined(RD_WG_SIZE_1)
#define BLOCK_SIZE2 RD_WG_SIZE_1
#elif defined(RD_WG_SIZE)
#define BLOCK_SIZE2 RD_WG_SIZE
#else
#define BLOCK_SIZE2 256
#endif

// co_run
bool gpu_run = true, cpu_run = false;
int work_dim = 1;
size_t cpu_global_size[3] = {0, 1, 1};
size_t gpu_global_size[3] = {0, 1, 1};
size_t global_offset[2] = {0, 0};
size_t global_work[3] = {0, 1, 1};
size_t local_work_size = BLOCK_SIZE;

int cpu_offset = 0;

// local variables
static cl_context context;
static cl_command_queue cmd_queue;
static cl_device_type device_type;
static cl_device_id *device_list;
static cl_int num_devices;

void kmeansOMP(float  *feature,   
			   float  *clusters,
			    int    *membership,
			    int     npoints,
			    int     cpusize,
				int     nclusters,
				int     nfeatures,
				int		offset,
				int		size
			  ) 
{

	//printf("OMP size: %d\n",cpusize);
#pragma omp parallel for num_threads(threadsNum)
	for (int j = 0; j < cpusize; j++) {
		/*
		int num = omp_get_num_threads();
		int rank = omp_get_thread_num();
		printf("Total num: %d, Rank : %d , j = %d\n",num, rank, j);
*/
		int index = 0;
		float min_dist=FLT_MAX;
		//printf("\npoint: %d\n", j);
//#pragma omp parallel for shared(min_dist, index)
		for(int i = 0; i < nclusters; i++){
			float dist = 0;
			float ans  = 0;
#pragma omp simd reduction(+:ans) 
			for (int l = 0; l < nfeatures; l++){
				float tmp = feature[l * npoints + j] - clusters[i * nfeatures + l];
				ans += tmp * tmp;			
			}
			dist = ans;   
			if (dist < min_dist) {
				min_dist = dist;
				index    = i;	
			}
		}
		membership[j] = index;
	}	
	return;
}

static int initialize(int use_gpu)
{
    cl_int result;
    size_t size;

    // create OpenCL context
    cl_platform_id platform_id;
    if (clGetPlatformIDs(1, &platform_id, NULL) != CL_SUCCESS)
    {
        printf("ERROR: clGetPlatformIDs(1,*,0) failed\n");
        return -1;
    }
    cl_context_properties ctxprop[] = {CL_CONTEXT_PLATFORM, (cl_context_properties)platform_id, 0};
    device_type = use_gpu ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU;
    context = clCreateContextFromType(ctxprop, device_type, NULL, NULL, NULL);
    if (!context)
    {
        printf("ERROR: clCreateContextFromType(%s) failed\n", use_gpu ? "GPU" : "CPU");
        return -1;
    }

    // get the list of GPUs
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &size);
    num_devices = (int)(size / sizeof(cl_device_id));

    if (result != CL_SUCCESS || num_devices < 1)
    {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }
    device_list = new cl_device_id[num_devices];
    if (!device_list)
    {
        printf("ERROR: new cl_device_id[] failed\n");
        return -1;
    }
    result = clGetContextInfo(context, CL_CONTEXT_DEVICES, size, device_list, NULL);
    if (result != CL_SUCCESS)
    {
        printf("ERROR: clGetContextInfo() failed\n");
        return -1;
    }

    // create command queue for the first device
    cmd_queue = clCreateCommandQueue(context, device_list[0], 0, NULL);
    if (!cmd_queue)
    {
        printf("ERROR: clCreateCommandQueue() failed\n");
        return -1;
    }

    return 0;
}

static int shutdown()
{
    // release resources
    if (cmd_queue)
        clReleaseCommandQueue(cmd_queue);
    if (context)
        clReleaseContext(context);
    if (device_list)
        delete device_list;

    // reset all variables
    cmd_queue = 0;
    context = 0;
    device_list = 0;
    num_devices = 0;
    device_type = 0;

    return 0;
}

cl_mem d_feature;
cl_mem d_feature_swap;
cl_mem d_cluster;
cl_mem d_membership;

cl_kernel kernel;
cl_kernel kernel_s;
cl_kernel kernel2;

int *membership_OCL;
int *membership_d;
float *feature_d;
float *clusters_d;
float *center_d;

int allocate(int n_points, int n_features, int n_clusters, float **feature)
{

    int sourcesize = 1024 * 1024;
    char *source = (char *)calloc(sourcesize, sizeof(char));
    if (!source)
    {
        printf("ERROR: calloc(%d) failed\n", sourcesize);
        return -1;
    }

    // read the kernel core source
    char *tempchar = "./kmeans.cl";
    FILE *fp = fopen(tempchar, "rb");
    if (!fp)
    {
        printf("ERROR: unable to open '%s'\n", tempchar);
        return -1;
    }
    fread(source + strlen(source), sourcesize, 1, fp);
    fclose(fp);

    // OpenCL initialization
    int use_gpu = 1;
    if (initialize(use_gpu))
        return -1;

    // compile kernel
    cl_int err = 0;
    const char *slist[2] = {source, 0};
    cl_program prog = clCreateProgramWithSource(context, 1, slist, NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateProgramWithSource() => %d\n", err);
        return -1;
    }
    err = clBuildProgram(prog, 0, NULL, NULL, NULL, NULL);
    { // show warnings/errors
        //	static char log[65536]; memset(log, 0, sizeof(log));
        //	cl_device_id device_id = 0;
        //	err = clGetContextInfo(context, CL_CONTEXT_DEVICES, sizeof(device_id), &device_id, NULL);
        //	clGetProgramBuildInfo(prog, device_id, CL_PROGRAM_BUILD_LOG, sizeof(log)-1, log, NULL);
        //	if(err || strstr(log,"warning:") || strstr(log, "error:")) printf("<<<<\n%s\n>>>>\n", log);
    }
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clBuildProgram() => %d\n", err);
        return -1;
    }

    char *kernel_kmeans_c = "kmeans_kernel_c";
    char *kernel_swap = "kmeans_swap";

    kernel_s = clCreateKernel(prog, kernel_kmeans_c, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateKernel() 0 => %d\n", err);
        return -1;
    }
    kernel2 = clCreateKernel(prog, kernel_swap, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateKernel() 0 => %d\n", err);
        return -1;
    }

    clReleaseProgram(prog);

    d_feature = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateBuffer d_feature (size:%d) => %d\n", n_points * n_features, err);
        return -1;
    }
    d_feature_swap = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * n_features * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateBuffer d_feature_swap (size:%d) => %d\n", n_points * n_features, err);
        return -1;
    }
    d_cluster = clCreateBuffer(context, CL_MEM_READ_WRITE, n_clusters * n_features * sizeof(float), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateBuffer d_cluster (size:%d) => %d\n", n_clusters * n_features, err);
        return -1;
    }
    d_membership = clCreateBuffer(context, CL_MEM_READ_WRITE, n_points * sizeof(int), NULL, &err);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clCreateBuffer d_membership (size:%d) => %d\n", n_points, err);
        return -1;
    }

    //write buffers
    err = clEnqueueWriteBuffer(cmd_queue, d_feature, 1, 0, n_points * n_features * sizeof(float), feature[0], 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clEnqueueWriteBuffer d_feature (size:%d) => %d\n", n_points * n_features, err);
        return -1;
    }

    clSetKernelArg(kernel2, 0, sizeof(void *), (void *)&d_feature);
    clSetKernelArg(kernel2, 1, sizeof(void *), (void *)&d_feature_swap);
    clSetKernelArg(kernel2, 2, sizeof(cl_int), (void *)&n_points);
    clSetKernelArg(kernel2, 3, sizeof(cl_int), (void *)&n_features);

    size_t global_work[3] = {n_points, 1, 1};
    /// Ke Wang adjustable local group size 2013/08/07 10:37:33
    size_t local_work_size = BLOCK_SIZE; // work group size is defined by RD_WG_SIZE_0 or RD_WG_SIZE_0_0 2014/06/10 17:00:51
    if (global_work[0] % local_work_size != 0)
        global_work[0] = (global_work[0] / local_work_size + 1) * local_work_size;

    global_work[0] = n_points;
    if (global_work[0] % local_work_size != 0)
        global_work[0] = (global_work[0] / local_work_size + 1) * local_work_size;

    if (cpu_offset > 0)
    {
        cpu_run = true;
    }
    cpu_global_size[0] = cpu_offset * global_work[0] / 100;
    if (cpu_global_size[0] % local_work_size != 0)
    {
        cpu_global_size[0] = (1 + cpu_global_size[0] / local_work_size) * local_work_size;
    }
    gpu_global_size[0] = global_work[0] - cpu_global_size[0];
    if (gpu_global_size[0] <= 0)
    {
        gpu_run = false;
    }
    global_offset[0] = cpu_global_size[0];
    printf("CPU size: %d, GPU size: %d\n", cpu_global_size[0], gpu_global_size[0]);

    err = clEnqueueNDRangeKernel(cmd_queue, kernel2, 1, NULL, global_work, &local_work_size, 0, 0, 0);
    if (err != CL_SUCCESS)
    {
        printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err);
        return -1;
    }

    membership_OCL = (int *)malloc(n_points * sizeof(int));
}

void deallocateMemory()
{
    clReleaseMemObject(d_feature);
    clReleaseMemObject(d_feature_swap);
    clReleaseMemObject(d_cluster);
    clReleaseMemObject(d_membership);
    free(membership_OCL);
}

int main(int argc, char **argv)
{
    printf("WG size of kernel_swap = %d, WG size of kernel_kmeans = %d, work size = %d \n", BLOCK_SIZE, BLOCK_SIZE2, CL_DEVICE_MAX_WORK_GROUP_SIZE);
    //	int corenum = omp_get_num_procs();
    //	printf("Core number: %d\n", corenum);
    setup(argc, argv);
    shutdown();
}

int kmeansOCL(float **feature, /* in: [npoints][nfeatures] */
              int n_features,
              int n_points,
              int n_clusters,
              int *membership,
              float **clusters,
              int *new_centers_len,
              float **new_centers)
{

    int delta = 0;
    int i, j, k;
    cl_int err = 0;
    static int count = 0;

    size_t global_work[3] = {n_points, 1, 1};

    /// Ke Wang adjustable local group size 2013/08/07 10:37:33
    size_t local_work_size = BLOCK_SIZE2; // work group size is defined by RD_WG_SIZE_1 or RD_WG_SIZE_1_0 2014/06/10 17:00:41
    if (global_work[0] % local_work_size != 0)
        global_work[0] = (global_work[0] / local_work_size + 1) * local_work_size;

    float *d_cluster_cpu = (float *)malloc(n_clusters * n_features * sizeof(float));
    clEnqueueReadBuffer(cmd_queue, d_cluster, CL_TRUE, 0, n_clusters * n_features * sizeof(float),
                        d_cluster_cpu, 0, 0, 0);

    if (err != CL_SUCCESS)
    {
        printf("ERROR: clEnqueueWriteBuffer d_cluster (size:%d) => %d\n", n_points, err);
        return -1;
    }

    int size = 0;
    int offset = 0;

    cl_event eventList;

    float *d_feature_swap_cpu = (float *)malloc(n_points * n_features * sizeof(float));
    int *d_membership_cpu = (int *)malloc(n_points * sizeof(int));

    double time1 = gettime();

    clEnqueueWriteBuffer(cmd_queue, d_cluster, 1, 0, n_clusters * n_features * sizeof(float),
                         clusters[0], 0, 0, 0);
    clEnqueueReadBuffer(cmd_queue, d_feature_swap, CL_TRUE, 0, n_points * n_features * sizeof(float),
                        d_feature_swap_cpu, 0, 0, 0);

    //count++;
    clEnqueueReadBuffer(cmd_queue, d_membership, CL_TRUE, 0, n_points * sizeof(int),
                        d_membership_cpu, 0, 0, 0);
    double time2 = gettime();
    printf("Read data time: %lf ms\n", 1000.0 * (time2 - time1));

    double t1 = gettime();

    if (gpu_run)
    {
                double t1 = gettime();

        clSetKernelArg(kernel_s, 0, sizeof(void *), (void *)&d_feature_swap);
        clSetKernelArg(kernel_s, 1, sizeof(void *), (void *)&d_cluster);
        clSetKernelArg(kernel_s, 2, sizeof(void *), (void *)&d_membership);
        clSetKernelArg(kernel_s, 3, sizeof(cl_int), (void *)&n_points);
        clSetKernelArg(kernel_s, 4, sizeof(cl_int), (void *)&n_clusters);
        clSetKernelArg(kernel_s, 5, sizeof(cl_int), (void *)&n_features);
        clSetKernelArg(kernel_s, 6, sizeof(cl_int), (void *)&offset);
        clSetKernelArg(kernel_s, 7, sizeof(cl_int), (void *)&size);

        err = clEnqueueNDRangeKernel(cmd_queue, kernel_s, 1, global_offset, gpu_global_size, &local_work_size, 0, 0, 0);
        if (err != CL_SUCCESS)
        {
            printf("ERROR: clEnqueueNDRangeKernel()=>%d failed\n", err);
            return -1;
        }
        clFinish(cmd_queue);
                double t2 = gettime();
        printf("GPU time: %lf ms\n", 1000*(t2-t1));
    }

    if (cpu_run)
    {
        double time1 = gettime();
    printf("cpu_global_size[0]:%d\n",cpu_global_size[0]);
        kmeansOMP((float *)d_feature_swap_cpu, (float *)d_cluster_cpu, membership_OCL,
                  n_points, cpu_global_size[0], n_clusters, n_features, 0, 0);
        //kmeansOMP((float *) d_feature_swap, (float *) d_cluster, (int *) d_membership, 
                //n_points, cpu_global_size[0], n_clusters, n_features, 0, 0); 

        double time2 = gettime();
        printf("OMP time: %lf ms\n", 1000.0 * (time2 - time1));
    }
/*
    if (gpu_run)
    {
        clFinish(cmd_queue);
        err = clEnqueueReadBuffer(cmd_queue, d_membership, 1, 0, n_points * sizeof(int), membership_OCL, 0, 0, 0);
        if (err != CL_SUCCESS)
        {
            printf("ERROR: Memcopy Out\n");
            return -1;
        }
    }*/
    double t2 = gettime();
    printf("One iteration time: %f ms\n", 1000.0 * (t2 - t1));

    delta = 0;
    for (i = 0; i < n_points; i++)
    {
        int cluster_id = membership_OCL[i];
        new_centers_len[cluster_id]++;
        if (membership_OCL[i] != membership[i])
        {
            delta++;
            membership[i] = membership_OCL[i];
        }
        for (j = 0; j < n_features; j++)
        {
            new_centers[cluster_id][j] += feature[i][j];
        }
    }

    return delta;
}
