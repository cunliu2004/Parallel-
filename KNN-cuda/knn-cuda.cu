#include <stdio.h>
#include <cuda.h>
#include <cublas.h>

#define BLOCK_DIM 16



__global__ void compute_distances(float * ref,
                                  int     ref_width,
                                  int     ref_pitch,
                                  float * query,
                                  int     query_width,
                                  int     query_pitch,
                                  int     height,
                                  float * dist) {

  
    __shared__ float shared_A[BLOCK_DIM][BLOCK_DIM];
    __shared__ float shared_B[BLOCK_DIM][BLOCK_DIM];

  
    __shared__ int begin_A;
    __shared__ int begin_B;
    __shared__ int step_A;
    __shared__ int step_B;
    __shared__ int end_A;


    int tx = threadIdx.x;
    int ty = threadIdx.y;

   
    float ssd = 0.f;


    begin_A = BLOCK_DIM * blockIdx.y;
    begin_B = BLOCK_DIM * blockIdx.x;
    step_A  = BLOCK_DIM * ref_pitch;
    step_B  = BLOCK_DIM * query_pitch;
    end_A   = begin_A + (height-1) * ref_pitch;

  
    int cond0 = (begin_A + tx < ref_width); 
    int cond1 = (begin_B + tx < query_width); 
    int cond2 = (begin_A + ty < ref_width); 

 
    for (int a = begin_A, b = begin_B; a <= end_A; a += step_A, b += step_B) {

   
        if (a/ref_pitch + ty < height) {
            shared_A[ty][tx] = (cond0)? ref[a + ref_pitch * ty + tx] : 0;
            shared_B[ty][tx] = (cond1)? query[b + query_pitch * ty + tx] : 0;
        }
        else {
            shared_A[ty][tx] = 0;
            shared_B[ty][tx] = 0;
        }

   
        __syncthreads();

        if (cond2 && cond1) {
            for (int k = 0; k < BLOCK_DIM; ++k){
                float tmp = shared_A[k][ty] - shared_B[k][tx];
                ssd += tmp*tmp;
            }
        }

       
        __syncthreads();
    }

 
    if (cond2 && cond1) {
        dist[ (begin_A + ty) * query_pitch + begin_B + tx ] = ssd;
    }
}



__global__ void compute_distance_texture(cudaTextureObject_t ref,
                                         int                 ref_width,
                                         float *             query,
                                         int                 query_width,
                                         int                 query_pitch,
                                         int                 height,
                                         float*              dist) {
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if ( xIndex<query_width && yIndex<ref_width) {
        float ssd = 0.f;
        for (int i=0; i<height; i++) {
            float tmp  = tex2D<float>(ref, (float)yIndex, (float)i) - query[i * query_pitch + xIndex];
            ssd += tmp * tmp;
        }
        dist[yIndex * query_pitch + xIndex] = ssd;
    }
}



__global__ void modified_insertion_sort(float * dist,
                                        int     dist_pitch,
                                        int *   index,
                                        int     index_pitch,
                                        int     width,
                                        int     height,
                                        int     k){


    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;


    if (xIndex < width) {

        float * p_dist  = dist  + xIndex;
        int *   p_index = index + xIndex;


        p_index[0] = 0;

 
        for (int i=1; i<height; ++i) {


            float curr_dist = p_dist[i*dist_pitch];
            int   curr_index  = i;

 
            if (i >= k && curr_dist >= p_dist[(k-1)*dist_pitch]) {
                continue;
            }


            int j = min(i, k-1);
            while (j > 0 && p_dist[(j-1)*dist_pitch] > curr_dist) {
                p_dist[j*dist_pitch]   = p_dist[(j-1)*dist_pitch];
                p_index[j*index_pitch] = p_index[(j-1)*index_pitch];
                --j;
            }

 
            p_dist[j*dist_pitch]   = curr_dist;
            p_index[j*index_pitch] = curr_index; 
        }
    }
}



__global__ void compute_sqrt(float * dist, int width, int pitch, int k){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        dist[yIndex*pitch + xIndex] = sqrt(dist[yIndex*pitch + xIndex]);
}



__global__ void compute_squared_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    if (xIndex<width){
        float sum = 0.f;
        for (int i=0; i<height; i++){
            float val = array[i*pitch+xIndex];
            sum += val*val;
        }
        norm[xIndex] = sum;
    }
}



__global__ void add_reference_points_norm(float * array, int width, int pitch, int height, float * norm){
    unsigned int tx = threadIdx.x;
    unsigned int ty = threadIdx.y;
    unsigned int xIndex = blockIdx.x * blockDim.x + tx;
    unsigned int yIndex = blockIdx.y * blockDim.y + ty;
    __shared__ float shared_vec[16];
    if (tx==0 && yIndex<height)
        shared_vec[ty] = norm[yIndex];
    __syncthreads();
    if (xIndex<width && yIndex<height)
        array[yIndex*pitch+xIndex] += shared_vec[ty];
}



__global__ void add_query_points_norm_and_sqrt(float * array, int width, int pitch, int k, float * norm){
    unsigned int xIndex = blockIdx.x * blockDim.x + threadIdx.x;
    unsigned int yIndex = blockIdx.y * blockDim.y + threadIdx.y;
    if (xIndex<width && yIndex<k)
        array[yIndex*pitch + xIndex] = sqrt(array[yIndex*pitch + xIndex] + norm[xIndex]);
}


bool knn_cuda_global(const float * ref,
                     int           ref_nb,
                     const float * query,
                     int           query_nb,
                     int           dim,
                     int           k,
                     float *       knn_dist,
                     int *         knn_index) {


    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);


    cudaError_t err0, err1, err2, err3;

    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }


    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }


    float * ref_dev   = NULL;
    float * query_dev = NULL;
    float * dist_dev  = NULL;
    int   * index_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    err0 = cudaMallocPitch((void**)&ref_dev,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    err3 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int,   k);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess) {
        printf("ERROR: Memory allocation error\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }


    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;


    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }


    err0 = cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,   ref,   ref_nb * size_of_float,   ref_nb * size_of_float,   dim, cudaMemcpyHostToDevice);
    err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }


    dim3 block0(BLOCK_DIM, BLOCK_DIM, 1);
    dim3 grid0(query_nb / BLOCK_DIM, ref_nb / BLOCK_DIM, 1);
    if (query_nb % BLOCK_DIM != 0) grid0.x += 1;
    if (ref_nb   % BLOCK_DIM != 0) grid0.y += 1;
    compute_distances<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, query_dev, query_nb, query_pitch, dim, dist_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }


    dim3 block1(256, 1, 1);
    dim3 grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0) grid1.x += 1;
    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }


    dim3 block2(16, 16, 1);
    dim3 grid2(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0) grid2.x += 1;
    if (k % 16 != 0)        grid2.y += 1;
    compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);	
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }


    err0 = cudaMemcpy2D(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch_in_bytes,  query_nb * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_nb * size_of_int,   index_dev, index_pitch_in_bytes, query_nb * size_of_int,   k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }

 
    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev); 

    return true;
}


bool knn_cuda_texture(const float * ref,
                      int           ref_nb,
                      const float * query,
                      int           query_nb,
                      int           dim,
                      int           k,
                      float *       knn_dist,
                      int *         knn_index) {


    unsigned int size_of_float = sizeof(float);
    unsigned int size_of_int   = sizeof(int);   

 
    cudaError_t err0, err1, err2;

 
    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }

  
    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }


    float * query_dev = NULL;
    float * dist_dev  = NULL;
    int *   index_dev = NULL;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    err0 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    err2 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int,   k);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess) {
        printf("ERROR: Memory allocation error (cudaMallocPitch)\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false;
    }

 
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;


    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev); 
        return false; 
    }


    err0 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);        
        return false; 
    }


    cudaArray* ref_array_dev = NULL;
    cudaChannelFormatDesc channel_desc = cudaCreateChannelDesc(32, 0, 0, 0, cudaChannelFormatKindFloat);
    err0 = cudaMallocArray(&ref_array_dev, &channel_desc, ref_nb, dim);
    if (err0 != cudaSuccess) {
        printf("ERROR: Memory allocation error (cudaMallocArray)\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        return false; 
    }


    err0 = cudaMemcpyToArray(ref_array_dev, 0, 0, ref, ref_nb * size_of_float * dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        return false; 
    }


    struct cudaResourceDesc res_desc;
    memset(&res_desc, 0, sizeof(res_desc));
    res_desc.resType         = cudaResourceTypeArray;
    res_desc.res.array.array = ref_array_dev;


    struct cudaTextureDesc tex_desc;
    memset(&tex_desc, 0, sizeof(tex_desc));
    tex_desc.addressMode[0]   = cudaAddressModeClamp;
    tex_desc.addressMode[1]   = cudaAddressModeClamp;
    tex_desc.filterMode       = cudaFilterModePoint;
    tex_desc.readMode         = cudaReadModeElementType;
    tex_desc.normalizedCoords = 0;


    cudaTextureObject_t ref_tex_dev = 0;
    err0 = cudaCreateTextureObject(&ref_tex_dev, &res_desc, &tex_desc, NULL);
    if (err0 != cudaSuccess) {
        printf("ERROR: Unable to create the texture\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        return false; 
    }


    dim3 block0(16, 16, 1);
    dim3 grid0(query_nb / 16, ref_nb / 16, 1);
    if (query_nb % 16 != 0) grid0.x += 1;
    if (ref_nb   % 16 != 0) grid0.y += 1;
    compute_distance_texture<<<grid0, block0>>>(ref_tex_dev, ref_nb, query_dev, query_nb, query_pitch, dim, dist_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }


    dim3 block1(256, 1, 1);
    dim3 grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0) grid1.x += 1;
    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }


    dim3 block2(16, 16, 1);
    dim3 grid2(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0) grid2.x += 1;
    if (k % 16 != 0)        grid2.y += 1;
    compute_sqrt<<<grid2, block2>>>(dist_dev, query_nb, query_pitch, k);	
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false;
    }


    err0 = cudaMemcpy2D(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch_in_bytes,  query_nb * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_nb * size_of_int,   index_dev, index_pitch_in_bytes, query_nb * size_of_int,   k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFreeArray(ref_array_dev);
        cudaDestroyTextureObject(ref_tex_dev);
        return false; 
    }


    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
    cudaFreeArray(ref_array_dev);
    cudaDestroyTextureObject(ref_tex_dev);

    return true;
}


bool knn_cublas(const float * ref,
                int           ref_nb,
                const float * query,
                int           query_nb,
                int           dim, 
                int           k, 
                float *       knn_dist,
                int *         knn_index) {


    const unsigned int size_of_float = sizeof(float);
    const unsigned int size_of_int   = sizeof(int);


    cudaError_t  err0, err1, err2, err3, err4, err5;


    int nb_devices;
    err0 = cudaGetDeviceCount(&nb_devices);
    if (err0 != cudaSuccess || nb_devices == 0) {
        printf("ERROR: No CUDA device found\n");
        return false;
    }


    err0 = cudaSetDevice(0);
    if (err0 != cudaSuccess) {
        printf("ERROR: Cannot set the chosen CUDA device\n");
        return false;
    }


    cublasInit();

    // Allocate global memory
    float * ref_dev        = NULL;
    float * query_dev      = NULL;
    float * dist_dev       = NULL;
    int   * index_dev      = NULL;
    float * ref_norm_dev   = NULL;
    float * query_norm_dev = NULL;
    size_t  ref_pitch_in_bytes;
    size_t  query_pitch_in_bytes;
    size_t  dist_pitch_in_bytes;
    size_t  index_pitch_in_bytes;
    err0 = cudaMallocPitch((void**)&ref_dev,   &ref_pitch_in_bytes,   ref_nb   * size_of_float, dim);
    err1 = cudaMallocPitch((void**)&query_dev, &query_pitch_in_bytes, query_nb * size_of_float, dim);
    err2 = cudaMallocPitch((void**)&dist_dev,  &dist_pitch_in_bytes,  query_nb * size_of_float, ref_nb);
    err3 = cudaMallocPitch((void**)&index_dev, &index_pitch_in_bytes, query_nb * size_of_int,   k);
    err4 = cudaMalloc((void**)&ref_norm_dev,   ref_nb   * size_of_float);
    err5 = cudaMalloc((void**)&query_norm_dev, query_nb * size_of_float);
    if (err0 != cudaSuccess || err1 != cudaSuccess || err2 != cudaSuccess || err3 != cudaSuccess || err4 != cudaSuccess || err5 != cudaSuccess) {
        printf("ERROR: Memory allocation error\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;
    }


    size_t ref_pitch   = ref_pitch_in_bytes   / size_of_float;
    size_t query_pitch = query_pitch_in_bytes / size_of_float;
    size_t dist_pitch  = dist_pitch_in_bytes  / size_of_float;
    size_t index_pitch = index_pitch_in_bytes / size_of_int;


    if (query_pitch != dist_pitch || query_pitch != index_pitch) {
        printf("ERROR: Invalid pitch value\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false; 
    }


    err0 = cudaMemcpy2D(ref_dev,   ref_pitch_in_bytes,   ref,   ref_nb * size_of_float,   ref_nb * size_of_float,   dim, cudaMemcpyHostToDevice);
    err1 = cudaMemcpy2D(query_dev, query_pitch_in_bytes, query, query_nb * size_of_float, query_nb * size_of_float, dim, cudaMemcpyHostToDevice);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from host to device\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false; 
    }


    dim3 block0(256, 1, 1);
    dim3 grid0(ref_nb / 256, 1, 1);
    if (ref_nb % 256 != 0) grid0.x += 1;
    compute_squared_norm<<<grid0, block0>>>(ref_dev, ref_nb, ref_pitch, dim, ref_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;
    }


    dim3 block1(256, 1, 1);
    dim3 grid1(query_nb / 256, 1, 1);
    if (query_nb % 256 != 0) grid1.x += 1;
    compute_squared_norm<<<grid1, block1>>>(query_dev, query_nb, query_pitch, dim, query_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;
    }


    cublasSgemm('n', 't', (int)query_pitch, (int)ref_pitch, dim, (float)-2.0, query_dev, query_pitch, ref_dev, ref_pitch, (float)0.0, dist_dev, query_pitch);
    if (cublasGetError() != CUBLAS_STATUS_SUCCESS) {
        printf("ERROR: Unable to execute cublasSgemm\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;       
    }


    dim3 block2(16, 16, 1);
    dim3 grid2(query_nb / 16, ref_nb / 16, 1);
    if (query_nb % 16 != 0) grid2.x += 1;
    if (ref_nb   % 16 != 0) grid2.y += 1;
    add_reference_points_norm<<<grid2, block2>>>(dist_dev, query_nb, dist_pitch, ref_nb, ref_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;
    }


    modified_insertion_sort<<<grid1, block1>>>(dist_dev, dist_pitch, index_dev, index_pitch, query_nb, ref_nb, k);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;
    }


    dim3 block3(16, 16, 1);
    dim3 grid3(query_nb / 16, k / 16, 1);
    if (query_nb % 16 != 0) grid3.x += 1;
    if (k        % 16 != 0) grid3.y += 1;
    add_query_points_norm_and_sqrt<<<grid3, block3>>>(dist_dev, query_nb, dist_pitch, k, query_norm_dev);
    if (cudaGetLastError() != cudaSuccess) {
        printf("ERROR: Unable to execute kernel\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false;
    }


    err0 = cudaMemcpy2D(knn_dist,  query_nb * size_of_float, dist_dev,  dist_pitch_in_bytes,  query_nb * size_of_float, k, cudaMemcpyDeviceToHost);
    err1 = cudaMemcpy2D(knn_index, query_nb * size_of_int,   index_dev, index_pitch_in_bytes, query_nb * size_of_int,   k, cudaMemcpyDeviceToHost);
    if (err0 != cudaSuccess || err1 != cudaSuccess) {
        printf("ERROR: Unable to copy data from device to host\n");
        cudaFree(ref_dev);
        cudaFree(query_dev);
        cudaFree(dist_dev);
        cudaFree(index_dev);
        cudaFree(ref_norm_dev);
        cudaFree(query_norm_dev);
        cublasShutdown();
        return false; 
    }


    cudaFree(ref_dev);
    cudaFree(query_dev);
    cudaFree(dist_dev);
    cudaFree(index_dev);
    cudaFree(ref_norm_dev);
    cudaFree(query_norm_dev);
    cublasShutdown();

    return true;
}
