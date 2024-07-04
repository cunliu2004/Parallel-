#include <algorithm>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include "knncuda.h"



void initialize_data(float * ref,
                     int     ref_nb,
                     float * query,
                     int     query_nb,
                     int     dim) {

   
    srand(time(NULL));


    for (int i=0; i<ref_nb*dim; ++i) {
        ref[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }

 
    for (int i=0; i<query_nb*dim; ++i) {
        query[i] = 10. * (float)(rand() / (double)RAND_MAX);
    }
}



float compute_distance(const float * ref,
                       int           ref_nb,
                       const float * query,
                       int           query_nb,
                       int           dim,
                       int           ref_index,
                       int           query_index) {
    float sum = 0.f;
    for (int d=0; d<dim; ++d) {
        const float diff = ref[d * ref_nb + ref_index] - query[d * query_nb + query_index];
        sum += diff * diff;
    }
    return sqrtf(sum);
}



void  modified_insertion_sort(float *dist, int *index, int length, int k){

   
    index[0] = 0;

   
    for (int i=1; i<length; ++i) {

 
        float curr_dist  = dist[i];
        int   curr_index = i;

     
        if (i >= k && curr_dist >= dist[k-1]) {
            continue;
        }

    
        int j = std::min(i, k-1);
        while (j > 0 && dist[j-1] > curr_dist) {
            dist[j]  = dist[j-1];
            index[j] = index[j-1];
            --j;
        }

     
        dist[j]  = curr_dist;
        index[j] = curr_index; 
    }
}



bool knn_c(const float * ref,
           int           ref_nb,
           const float * query,
           int           query_nb,
           int           dim,
           int           k,
           float *       knn_dist,
           int *         knn_index) {


    float * dist  = (float *) malloc(ref_nb * sizeof(float));
    int *   index = (int *)   malloc(ref_nb * sizeof(int));

   
    if (!dist || !index) {
        printf("Memory allocation error\n");
        free(dist);
        free(index);
        return false;
    }

 
    for (int i=0; i<query_nb; ++i) {

       
        for (int j=0; j<ref_nb; ++j) {
            dist[j]  = compute_distance(ref, ref_nb, query, query_nb, dim, j, i);
            index[j] = j;
        }

   
        modified_insertion_sort(dist, index, ref_nb, k);

       
        for (int j=0; j<k; ++j) {
            knn_dist[j * query_nb + i]  = dist[j];
            knn_index[j * query_nb + i] = index[j];
        }
    }

   
    free(dist);
    free(index);

    return true;

}



bool test(const float * ref,
          int           ref_nb,
          const float * query,
          int           query_nb,
          int           dim,
          int           k,
          float *       gt_knn_dist,
          int *         gt_knn_index,
          bool (*knn)(const float *, int, const float *, int, int, int, float *, int *),
          const char *  name,
          int           nb_iterations) {

 
    const float precision    = 0.001f; 
    const float min_accuracy = 0.999f; 

   
    printf("- %-17s : ", name);

  
    float * test_knn_dist  = (float*) malloc(query_nb * k * sizeof(float));
    int   * test_knn_index = (int*)   malloc(query_nb * k * sizeof(int));

 
    if (!test_knn_dist || !test_knn_index) {
        printf("ALLOCATION ERROR\n");
        free(test_knn_dist);
        free(test_knn_index);
        return false;
    }


    struct timeval tic;
    gettimeofday(&tic, NULL);

  
    for (int i=0; i<nb_iterations; ++i) {
        if (!knn(ref, ref_nb, query, query_nb, dim, k, test_knn_dist, test_knn_index)) {
            free(test_knn_dist);
            free(test_knn_index);
            return false;
        }
    }

 
    struct timeval toc;
    gettimeofday(&toc, NULL);

  
    double elapsed_time = toc.tv_sec - tic.tv_sec;
    elapsed_time += (toc.tv_usec - tic.tv_usec) / 1000000.;

  
    int nb_correct_precisions = 0;
    int nb_correct_indexes    = 0;
    for (int i=0; i<query_nb*k; ++i) {
        if (fabs(test_knn_dist[i] - gt_knn_dist[i]) <= precision) {
            nb_correct_precisions++;
        }
        if (test_knn_index[i] == gt_knn_index[i]) {
            nb_correct_indexes++;
        }
    }

    
    float precision_accuracy = nb_correct_precisions / ((float) query_nb * k);
    float index_accuracy     = nb_correct_indexes    / ((float) query_nb * k);

  
    if (precision_accuracy >= min_accuracy && index_accuracy >= min_accuracy ) {
        printf("PASSED in %8.5f seconds (averaged over %3d iterations)\n", elapsed_time / nb_iterations, nb_iterations);
    }
    else {
        printf("FAILED\n");
    }

    
    free(test_knn_dist);
    free(test_knn_index);

    return true;
}



int main(void) {

   
    const int ref_nb   = 16384;
    const int query_nb = 4096;
    const int dim      = 128;
    const int k        = 16;

  
    printf("PARAMETERS\n");
    printf("- Number reference points : %d\n",   ref_nb);
    printf("- Number query points     : %d\n",   query_nb);
    printf("- Dimension of points     : %d\n",   dim);
    printf("- Number of neighbors     : %d\n\n", k);

   
    if (ref_nb<k) {
        printf("Error: k value is larger that the number of reference points\n");
        return EXIT_FAILURE;
    }

 
    float * ref        = (float*) malloc(ref_nb   * dim * sizeof(float));
    float * query      = (float*) malloc(query_nb * dim * sizeof(float));
    float * knn_dist   = (float*) malloc(query_nb * k   * sizeof(float));
    int   * knn_index  = (int*)   malloc(query_nb * k   * sizeof(int));


    if (!ref || !query || !knn_dist || !knn_index) {
        printf("Error: Memory allocation error\n"); 
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }

   
    initialize_data(ref, ref_nb, query, query_nb, dim);

  
    printf("Ground truth computation in progress...\n\n");
    if (!knn_c(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index)) {
        free(ref);
	    free(query);
	    free(knn_dist);
	    free(knn_index);
        return EXIT_FAILURE;
    }


    printf("TESTS\n");
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_c,            "knn_c",              2);
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_global,  "knn_cuda_global",  100); 
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cuda_texture, "knn_cuda_texture", 100); 
    test(ref, ref_nb, query, query_nb, dim, k, knn_dist, knn_index, &knn_cublas,       "knn_cublas",       100); 


    free(ref);
    free(query);
    free(knn_dist);
    free(knn_index);

    return EXIT_SUCCESS;
}
