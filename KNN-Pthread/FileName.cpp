#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <immintrin.h>
#include <time.h>
#include <cstdlib>
#include <fstream>
#include <pthread.h>

using namespace std;

const int threadcount = 4;
//存储数据点的特征和标签
struct DataPoint
{
    vector<double> features;
    int label;
};
//计算欧式距离
double euclidean_distance(const vector<double>& a, const vector<double>& b) {
    double distance = 0.0;
    //    cout<<"a.size: "<<a.size()<<endl;  //128
    //    cout<<"b.size: "<<b.size()<<endl;  //128
    //确保两个向量长度相等
    for (size_t i = 0; i < a.size(); i++)
    {
        distance += pow(a[i] - b[i], 2);
    }
    return sqrt(distance);
}
//对测试点进行KNN分类
int knn_predict(const vector<DataPoint>& training_set, const vector<double>& test_point, int k) {
    vector<pair<double, int>> distances(training_set.size());
    //    cout<<"training_set.size: "<<training_set.size()<<endl;  //10000
    for (size_t i = 0; i < training_set.size(); i++)
    {
        double distance = euclidean_distance(training_set[i].features, test_point);
        distances[i] = make_pair(distance, training_set[i].label);
    }
    sort(distances.begin(), distances.end());
    //    for (int i = 0; i < training_set.size(); i++)
    //    {
    //        cout<<distances[i].first<<endl;
    //    }
    //升序
    vector<int> k_nearest_neighbors(k);
    for (int i = 0; i < k; i++)
    {
        k_nearest_neighbors[i] = distances[i].second;
    }
    sort(k_nearest_neighbors.begin(), k_nearest_neighbors.end());
    int max_count = 1;
    //定义众数
    int mode = k_nearest_neighbors[0];
    //计算连续的标签出现的次数
    int count = 1;
    for (int i = 1; i < k; i++)
    {
        if (k_nearest_neighbors[i] == k_nearest_neighbors[i - 1])
        {
            count++;
        }
        else
        {
            if (count > max_count)
            {
                max_count = count;
                mode = k_nearest_neighbors[i - 1];
            }
            count = 1;
        }
    }
    if (count > max_count)
    {
        mode = k_nearest_neighbors[k - 1];
    }
    return mode;
}
//SIMD计算距离
double euclidean_distance_parallel(vector<double>& a, vector<double>& b)
{
    double distance = 0.0;
    __m256d a_reg, b_reg, d_reg, sum_reg = _mm256_setzero_pd();
    for (size_t i = 0; i < a.size(); i += 4)
    {
        a_reg = _mm256_loadu_pd(&a[i]);
        b_reg = _mm256_loadu_pd(&b[i]);
        d_reg = _mm256_sub_pd(a_reg, b_reg);
        sum_reg = _mm256_fmadd_pd(d_reg, d_reg, sum_reg);
    }
    double sum[4];
    _mm256_storeu_pd(sum, sum_reg);
    for (int i = 0; i < 4; i++)
    {
        distance += sum[i];
    }
    for (size_t i = a.size() - (a.size() % 4); i < a.size(); i++)
    {
        distance += pow(a[i] - b[i], 2);
    }
    return sqrt(distance);
}
//定义线程环境传递数据
struct threaddata
{
    vector<DataPoint>* training_set;
    vector<double>* test_point;
    vector< pair<double, int>>* distances;
    int i = 4;
    int num;
    threaddata() {}
    threaddata(vector<DataPoint>* training_set,
        vector<double>* test_point,
        vector< pair<double, int>>* distances,
        int i, int num) {
        this->set_values(training_set, test_point, distances, i, num);
    }
    void set_values(vector<DataPoint>* training_set,
        vector<double>* test_point,
        vector< pair<double, int>>* distances,
        int i, int num) {
        this->training_set = training_set;
        this->test_point = test_point;
        this->distances = distances;
        this->i = i;
        this->num = num;
    }
};

typedef struct threaddata ThreadData;
//定义多线程计算距离函数
void* cal_dis_pthread(void* arg)
{
    ThreadData* data = (ThreadData*)(arg);
    size_t id = data->i;
    for (int i = id * (data->num); i < (id + 1) * (data->num); i++)
    {
        double distance = euclidean_distance_parallel(data->training_set[0][i].features, data->test_point[0]);
        data->distances[0][i] = make_pair(distance, data->training_set[0][i].label);
    }
    return NULL;
}
//多线程预测测试点类别
int knn_predict_parallel(vector<DataPoint>& training_set, vector<double>& test_point, int k)
{
    //存储每个训练点与测试点之间的距离和对应的标签
    vector<pair<double, int>> distances(training_set.size());
    pthread_t* tid;
    //计算每个线程需要处理的数据点数量
    int per_num = training_set.size() / 4;
    tid = (pthread_t*)malloc(threadcount * sizeof(pthread_t));

    ThreadData data[threadcount];
    for (int i = 0; i < threadcount; i++)
    {
        data[i].set_values(&training_set, &test_point, &distances, i, per_num);
    }
    //创建线程
    for (int i = 0; i < threadcount; i++)
    {
        pthread_create(&tid[i], NULL, cal_dis_pthread, (void*)&data[i]);
    }
    //等待线程完成计算
    for (int i = 0; i < threadcount; i++)
    {
        pthread_join(tid[i], NULL);
    }
    free(tid);

    sort(distances.begin(), distances.end());
    //记录类别标签
    vector<int> k_nearest_neighbors(k);
    for (int i = 0; i < k; i++)
    {
        k_nearest_neighbors[i] = distances[i].second;
    }
    sort(k_nearest_neighbors.begin(), k_nearest_neighbors.end());
    int max_count = 1;
    int mode = k_nearest_neighbors[0];
    int count = 1;
    for (int i = 1; i < k; i++) {
        if (k_nearest_neighbors[i] == k_nearest_neighbors[i - 1])
        {
            count++;
        }
        else
        {
            if (count > max_count)
            {
                max_count = count;
                mode = k_nearest_neighbors[i - 1];
            }
            count = 1;
        }
    }
    if (count > max_count)
    {
        mode = k_nearest_neighbors[k - 1];
    }
    return mode;
}

int main() {
    int MAX_DIMENSIONS = 2048;
    int MAX_N = 1000000;
    //    int MAX_N = 100;
    int MAX_K = 50;

    int default_d = 64;
    int default_n = MAX_N;
    int default_k = MAX_K;

    int DIMENSIONS, n, k;
    srand(time(0));

    // N
    cout << "----------------Results under different N----------------" << endl;
    ofstream outfile_n("output_n.csv");
    DIMENSIONS = default_d;
    k = default_k;

    for (int n = 100; n <= MAX_N; n = n * 10)
    {
        cout << "Results with N: " << n << endl;
        //初始化训练集
        vector<DataPoint> training_set(n);

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < DIMENSIONS; d++)
            {
                //随机生成特征
                training_set[i].features.push_back((float)rand() / RAND_MAX);
            }
            //分配标签
            training_set[i].label = rand() % 10;
        }
        //创建随机测试点
        DataPoint datapoint_test_point;
        //距离特征提取
        for (int d = 0; d < DIMENSIONS; d++)
        {
            datapoint_test_point.features.push_back((float)rand() / RAND_MAX);
        }
        vector<double> test_point;
        test_point = datapoint_test_point.features;


        clock_t start_knn, end_knn;
        start_knn = clock();
        //串行KNN
        int predicted_label = knn_predict(training_set, test_point, k);
        end_knn = clock();   //结束时间
        double used_time = double(end_knn - start_knn);
        cout << "Predicted label with kNN: " << predicted_label << endl;
        cout << "Algorithm execution time: " << used_time << "ms" << endl;  //输出时间（单位：ｓ）

        clock_t start_parallel, end_parallel;
        start_parallel = clock();
        //并行KNN
        int predicted_label_parallel = knn_predict_parallel(training_set, test_point, k);
        end_parallel = clock();
        double used_time_parallel = double(end_parallel - start_parallel);
        cout << "Predicted label with parallel_kNN: " << predicted_label_parallel << endl;
        cout << "Parallel algorithm execution time: " << used_time_parallel << "ms" << endl;  //输出时间（单位：ｓ）

        cout << endl;

        outfile_n << n << "," << used_time << "," << used_time_parallel << endl;
        vector<DataPoint>().swap(training_set);
    }
    outfile_n.close(); // 关闭输出流


    // DIMENSIONS
    cout << "----------------Results under different D----------------" << endl;
    ofstream outfile_d("output_d.csv");
    n = default_n;
    k = default_k;
    //遍历不同维度
    for (int d = 8; d <= MAX_DIMENSIONS; d = d * 2)
    {
        cout << "Results with D: " << d << endl;
        vector<DataPoint> training_set(n);

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < DIMENSIONS; d++)
            {
                //生成训练集
                training_set[i].features.push_back((float)rand() / RAND_MAX);
            }
            //分配标签
            training_set[i].label = rand() % 10;
        }
        //提取特征
        DataPoint datapoint_test_point;
        for (int d = 0; d < DIMENSIONS; d++)
        {
            datapoint_test_point.features.push_back((float)rand() / RAND_MAX);
        }
        vector<double> test_point;
        test_point = datapoint_test_point.features;


        clock_t start_knn, end_knn;
        start_knn = clock();
        //串行
        int predicted_label = knn_predict(training_set, test_point, k);
        end_knn = clock();   //结束时间
        double used_time = double(end_knn - start_knn);
        cout << "Predicted label with kNN: " << predicted_label << endl;
        cout << "Algorithm execution time: " << used_time << "ms" << endl;  //输出时间（单位：ｓ）

        clock_t start_parallel, end_parallel;
        start_parallel = clock();
        //并行
        int predicted_label_parallel = knn_predict_parallel(training_set, test_point, k);
        end_parallel = clock();
        double used_time_parallel = double(end_parallel - start_parallel);
        cout << "Predicted label with parallel_kNN: " << predicted_label_parallel << endl;
        cout << "Parallel algorithm execution time: " << used_time_parallel << "ms" << endl;  //输出时间（单位：ｓ）

        cout << endl;

        outfile_d << d << "," << used_time << "," << used_time_parallel << endl;
        vector<DataPoint>().swap(training_set);
    }
    outfile_d.close(); // 关闭输出流


    // 比较不同k值下k近邻算法及并行版本
    cout << "----------------Results under different K----------------" << endl;
    ofstream outfile_k("output_k.csv");
    n = default_n;
    DIMENSIONS = default_d;

    for (int k = 5; k <= MAX_K; k = k + 5)
    {
        cout << "Results with K: " << k << endl;
        vector<DataPoint> training_set(n);

        for (int i = 0; i < n; i++)
        {
            for (int d = 0; d < DIMENSIONS; d++)
            {
                training_set[i].features.push_back((float)rand() / RAND_MAX);
            }
            training_set[i].label = rand() % 10;
        }

        DataPoint datapoint_test_point;
        for (int d = 0; d < DIMENSIONS; d++)
        {
            datapoint_test_point.features.push_back((float)rand() / RAND_MAX);
        }
        vector<double> test_point;
        test_point = datapoint_test_point.features;


        clock_t start_knn, end_knn;
        start_knn = clock();
        int predicted_label = knn_predict(training_set, test_point, k);
        end_knn = clock();   //结束时间
        double used_time = double(end_knn - start_knn);
        cout << "Predicted label with kNN: " << predicted_label << endl;
        cout << "Algorithm execution time: " << used_time << "ms" << endl;  //输出时间（单位：ｓ）

        clock_t start_parallel, end_parallel;
        start_parallel = clock();
        int predicted_label_parallel = knn_predict_parallel(training_set, test_point, k);
        end_parallel = clock();
        double used_time_parallel = double(end_parallel - start_parallel);
        cout << "Predicted label with parallel_kNN: " << predicted_label_parallel << endl;
        cout << "Parallel algorithm execution time: " << used_time_parallel << "ms" << endl;  //输出时间（单位：ｓ）

        cout << endl;

        outfile_k << k << "," << used_time << "," << used_time_parallel << endl;
        vector<DataPoint>().swap(training_set);
    }
    outfile_k.close(); // 关闭输出流


    return 0;
}

