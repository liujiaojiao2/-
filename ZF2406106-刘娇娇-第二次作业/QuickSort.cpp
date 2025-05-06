//
// Created by liujiaojiao on 25-4-24.
//
#include <iostream>
#include <vector>
#include <thread>
#include <algorithm>
#include <chrono>
#include <pthread.h>

using namespace std;
using namespace std::chrono;

// 数组大小
const int ARRAY_SIZE = 20000000;

// 线程数
int NUM_THREADS = 4;

// 快速排序函数
void quickSort(vector<double>& arr, int low, int high) {
    if (low < high) {
        // 选取最后一个元素作为 pivot
        double pivot = arr[high];
        int i = (low - 1);

        for (int j = low; j <= high - 1; j++) {
            if (arr[j] < pivot) {
                i++;
                swap(arr[i], arr[j]);
            }
        }
        swap(arr[i + 1], arr[high]);
        int pi = i + 1;

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

// 并行快速排序线程函数
void* parallelQuickSortThread(void* arg) {
    vector<double>* arr = (vector<double>*)arg;
    quickSort(*arr, 0, arr->size() - 1);
    pthread_exit(NULL);
    return NULL;
}

// 并行快速排序函数
void parallelQuickSort(vector<double>& arr, int numThreads) {
    if (arr.empty()) return;

    vector<pthread_t> threads(numThreads);
    vector<vector<double>> subArrays(numThreads);
    int subSize = arr.size() / numThreads;
    int remainder = arr.size() % numThreads;

    for (int i = 0; i < numThreads; i++) {
        int start = i * subSize;
        int end = (i + 1) * subSize - 1;
        if (i == numThreads - 1) {
            end += remainder;
        }
        subArrays[i] = vector<double>(arr.begin() + start, arr.begin() + end + 1);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_create(&threads[i], NULL, parallelQuickSortThread, &subArrays[i]);
    }

    for (int i = 0; i < numThreads; i++) {
        pthread_join(threads[i], NULL);
    }

    // 合并排序后的子数组
    arr.clear();
    for (int i = 0; i < numThreads; i++) {
        arr.insert(arr.end(), subArrays[i].begin(), subArrays[i].end());
    }
}

int main() {
    // 生成随机数数组
    vector<double> arr(ARRAY_SIZE);
    for (int i = 0; i < ARRAY_SIZE; i++) {
        arr[i] = (double)rand() / RAND_MAX * 1000; // 生成 0 到 1000 之间的随机数
    }

    cout << "数组大小: " << ARRAY_SIZE << endl;

    for (int numThreads : {1, 2, 4, 8}) {
        NUM_THREADS = numThreads;
        vector<double> arr_copy = arr; // 复制数组，保证每次排序的数据一致

        auto start = high_resolution_clock::now();
        parallelQuickSort(arr_copy, NUM_THREADS);
        auto end = high_resolution_clock::now();
        auto duration = duration_cast<milliseconds>(end - start);

        cout << NUM_THREADS << " 线程排序耗时: " << duration.count() << " 毫秒" << endl;

        // 可以选择性地验证排序结果
        // bool isSorted = is_sorted(arr_copy.begin(), arr_copy.end());
        // cout << "数组是否已排序: " << (isSorted ? "是" : "否") << endl;
    }

    return 0;
}