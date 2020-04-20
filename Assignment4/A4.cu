#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/sequence.h>
#include <thrust/transform.h>
#include <thrust/functional.h>

#include <iostream>
#include <thrust/count.h>
#include <vector>
#include <utility>
using namespace std;

struct my_functor
{
    const int a;
    my_functor(int _a) : a(_a) {}
    __host__ __device__
        int operator()(const int& x) const { 
            return max(x-a, 0);
        }
};

int schedule(int N, int M, int* arrival_times, int* burst_times, int** cores_schedules, int* cs_lengths)
{
    thrust::device_vector<int> remaining_time(M, 0);
    int arrival, prev_arrival=1, turnaround=0;
    vector<int> cores[M];
    for(int i=0; i<N; ++i)
    {
        arrival = arrival_times[i];
        thrust::transform(remaining_time.begin(), remaining_time.end(), remaining_time.begin(),my_functor(arrival-prev_arrival));
        int idx = thrust::min_element(remaining_time.begin(), remaining_time.end())-remaining_time.begin();
        thrust::for_each(remaining_time.begin()+idx, remaining_time.begin()+idx+1, thrust::placeholders::_1 += burst_times[i]);
        turnaround += remaining_time[idx];
        cores[idx].push_back(i);
        // cs_lengths[idx]++;
        prev_arrival = arrival;
    }

    // Populate output vectors
    for(int i=0; i<M; ++i)
    {
        cs_lengths[i] = cores[i].size();
        cores_schedules[i]= (int*) malloc(cs_lengths[i] * sizeof(int));
        for(int j = 0; j< cs_lengths[i]; ++j)
        {
            cores_schedules[i][j] = cores[i][j];
        }
    }
    return turnaround;
}
