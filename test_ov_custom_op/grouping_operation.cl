__kernel void groupingOperationCLKernel(__global const float* features, __global const int* idx, __global float* output
                                        // const int b, const int c, const int n, const int npoints, const int nsample
                                        ) {
    int batch_index = get_group_id(0); // 当前批次索引
    int local_id = get_local_id(0);   // 当前工作项在局部内存中的索引
    int global_id = get_global_id(0); // 当前工作项的全局索引

    // 每个工作组处理一个通道的所有点
    int channel = get_group_id(1); // 当前通道索引

    printf("===INPUT0_DIMS : %d===\n", INPUT0_DIMS);
    printf("===INPUT1_DIMS : %d===\n", INPUT1_DIMS);
    // printf("===attributes===\n",b,c,n,npoints,nsample);

    __local float local_features[256]; // 局部内存，大小根据硬件调整
    __local int local_idx[256];        // 局部内存，大小根据硬件调整

    // 每个工作项加载一个点的特征
    if (global_id < npoints * nsample) {
        int point_id = global_id / nsample; // 当前点的索引
        int sample_id = global_id % nsample; // 当前采样的索引

        // 加载索引到局部内存
        local_idx[local_id] = idx[point_id * nsample + sample_id];

        // 加载特征到局部内存
        int feature_idx = channel * n + local_idx[local_id];
        local_features[local_id] = features[feature_idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE); // 确保所有数据加载完成

    // 将局部内存中的数据写入输出
    if (global_id < npoints * nsample) {
        int point_id = global_id / nsample;
        int sample_id = global_id % nsample;

        // 计算输出的索引
        int output_idx = (channel * npoints + point_id) * nsample + sample_id;
        output[output_idx] = local_features[local_id];
    }
}