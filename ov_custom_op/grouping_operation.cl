__kernel void groupingOperationCLKernel(
    __global const float* features,
    __global const int* idx,
    __global float* output,
    const int b, // batch size
    const int c, // number of channels
    const int n, // number of points in features
    const int npoint, // number of points in idx
    const int nsample // number of samples in idx
) {
    // 计算全局ID
    int gid = get_global_id(0);
    
    if (gid >= b * c * npoint * nsample) return;

    // 计算索引
    int sample_index = gid % nsample;
    int point_index = (gid / nsample) % npoint;
    int channel_index = (gid / (npoint * nsample)) % c;
    int batch_index = gid / (c * npoint * nsample);

    // 获取当前batch、channel、point和sample对应的特征值和索引
    int idx_offset = batch_index * npoint * nsample + point_index * nsample + sample_index;
    int feature_offset = batch_index * c * n + channel_index * n;
    int out_offset = batch_index * c * npoint * nsample + channel_index * npoint * nsample + point_index * nsample + sample_index;

    int ii = idx[idx_offset];
    if(ii >= 0 && ii < n) {
        output[out_offset] = features[feature_offset + ii];
    } else {
        output[out_offset] = 0.0f;
    }
}

void groupingOperationWrapper(
    __global const float* features,
    __global const int* idx,
    __global float* output){
    int b = 32;
    int c = 3;
    int n = 1024;
    int npoint = 128;
    int nsample = 64;

    groupingOperationCLKernel(
        features,
        idx,
        output,
        b,
        c,
        n,
        npoint,
        nsample
    );

}