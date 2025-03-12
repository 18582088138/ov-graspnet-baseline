__kernel void cylinderQueryCLKernel(
    __global const float* new_xyz,
    __global const float* xyz,
    __global const float* rot,
    const float radius2,
    const float hmin,
    const float hmax,
    const int nsample,
    const int b, // batch size
    const int n, // number of points in xyz
    const int m, // number of new_xyz points
    __global int* idx) {

    // 获取全局ID
    int gid = get_global_id(0);
    // 计算batch index, 新点index
    int j = gid / m;
    int point_index = gid % m;

    if (j < b && point_index < m) {
        // 计算当前批次中xyz, new_xyz, rot 和 idx 的起始位置
        int batch_offset_xyz = j * n * 3;
        int batch_offset_new_xyz = j * m * 3;
        int batch_offset_rot = j * m * 9;
        int batch_offset_idx = j * m * nsample;

        // 获取当前点坐标和旋转矩阵
        float new_x = new_xyz[batch_offset_new_xyz + point_index * 3 + 0];
        float new_y = new_xyz[batch_offset_new_xyz + point_index * 3 + 1];
        float new_z = new_xyz[batch_offset_new_xyz + point_index * 3 + 2];

        float r[9] = {
            rot[batch_offset_rot + point_index * 9 + 0], rot[batch_offset_rot + point_index * 9 + 1], rot[batch_offset_rot + point_index * 9 + 2],
            rot[batch_offset_rot + point_index * 9 + 3], rot[batch_offset_rot + point_index * 9 + 4], rot[batch_offset_rot + point_index * 9 + 5],
            rot[batch_offset_rot + point_index * 9 + 6], rot[batch_offset_rot + point_index * 9 + 7], rot[batch_offset_rot + point_index * 9 + 8]
        };

        int cnt = 0;
        for (int k = 0; k < n && cnt < nsample; ++k) {
            // 计算点相对于新点的位置，并应用旋转
            float x = xyz[batch_offset_xyz + k * 3 + 0] - new_x;
            float y = xyz[batch_offset_xyz + k * 3 + 1] - new_y;
            float z = xyz[batch_offset_xyz + k * 3 + 2] - new_z;
            float x_rot = r[0] * x + r[3] * y + r[6] * z;
            float y_rot = r[1] * x + r[4] * y + r[7] * z;
            float z_rot = r[2] * x + r[5] * y + r[8] * z;

            // 判断是否在圆柱体内
            float d2 = y_rot * y_rot + z_rot * z_rot;
            if (d2 < radius2 && x_rot > hmin && x_rot < hmax) {
                if (cnt == 0) {
                    for (int l = 0; l < nsample; ++l) {
                        idx[batch_offset_idx + point_index * nsample + l] = k;
                    }
                }
                idx[batch_offset_idx + point_index * nsample + cnt] = k;
                ++cnt;
            }
        }
    }
}
