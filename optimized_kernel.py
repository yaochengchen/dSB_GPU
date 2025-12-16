__global__ void fused_dynamics_optimized_kernel(
    float* y,                   // 输入输出: y 数组 (N, batch_size)
    float* x,                   // 输入输出: x 数组 (N, batch_size)
    const float delta,          // 标量参数
    const float* p_array,       // p 数组 (n_iterations,)
    const float xi,             // 标量参数
    const float dt,             // 时间步长
    const int* J_indices,       // CSR格式稀疏矩阵的列索引
    const float* J_values,      // CSR格式稀疏矩阵的值
    const int* J_crow_ptr,      // CSR格式稀疏矩阵的行指针
    const int N,                // 矩阵维度
    const int batch_size,       // batch大小
    const int n_iterations      // 迭代次数
) {
    // ⭐ 新的线程布局：每个block处理一个batch
    int batch = blockIdx.y;     // 每个block对应一个batch
    int row = threadIdx.x + threadIdx.y * blockDim.x;  // block内的线程处理不同的rows
    
    if (batch >= batch_size || row >= N) return;
    
    // 计算在flatten数组中的索引
    int idx = row * batch_size + batch;
    
    // 读取初始值
    float x_val = x[idx];
    float y_val = y[idx];
    
    // 在kernel内部执行所有时间步
    for (int iter = 0; iter < n_iterations; iter++) {
        float p_i = p_array[iter];  // 获取当前时间步的p值
        
        // Step 1: 计算稀疏矩阵乘法 J * sign(x) 的第row行
        float sparse_result = 0.0f;
        int start = J_crow_ptr[row];
        int end = J_crow_ptr[row + 1];
        
        for (int j = start; j < end; j++) {
            int col_idx = J_indices[j];
            // ⭐ 关键：只读取同一个batch内其他row的数据
            float col_x = x[col_idx * batch_size + batch];
            
            // 计算 sign(col_x)
            float col_sign = (col_x > 0.0f) ? 1.0f : ((col_x < 0.0f) ? -1.0f : 0.0f);
            sparse_result += J_values[j] * col_sign;
        }
        
        // Step 2: 更新 y
        float term1 = -(delta - p_i) * x_val;
        float term2 = xi * sparse_result;
        y_val += (term1 + term2) * dt;
        
        // Step 3: 更新 x
        x_val += dt * y_val * delta;
        
        // Step 4: 条件截断
        float abs_x = fabsf(x_val);
        if (abs_x > 1.0f) {
            x_val = (x_val > 0.0f) ? 1.0f : -1.0f;
            y_val = 0.0f;
        }
        
        // ⭐ 将更新后的值写回全局内存
        x[idx] = x_val;
        y[idx] = y_val;
        
        // ⭐ 关键改进：只需要block内同步，因为同一个batch的所有线程都在同一个block内！
        __syncthreads();  // 不再需要昂贵的 grid.sync()！
    }
}

// 对应的launch配置
void launch_optimized_kernel(/* parameters */) {
    // ⭐ 新的grid/block配置
    if (N <= 1024) {
        // 情况1：N较小，可以用一个block处理一个batch的所有rows
        dim3 blocks(1, batch_size);        // 每个batch一个block
        dim3 threads(N, 1);                // 每个线程处理一个row
    } else {
        // 情况2：N较大，需要多个block处理一个batch
        int threads_per_block = 256;       // 或者512, 1024
        int blocks_per_batch = (N + threads_per_block - 1) / threads_per_block;
        
        dim3 blocks(blocks_per_batch, batch_size);
        dim3 threads(threads_per_block, 1);
        
        // 注意：这种情况下可能需要额外处理跨block的同步
        // 或者重新设计算法来避免这种情况
    }
    
    fused_dynamics_optimized_kernel<<<blocks, threads>>>(
        y, x, delta, p_array, xi, dt,
        J_indices, J_values, J_crow_ptr,
        N, batch_size, n_iterations
    );
}
