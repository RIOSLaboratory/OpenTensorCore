#  TensorCore1设计规范（Design Spe

- [修改注意事项](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-修改注意事项)

- [0 设计需求与背景知识](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0设计需求与背景知识) 

- - [0.1 设计需求](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.1设计需求) 

  - - [0.1.1 核心特性](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.1.1核心特性)
    - [0.1.2 精度模式](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.1.2精度模式)
    - [0.1.3 矩阵运算](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.1.3矩阵运算)
    - [0.1.4 性能指标](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.1.4性能指标)

  - [0.2 背景知识](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2背景知识) 

  - - [0.2.1 矩阵乘加代码示例](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2.1矩阵乘加代码)
    - [0.2.2 行优先与列优先](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2.2行优先与列优)
    - [0.2.3 CUTLASS库行优先](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2.3CUTLAS)
    - [0.2.4 AXI-Stream协议](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2.4AXI-St)
    - [0.2.5 GEMM](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2.5GEMM)
    - [0.2.6 数据格式](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-0.2.6数据格式)

- [1 全局参数及模块列表](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-1全局参数及模块列表) 

- - [1.1 全局参数](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-1.1全局参数)
  - [1.2 模块列表](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-1.2模块列表)

- [2 顶层](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-2顶层) 

- - [2.1 顶层架构](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-2.1顶层架构)
  - [2.2 tensor_core顶层模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-2.2tensor_c)

- [2.3 存储架构](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-2.3存储架构)

- [3 格式转换模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-3格式转换模块) 

- - [3.1 to_fp9_con模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-3.1to_fp9_c)
  - [3.2 to_next_con模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-3.2to_next_)

- [4 张量计算模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-4张量计算模块) 

- - [4.1 mm_mul_add模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-4.1mm_mul_a)
  - [4.3 mv_mul模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-4.3mv_mul模块)
  - [4.4 tc_mul模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-4.4tc_mul模块)
  - [4.5 tc_add模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-4.5tc_add模块)

- [5 数据通道模块 ](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-5数据通道模块) 

- - [5.1 tc_mul模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-5.1tc_mul模块)
  - [5.2 tc_add模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-5.2tc_add模块)
  - [5.3 tc_mm_add模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-5.3tc_mm_ad)

- [6 基本计算模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6基本计算模块) 

- - [6.1 fmul_s1模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.1fmul_s1模)
  - [6.2 naivemultiplier模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.2naivemul)
  - [6.3 fmul_s2模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.3fmul_s2模)
  - [6.4 fmul_s3模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.4fmul_s3模)
  - [6.5 fadd_s1模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.5fadd_s1模)
  - [6.6 fadd_s2模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.6fadd_s2模)
  - [6.7 far_path模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.7far_path)
  - [6.8 near_path模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-6.8near_pat)

- [7 舍入模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-7舍入模块)

- [8 辅助计算模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-8辅助计算模块) 

- - [8.1 shift_right_jam右移模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-8.1shift_ri)
  - [8.2 lza前导0预测模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-8.2lza前导0预测)
  - [8.3 lzc前导0计数模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-8.3lzc前导0计数)

- [9 存储模块SRAM](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-9存储模块SRAM) 

- - [9.1 amem模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-9.1amem模块)
  - [9.2 bmem模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-9.2bmem模块)
  - [9.3 cmem模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-9.3cmem模块)
  - [9.4 md_data模块](https://tensor03.cn6.quickconnect.cn/oo/r/15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP#heading_id=TensorCore1设计规范（DesignSpec）-9.4md_data模)

# 修改注意事项

1.  有些模块用了axistream接口，有些没用。这个是有问题的，至少要保证在两级FIFO或SRAM之间的模块接口保持一致。因为stream和非stream接口之间的转换必须借助FIFO才能实现。
   \2. tc_mul是A的多个列（行）同时对应B的一行（列），这里就存在A需要多次读取的问题，那么多次读取前面的SRAM就不能是FIFO，因为它需要一次写入，多次读取。这里需要有一个专用的类似多维DMA的电路来实现。接口可以设计成流式，但是这一级的SRAM不是FIFO。
   \3. 文档里的6的基本计算模块都是组合逻辑么？

   \4. 2.2 主要更改点 
      in_valid由a/b/c三个valid和三个last相与，out_ready由三个ready相与（按照AXI Stream协议处理）

     在顶层不能保证a，b，c数据同时到达，现实中是无法做到在顶层模块做到的。在tensorcore外没法同步a,b,c他们应该是串行的。这个同步应该在内部的第一级RAM/FIFO输出做。

 

 

本文按照tensorcore自顶向下的次序，对设计要求进行阐述

# 0 设计需求与背景知识

## 0.1 设计需求

TensorCore是GPU中的专用硬件单元，主要用于加速矩阵运算，特别是在深度学习和高性能计算（HPC）中。TensorCore支持多种数据格式，能够高效地执行混合精度计算。



设计需求如下：

1. 1）兼容现有gpgpu-sim里的函数接口模式，方便编译器开发者与开源社区为该TensorCore适配CUDA编译器。
2. 2）在兼容gpgpu-sim的前提下支持FP8和FP4，方便性能提升。
3. 3）未来能整合入vortex。

 

1. 合作方需求：MAC或TensorCore本体，标准化，方便嵌入各种设计。

说明：不包括TensorCore外的处理器核。

Tensor Core 是一种专为加速矩阵乘法和累加操作设计的硬件单元，其核心目标是通过并行化和流水线技术加速矩阵乘法和累加操作（GEMM)，广泛应用于人工智能（AI）与高性能计算（HPC)任务。

本规格说明书描述了一种支持 **FP16/BF16（BF16后面支持）,FP8(E5M2和E4M3),FP4** 输入，FP16/**FP32**累加的 Tensor Core 设计。

### **0.1.1 核心特性**

- **混合精度支持：**

- - 输入数据格式：FP16/BF16（**BF16后面支持**）、FP8(E5M2和E4M3)、FP4。（均转换为FP9再进入TensorCore）
  - 内部累加精度：FP16、**P22（原为FP32）**。**F**（建议只选一个作为内部累加精度，方便设计）

- - 输出数据格式：FP8/FP16/**FP32**

- **高吞吐量**：

- - 针对密集矩阵运算（如 GEMM：通用矩阵乘法）进行优化。

- - 支持数据搬运与计算并行，性能x2，寄存器开销x2（选配）（如果不是gpgpu-sim已经支持的，建议去掉，否则编译器工作量太大）

- **灵活的数据类型**：

- - 支持多种输入和输出数据类型。

- **内存高效性**：

- - 利用共享内存和寄存器实现低延迟数据访问。

### **0.1.2** **精度模式**

Tensor Core 支持以下精度模式：

|                                   |      |                   |                |
| --------------------------------- | ---- | ----------------- | -------------- |
| **FP16**/BF16（**BF16后面支持**） | FP32 | **FP32**          | 深度学习训练   |
| **FP8**                           | FP32 | FP8/FP16/**FP32** | 深度学习推理   |
| **FP4**                           | FP32 | FP8/FP16/**FP32** | 低比特神经网络 |

 

| **输入精度**                  | **累加精度** | **输出精度**      | **主要应用场景** |
| ----------------------------- | ------------ | ----------------- | ---------------- |
| **FP16**/BF16（BF16后面支持） | FP22         | **FP32**          | 深度学习训练     |
| **FP8**                       | FP22         | FP8/FP16/**FP32** | 深度学习推理     |
| **FP4**                       | FP22         | FP8/FP16/**FP32** | 低比特神经网络   |

  **优先支持粗体标注精度**

**注意：输入数据均转为FP8再进入TensorCore参与计算。**

### **0.1.3** **矩阵运算**

Tensor Core 执行矩阵乘法和累加操作，形式如下：

　Ｄ=A·B+C （矩阵内积再加）

其中：

- A 和 B 是输入矩阵。
- C 是累加矩阵。

矩阵乘法的定义要求输出矩阵的行数继承自 A 的行数，列数继承自 B 的列数。其中A为M∗K维矩阵， B为K∗N维矩阵， C为M∗N维矩阵。则A×B结果矩阵的维度为M*N。GPU的每个线程块通常支持最多1024个线程。

 

**支持的输入矩阵形状**

**硬件级支持：16x16x16**

1. **CUDA编译级支持：**

- FP16/BF16: **m16n16k16**/m32n8k16/m8n32k16

- FP8      : **m16n16k16**/m32n8k16/m8n32k16
- FP4      : **m16n16k16**/m32n8k16/m8n32k16

  **优先支持粗体标注形状**

1. Dimensions m,n,k: m x k matrix_a; k x n matrix_b; m x naccumulator

**数据流**

Tensor Core 内部有高达4Kbyte的寄存器用于缓存输入矩阵，A,B占512byte, C,D占1Kbyte。每个块(tile)分为32个threads，每个thread包含4个32bit的寄存器，以兼容CUDA wmma API。（大小待确认）

1. **输入矩阵**：

2. - 从共享内存或者全局内存加载，并做0延迟的矩阵转置运算后存入寄存器Tile，（共享内存和全局内存最好写为sharememory与global memory，避免误解）
   - 在寄存器中暂存以实现低延迟访问。A,C矩阵在寄存器中为行优先排序，B矩阵为列优先排序。

3. **计算**：

4. - 矩阵乘法和累加操作在多个线程中并行执行

5. **输出矩阵**：

6. - 结果存储在输出寄存器Tile中，D矩阵在寄存器中为行优先排序。
   - 将寄存器中的输出结果做0延迟转置后，写回共享内存或全局内存。

### **0.1.4** **性能指标**

- **计算单元：**

- - 一个TensorCore包含32个并行计算线程(兼容CUDA)（数量待确认）

- - 每个线程包含1-8个(可配置）Mac运算模块
  - 每个Mac运算模块每个周期处理1个32位寄存器中的A，B输入累加。

- **吞吐量**：

- - FP16/BF16：128/1028   FOPS/cycle（数量待确认）
  - FP8： 256/2056   FOPS/cycle（数量待确认）
  - FP4：512/4096  FOPS/cycle（数量待确认）

- **延迟**： 

- - 10-40个cycle(受数据总线宽度与Mac配置数量，数据类型以及矩阵形状影响）

- **能效**：

- 设计目标为高能效显著大于10TOPS/W

## 0.2 背景知识

### **0.2.1 矩阵乘加代码示例**

 

// GPU计算：矩阵乘CUDA核函数

__global__ void matmulKernel(const float *A, const float *B, float *C,

​               float alpha, float beta, int M, int N, int K) {

  // 获取thread在三级视图grid -> block -> thread中的索引

  // thread的索引为(m,n)，则负责计算元素C[m][n]

  int m = threadIdx.x + blockDim.x * blockIdx.x;

  int n = threadIdx.y + blockDim.y * blockIdx.y;

  if (m >= M || n >= N)

​    return;

  float acc = 0.0f;

  for (int k = 0; k < K; ++k) {

​    acc += A[m * K + k] * B[k * N + n];

  }

  C[m * N + n] = acc * alpha + C[m * N + n] * beta;

}

// GPU计算：矩阵乘CUDA实现

void cudaMatMulSample(const float *A, const float *B, float *C, float alpha,

​           float beta, int M, int N, int K) {

  dim3 block(16, 16);

  dim3 grid((M + block.x - 1) / block.x, (N + block.y - 1) / block.y);

  matmulKernel<<<grid, block>>>(A, B, C, alpha, beta, M, N, K);

}

 

### **0.2.2 行优先与列优先**

矩阵存储方式有两种，一种是“行主序（row-major order）/行优先”，另一种就是“列主序（column-major order）/列优先”/

1. Direct3D 采用行主序(Row major)存储
   “Effect matrix parameters and HLSL matrix variables can define whether the value is a row-major or column-major matrix; however, the DirectX APIs always treat D3DMATRIX and D3DXMATRIX as row-major.”OpenGL 采用列主序(Colume major)存储
   “The m parameter points to a 4x4 matrix of single- or double-precisionfloating-point values stored in column-major order. That is, the matrix isstored as follows”

1. 存储顺序说明了线性代数中的矩阵如何在线性的内存数组中存储，d3d 将每一行在数组中按行存储，而opengl将每一列存储到数组的每一行中：
2. 　因此，对于线程代数中的同一个矩阵，则在d3d和OpenGL中有不同的表示形式：

 

线代矩阵：a11,a12,a13,a14      d3d保存: a11,a12,a13,a14      OpenGL保存: a11,a21,a31,a41

​            a21,a22,a23,a24             a21,a22,a23,a24            a12,a22,a32,a42

​            a31,a32,a33,a34             a31,a32,a33,a34            a13,a23,a33,a43

​            a41,a42,a43,a44             a41,a42,a43,a44            a14,a24,a34,a44

 

主机侧的矩阵乘法为行优先存储，CUBLAS默认矩阵数据存储为列优先存储。

1. 解决这个问题，C=A×B=(B^T×A^T)^T，也就是将矩阵A,B先按照列优先读取（转置：row major的A矩阵shape为[M,K],A.T的shape显然是[K,M]），调换A与B的前后顺序，得到矩阵C的转置C^T=B^T×A^T，此时为列优先存储，要再次转置，即按行优先读取，得到目标矩阵C。

### **0.2.3 CUTLASS库行优先**

1. CUTLASS为高性能线性代数提供了CUDA C++开源模版库。

矩阵乘：

- 运行效率：CUTLASS≈CUBLAS>CUDA编程自行实现
- 灵活性：CUTLASS>CUBLAS
- CUTLASS要求C++17主机编译器,并在使用CUDA     12.4工具包构建时性能最佳。

1. 这里cutlass::layout::RowMajor与cutlass::layout::ColumnMajor分别是矩阵内存布局是行主序和列主序。
   参数OperatorClass_表示默认由cuda core计算，也就是arch::OpClassSimt，如果是让tensor core计算，那么就要用arch::OpClassTensorOp
   cuda处理问题的时候，一般是先将问题划分成多个小问题，然后每个小问题规模由一个Threadblock处理，ThreadblockShape_含义就是Threadblock所要处理的问题规模。
   同理，Threadblock所处理的问题规模，由好多个Warp一起完成的，WarpShape_就是每个Warp所要处理的问题规模

 

// GPU计算：矩阵乘CUTLASS实现

void cutlassMatMulSample(const float *A, const float *B, float *C, float alpha,

​             float beta, int M, int N, int K) {

  using Gemm = cutlass::gemm::device::Gemm<

​    float, cutlass::layout::RowMajor, float, cutlass::layout::RowMajor,

​    float, cutlass::layout::RowMajor, float, cutlass::arch::OpClassSimt,

​    cutlass::arch::Sm80>;

  Gemm::Arguments args({M, N, K}, {A, K}, {B, N}, {C, N}, {C, N},

​             {alpha, beta});

  Gemm gemm_op;

  gemm_op.initialize(args);

  gemm_op();

}

 

### 0.2.4AXI-Stream协议

参考文章 [AXIStream协议](http:///pages/viewpage.action?pageId=90014370)

### 0.2.5 GEMM

 Tensor Cores专为加速通用矩阵乘法（GEMM）操作设计。每个时钟周期可以执行64次FP16 FMA（FusedMultiply-Add）操作，显著提高了运算速度。   

 具体实现**：**在Ampere架构中，Tensor Cores能够处理大小为M×K与K×N的矩阵相乘，通过优化的硬件结构实现高效并行计算。

 应用场景：适用于卷积神经网络（CNN）、循环神经网络（RNN）等深度学习模型中的关键运算，如图像识别和自然语言处理（NLP）。例如，在ResNet或VGG模型中，TensorCores可以加速特征提取层中的矩阵乘法，从而加快整个模型的训练过程。



### 0.2.6 数据格式

**输入格式**

- FP16（半精度浮点数）：16位浮点数，1位符号，5位指数，10位尾数。
- FP8：8位浮点数，1位符号，E5M2有5位指数和2位尾数，E4M3有4位指数和3位尾数。
- FP4：4位浮点数，1位符号，2位指数，1位尾数。

 **累加格式**

- FP16（半精度浮点数）：16位浮点数，1位符号，5位指数，10位尾数。
- FP32（单精度浮点数）：32位浮点数，1位符号，8位指数，23位尾数。

![img](https://raw.githubusercontent.com/chenweiphd/typopic/master/img)

**使用场景**

- FP32（单精度浮点数）：适合深度学习训练，提供最高的精度，用于累加和存储中间结果。
- FP16（半精度浮点数）：适合深度学习训练和推理，提供较高的精度和性能平衡。
- FP8：适合深度学习推理，提供较低的精度但较高的动态范围和性能。
- FP4：适合深度学习推理，提供极低的精度但极高的性能，适合模型压缩和加速。

参考文档

https://images.nvidia.cn/aem-dam/en-zz/Solutions/data-center/nvidia-ampere-architecture-whitepaper.pdf

https://www.advancedclustering.com/wp-content/uploads/2022/03/gtc22-whitepaper-hopper.pdf (FP8)

# 1 全局参数及模块列表

## 1.1 全局参数

| **参数名**          | **缺省值**                  | **描述**                                      | **备注**                                      |      |
| ------------------- | --------------------------- | --------------------------------------------- | --------------------------------------------- | ---- |
| SHAPE_K             |                             | tc_mul需要的乘法器个数，即A的列数（=B的行数） |                                               |      |
| SHAPE_M             |                             | 每个mv_mul中的tc_mul个数（=A的行数）          |                                               |      |
| SHAPE_N             |                             | mv_mul的个数（=B的列数）                      |                                               |      |
| EXPWIDTH            |                             | 指数位宽                                      |                                               |      |
| PRECISION           |                             | 精度，通常也叫尾数位宽，包括隐含位            |                                               |      |
| OUTPC               |                             | 输出精度，用于fadd_s1模块                     |                                               |      |
| LATENCY             |                             | 流水线延迟                                    |                                               |      |
|                     |                             |                                               |                                               |      |
| MATRIX_BUS_WIDTH    | 512（注意这里与前一版不同） | A, B矩阵的总线位宽                            | must be 2^n                                   |      |
| MAC_PER_THREAD      | 2                           | 每个THREAD的MAC数量                           |                                               |      |
|                     |                             |                                               |                                               |      |
| PARALLEL_DATA_MOVER | 0                           | 0: 不支持并行数据搬运1: 支持并行数据搬运      | 暂不支持并行数据搬运                          |      |
| TRANSPOSE           | 0                           | 0: 不支持矩阵转置1: 支持矩阵转置              | 当设为1时transpose_en端口无效暂不支持矩阵转置 |      |
| **参数名称**        | **缺省值**                  | **描述**                                      | **备注**                                      |      |
|                     |                             |                                               |                                               |      |
| MATRIX_BUS_WIDTH    | 1024                        | A, B矩阵的总线位宽                            | must be 2^n                                   |      |
| MAC_PER_THREAD      | 2                           | 每个THREAD的MAC数量                           |                                               |      |
|                     |                             |                                               |                                               |      |
| PARALLEL_DATA_MOVER | 0                           | 0: 不支持并行数据搬运1: 支持并行数据搬运      | 暂不支持并行数据搬运                          |      |
| TRANSPOSE           | 0                           | 0: 不支持矩阵转置1: 支持矩阵转置              | 当设为1时transpose_en端口无效暂不支持矩阵转置 |      |
|                     |                             |                                               |                                               |      |

## 1.2 模块列表

| **分类**         | **模块**                                            | **描述**                                                     |
| ---------------- | --------------------------------------------------- | ------------------------------------------------------------ |
| **顶层**         | tensor_core.v                                       | 张量核心顶层设计模块                                         |
|                  | tensor_core_exev.v                                  | 张量核心顶层设计模块(最顶层)，多线程调度和结果汇总           |
| **格式转换**     | to_fp8_con.v                                        | FP8格式转换器（顶层接口）                                    |
|                  | to_fp8_core.v                                       | FP8格式转换核心逻辑模块(最后是转化为FP9)                     |
|                  | fp22_to_fp8_con.v                                   | FP22格式到FP8格式的专用转换器                                |
| **张量计算**     | tc_dot_product.v                                    | 点积计算单元（核心计算模块）                                 |
|                  | tc_mul_pipe.v                                       | 乘法流水线模块，3级流水线，集成fmul_s1/2/3                   |
|                  | tc_add_pipe.v                                       | 加法流水线模块，2级流水线                                    |
|                  | naivemultiplier.v                                   | 基础尾数乘法器（组合逻辑）                                   |
| **浮点运算单元** | fadd_s1.v                                           | 浮点加法器第一阶段，接收tc_mul的累加中间值，操作数对齐和特殊值处理 |
|                  | fadd_s2.v                                           | 浮点加法器第二阶段，尾数相加和结果规格化                     |
|                  | fmul_s1.v                                           | 浮点乘法器第一阶段，接收来自tc_mul_pipe的输入，指数处理和尾数预处理 |
|                  | fmul_s2.v                                           | 浮点乘法器第二阶段，调用naivemultiplier.v生成部分积，进行压缩处理 |
|                  | fmul_s3.v                                           | 浮点乘法器第三阶段，最终结果规格化和舍入，输出到tc_mul_pipe的结果总线 |
| **辅助计算**     | [cf_math_pkg.sv](http://cf_math_pkg.sv)             | 数学函数和常数定义包                                         |
|                  | [norm_div_sqrt_mvp.sv](http://norm_div_sqrt_mvp.sv) | 支持乘法的归一化/除法/平方根计算模块                         |
| **存储单元**     | singleport_SRAM.v                                   | 单端口SRAM存储器模型                                         |
| **配置与控制**   | config_registers.v                                  | 配置寄存器模块                                               |
| **项目支撑**     | [define.sv](http://define.sv)                       | 全局宏定义文件                                               |
|                  | filelist.f                                          | 项目文件列表（用于编译/仿真）                                |
|                  | run.tcl                                             | 自动化运行脚本                                               |
| **日志文件**     | command.log                                         | 命令执行历史日志                                             |
|                  | scl.log                                             | 脚本运行日志                                                 |
| **其他文件**     | default.svf                                         | 默认配置文件                                                 |
|                  | qrd_mvp.pvk                                         | 未知                                                         |

# 2 顶层

## 2.1 顶层架构

位于网盘/MB1_RDPrjIntern/prj25_TensorCore1/框图/TensoreCore架构图.oslides

 

![img](https://tensor03.cn6.quickconnect.cn/direct/oo/file/1030_KH2F6SQNFD3A78OS4LJRUBF8LO.doc/EGDAO3K67H2J7FNPE2FS94S1A4/img?tid="tto6RZUAk5iD12wYIH4XHLY1mYCynzQ5J7aQtghIrYy1wIoEM6EdEd8mwtwPf6MqEU_9h8dZIbGmGu23"&linkId="15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP")

 

1. m：矩阵，r_v：行向量，c_v：列向量，ABC矩阵维数均为8*8，（默认）精度相同，8个mv_mul模块，每个mv_mul模块中有8个tc_mul模块

1. 总线与tensor_core模块之间：     

2. 1. 来自总线上的数据以矩阵形式送入tensor_core模块，其中A/C元素均为行主存，B元素为行写入列读取（初步考虑列主存，交叉字线结构sram），位宽都是512

3. 格式转换器和sram：     

4. 1. to_fp9_con将所有精度的数据转换为fp9（e5m3），to_next_con根据tc_mm_add的输入精度选择不同的格式将输入转换为fp4-fp22
   2. 插入SRAM用于存储计算结果，fp9 sram的位宽为512/8*9，若总线数据精度为fp4，则需要并串转换存2次，若数据精度为fp16，则计算2个周期存1次

5. mm_mul_add模块和最后一级SRAM：

6. 1. 把gemm算法分解为gemv算法，让A矩阵分别乘B矩阵的每个列向量，结果列向量组以fp（9e5m3）输入到tc_mm_add模块构成中间矩阵，与来自SRAM（fp4-fp22）的矩阵元素做累加
   2. 输出位宽为512，根据sram内部数据存储格式调整输出次数
   3. 最后一级SRAM用于暂存计算结果，输入位宽为512，若数据精度为fp8，则直接占用512，若数据精度为fp4则存两批到512，若数据精度为fp16，则计算2个周期存1次，若数据精度为fp32，则计算4个周期存1次

7. mv_mul模块：     

8. 1. 对于一个mv_mul而言，输入数据为A矩阵和B矩阵的一个列向量
   2. 将A矩阵分解为若干个行向量，分发到8个tc_mul组成的阵列在其内部做向量的元素乘法，输出的结果为A的第n行向量与B的第n行标量相乘的行向量
   3. 上一步得到的行向量组送入tc_add加法树得到一个列向量

9. tc_mm_add模块：     

10. 1. 若干个列向量拼接组成中间矩阵和来自SRAM（fp4-fp22）的矩阵元素做累加

11. tensor_core模块输出：     

12. 1. 输出精度为fp4-fp32，和SRAM存储数据的精度保持一致

## 

- 

- **功能**：

- - 执行矩阵乘法和累加操作（D=A×B+C）。
  - 支持混合精度计算（如 FP16/BF16 /FP8输入、FP32 累加）。

- **设计细节**：

- - 由多个并行的Mac单元组成，每个单元负责一个小矩阵块（元素集）的计算。
  - 支持 Warp 级别的并行计算，每个Warp的线程配置1-8个Mac阵列。

- - 每个Mac运算单元每个周期处理1个32位存储中的A，B输入累加。

- 通过硬件优化减少乘加操作的延迟和功耗。

## 4.3 mv_mul模块

该模块实现了gemv的功能，大致流程如下：

1. 例化shape_m个tc_mul模块，分发A矩阵的每行数据到tc_mul中，用于A的一行和B的一列相乘

1. 例化tc_add模块，实现输入行向量组的对行求和，此时输出为shape_n*1的列向量

| **引脚名称**    | **方向** | **位宽**   | **说明**                                       |
| --------------- | -------- | ---------- | ---------------------------------------------- |
| 输入数据信号    |          |            |                                                |
| a_i             | input    | 512/8*9    | 数据信号，矩阵A元素输入                        |
| b_i             | input    | 8*9        | 数据信号，矩阵B的列向量元素输入                |
| 只读控制信号    |          |            |                                                |
| rm_i            | input    | 3          | 控制信号，舍入模式                             |
| ctrl_reg_idxw_i | input    | 8          | 控制信号，写回目的寄存器索引（包括扩展寄存器） |
| ctrl_wid_i      | input    | DEPTH_WARP | 控制信号，Warp ID                              |
| 握手信号        |          |            |                                                |
| in_valid_i      | input    | 1          | 握手信号，输入数据是否有效                     |
| out_ready_i     | input    | 1          | 握手信号，下游是否准备好接收结果               |
| in_ready_o      | output   | 1          | 状态信号，本模块是否准备好接收数据             |
| out_valid_o     | output   | 1          | 状态信号，输出结果是否有效                     |
| 输出数据信号    |          |            |                                                |
| result_o        | output   | 8*9        | 数据信号，A矩阵乘B矩阵列向量得到的列向量       |
| fflags_o        | output   | 5          | 异常标志寄存器输出（如 overflow）              |

## 4.4 tc_mul模块

该模块实现了A的每一行与B的每一列相乘，可以例化SHAPE_K=8个ventus的tc_mul_pipe模块，并对位宽适当调整来实现

| **引脚名称**    | **方向** | **位宽**   | **说明**                                     |
| --------------- | -------- | ---------- | -------------------------------------------- |
| 输入数据信号    |          |            |                                              |
| a\|b_i          | input    | 8*9        | 输入A/B矩阵元素数据                          |
| 只读控制信号    |          |            |                                              |
| rm_i            | input    | 3          | 舍入模式（IEEE 754）                         |
| ctrl_c_i        | input    | 4/8/16     | 控制信号，累加项 C                           |
| ctrl_rm_i       | input    | 3          | 控制信号，舍入模式                           |
| ctrl_reg_idxw_i | input    | 8          | 控制信号，目标寄存器索引                     |
| ctrl_warpid_i   | input    | DEPTH_WARP | 控制信号，Warp ID                            |
| 握手信号        |          |            |                                              |
| in_valid_i      | input    | 1          | 握手信号，输入数据是否有效                   |
| out_ready_i     | input    | 1          | 握手信号，下游模块是否准备好接收输出         |
| in_ready_o      | output   | 1          | 握手信号，本模块是否准备好接收数据           |
| out_valid_o     | output   | 1          | 握手信号，输出数据是否有效                   |
| 输出数据信号    |          |            |                                              |
| result_o        | output   | 8*9        | 矩阵A的行向量元素乘B的列向量元素组成的行向量 |
| fflags_o        | output   | 5          | 浮点异常标志（如 overflow, underflow 等）    |
| 传递控制信号    |          |            |                                              |
| ctrl_c_o        | output   | 4/8/16     | 传递控制信号，累加项 C                       |
| ctrl_rm_o       | output   | 3          | 传递控制信号，舍入模式                       |
| ctrl_reg_idxw_o | output   | 8          | 传递控制信号，目标寄存器索引                 |
| ctrl_warpid_o   | output   | DEPTH_WARP | 传递控制信号，Warp ID                        |

## 4.5 tc_add模块

该模块实现了对tc_mul阵列输出的行向量组的行累加，可以通过例化SHAPE_N=8个加法树实现，加法树的构造参考ventus的tc_add_pipe，同样也只是位宽的调整

| **引脚名称**    | **方向** | **位宽**   | **说明**                                  |
| --------------- | -------- | ---------- | ----------------------------------------- |
| 输入数据信号    |          |            |                                           |
| r_v_i           | input    | 8*8*9      | 输入行向量组元素数据                      |
| 只读控制信号    |          |            |                                           |
| rm_i            | input    | 3          | 舍入模式（IEEE 754）                      |
| ctrl_c_i        | input    | 4/8/16     | 控制信号，累加项 C                        |
| ctrl_rm_i       | input    | 3          | 控制信号，舍入模式                        |
| ctrl_reg_idxw_i | input    | 8          | 控制信号，目标寄存器索引                  |
| ctrl_warpid_i   | input    | DEPTH_WARP | 控制信号，Warp ID                         |
| 握手信号        |          |            |                                           |
| in_valid_i      | input    | 1          | 握手信号，输入数据是否有效                |
| out_ready_i     | input    | 1          | 握手信号，下游模块是否准备好接收输出      |
| in_ready_o      | output   | 1          | 握手信号，本模块是否准备好接收数据        |
| out_valid_o     | output   | 1          | 握手信号，输出数据是否有效                |
| 输出数据信号    |          |            |                                           |
| result_o        | output   | 8*9        | 对行向量组沿行求和的列向量                |
| fflags_o        | output   | 5          | 浮点异常标志（如 overflow, underflow 等） |
| 传递控制信号    |          |            |                                           |
| ctrl_c_o        | output   | 4/8/16     | 传递控制信号，累加项 C                    |
| ctrl_rm_o       | output   | 3          | 传递控制信号，舍入模式                    |
| ctrl_reg_idxw_o | output   | 8          | 传递控制信号，目标寄存器索引              |
| ctrl_warpid_o   | output   | DEPTH_WARP | 传递控制信号，Warp ID                     |

# 5 数据通道模块 

## 5.1 tc_mul模块

该模块用于对单个A矩阵元素和B矩阵元素相乘，是构成点积模块的基本单元之一。

1. 乘法流水线分3级实现，用反压信号控制流水线寄存器更新：

1. 预处理阶段（fmul_s1）和尾数相乘（naivemultiplier）
2. 第一级输出整合阶段（fmul_s2）
3. 舍入与规格化阶段（fmul_s3）
4. 扩位至fp22

| **引脚名称**    | **方向** | **位宽**   | **说明**                                  |
| --------------- | -------- | ---------- | ----------------------------------------- |
| 输入数据信号    |          |            |                                           |
| a\|b_i          | input    | 5+4        | 输入A/B矩阵元素数据                       |
| 只读控制信号    |          |            |                                           |
| rm_i            | input    | 3          | 舍入模式（IEEE 754）                      |
| ctrl_c_i        | input    | 4/8/16     | 控制信号，累加项 C                        |
| ctrl_rm_i       | input    | 3          | 控制信号，舍入模式                        |
| ctrl_reg_idxw_i | input    | 8          | 控制信号，目标寄存器索引                  |
| ctrl_warpid_i   | input    | DEPTH_WARP | 控制信号，Warp ID                         |
| 握手信号        |          |            |                                           |
| in_valid_i      | input    | 1          | 握手信号，输入数据是否有效                |
| out_ready_i     | input    | 1          | 握手信号，下游模块是否准备好接收输出      |
| in_ready_o      | output   | 1          | 握手信号，本模块是否准备好接收数据        |
| out_valid_o     | output   | 1          | 握手信号，输出数据是否有效                |
| 输出数据信号    |          |            |                                           |
| result_o        | output   | 5+4        | 浮点乘法结果（带累加）                    |
| fflags_o        | output   | 5          | 浮点异常标志（如 overflow, underflow 等） |
| 传递控制信号    |          |            |                                           |
| ctrl_c_o        | output   | 4/8/16     | 传递控制信号，累加项 C                    |
| ctrl_rm_o       | output   | 3          | 传递控制信号，舍入模式                    |
| ctrl_reg_idxw_o | output   | 8          | 传递控制信号，目标寄存器索引              |
| ctrl_warpid_o   | output   | DEPTH_WARP | 传递控制信号，Warp ID                     |

## 5.2 tc_add模块

1. 该模块用于实现加法树和与C矩阵元素相加，是构成点积模块的基本单元之一
2. 加法流水线分2级实现，用反压信号控制流水线寄存器更新：

1. fadd_s1对两个浮点数分类，选择计算路径相加
2. fadd_s2处理舍入，溢出和规格化

pinlist和tc_mul_pipe模块相同，位宽略有差异

## 5.3tc_mm_add模块

在tc_mm_add模块例化时需注意：

1. 对前级模块mv_mul的输出进行整合：前级输出为列向量组，进行位拼接形成矩阵
2. 混合精度相加时，需要适当扩位

| **引脚名称**    | **方向** | **位宽**   | **说明**                                  |
| --------------- | -------- | ---------- | ----------------------------------------- |
| 输入数据信号    |          |            |                                           |
| c_v_i           | input    | 8*8*9      | 输入列向量组元素数据                      |
| 只读控制信号    |          |            |                                           |
| rm_i            | input    | 3          | 舍入模式（IEEE 754）                      |
| ctrl_c_i        | input    | 4/8/16     | 控制信号，累加项 C                        |
| ctrl_rm_i       | input    | 3          | 控制信号，舍入模式                        |
| ctrl_reg_idxw_i | input    | 8          | 控制信号，目标寄存器索引                  |
| ctrl_warpid_i   | input    | DEPTH_WARP | 控制信号，Warp ID                         |
| 握手信号        |          |            |                                           |
| in_valid_i      | input    | 1          | 握手信号，输入数据是否有效                |
| out_ready_i     | input    | 1          | 握手信号，下游模块是否准备好接收输出      |
| in_ready_o      | output   | 1          | 握手信号，本模块是否准备好接收数据        |
| out_valid_o     | output   | 1          | 握手信号，输出数据是否有效                |
| 输出数据信号    |          |            |                                           |
| result_o        | output   | 512        | 行主序输出的结果矩阵                      |
| fflags_o        | output   | 5          | 浮点异常标志（如 overflow, underflow 等） |
| 传递控制信号    |          |            |                                           |
| ctrl_c_o        | output   | 4/8/16     | 传递控制信号，累加项 C                    |
| ctrl_rm_o       | output   | 3          | 传递控制信号，舍入模式                    |
| ctrl_reg_idxw_o | output   | 8          | 传递控制信号，目标寄存器索引              |
| ctrl_warpid_o   | output   | DEPTH_WARP | 传递控制信号，Warp ID                     |

# 6 基本计算模块

## 6.1 fmul_s1模块

该模块具有以下功能：

1.  对输入浮点数进行分类（规格化/非规格化/无穷/NaN等）
   \2. 计算初步乘法结果（符号/指数/尾数）
   \3. 处理特殊值情况（NaN/Inf/零等）
   \4. 为第二阶段准备规格化参数

| **引脚名称**               | **方向** | **位宽** | **说明**                  |
| -------------------------- | -------- | -------- | ------------------------- |
| 输入数据信号               |          |          |                           |
| s_axis_tdata_a             | input    | 5+4      | 矩阵 A 元素输入           |
| s_axis_tdata_b             | input    | 5+4      | 矩阵 B 元素输入           |
| 只读控制信号               |          |          |                           |
| rm_i                       | input    | 3        | 舍入模式（IEEE 754）      |
| 特殊情况与异常输出         |          |          |                           |
| out_special_case_valid_o   | output   | 1        | 特殊值标志是否有效        |
| out_special_case_nan_o     | output   | 1        | 结果是否为 NaN            |
| out_special_case_inf_o     | output   | 1        | 结果是否为 ±Inf           |
| out_special_case_inv_o     | output   | 1        | 是否为无效操作（invalid） |
| out_special_case_haszero_o | output   | 1        | 是否存在零操作数          |
| out_early_overflow_o       | output   | 1        | 是否发生早期溢出          |
| out_may_be_subnormal_o     | output   | 1        | 可能为非规格化数          |
| 输出数据信号               |          |          |                           |
| out_prod_sign_o            | output   | 1        | 乘积符号位                |
| out_shift_amt_o            | output   | 5+1      | 最终移位量                |
| out_exp_shifted_o          | output   | 5+1      | 移位后指数                |
| 传递控制信号               |          |          |                           |
| out_rm_o                   | output   | 3        | 传递舍入模式              |

## 6.2naivemultiplier模块

1. 该模块用于尾数相乘，LEN=4

| **引脚名称**   | **方向** | **位宽** | **说明**               |
| -------------- | -------- | -------- | ---------------------- |
| regenable      | input    | 1        | 控制信号，寄存器写使能 |
| s_axis_tdata_a | input    | LEN      | 矩阵 A 元素尾数输入    |
| s_axis_tdata_b | input    | LEN      | 矩阵 B 元素尾数输入    |
| result         | output   | LEN * 2  | 数据输出，尾数相乘结果 |

## 6.3 fmul_s2模块

该模块用于传递第一阶段处理结果，EXPWIDTH=5，PRECISION=4

| **引脚名称**                 | **方向** | **位宽**    | **说明**         |
| ---------------------------- | -------- | ----------- | ---------------- |
| 特殊情况、异常与舍入控制信号 |          |             |                  |
| in_special_case_valid_i      | inout    | 1           | 特殊路径有效标志 |
| in_special_case_nan_i        | inout    | 1           | NaN 结果标志     |
| in_special_case_inf_i        | inout    | 1           | 无穷标志         |
| in_special_case_inv_i        | inout    | 1           | 无效操作标志     |
| in_special_case_haszero_i    | inout    | 1           | 存在零操作数     |
| in_early_overflow_i          | inout    | 1           | 早期溢出标志     |
| in_may_be_subnormal_i        | inout    | 1           | 可能为非规格化数 |
| in_rm_i                      | inout    | 3           | 舍入模式         |
| 输出数据信号                 |          |             |                  |
| in_prod_sign_i               | inout    | 1           | 乘积符号位       |
| in_shift_amt_i               | inout    | EXPWIDTH+1  | 尾数移位量       |
| in_exp_shifted_i             | inout    | EXPWIDTH+1  | 调整后指数       |
| prod_i                       | inout    | PRECISION*2 | 尾数乘积         |

## 6.4 fmul_s3模块

EXPWIDTH=5, PRECISION=4

该模块的功能为：

1.  对乘法结果进行最终舍入处理
   \2. 处理溢出情况
   \3. 生成规格化结果和异常标志
   \4. 输出到加法树的中间结果

| **引脚名称**                 | **方向** | **位宽**             | **说明**                                   |
| ---------------------------- | -------- | -------------------- | ------------------------------------------ |
| 输入数据信号                 |          |                      |                                            |
| in_prod_i                    | input    | PRECISION * 2        | 数据信号，乘积中间结果                     |
| in_prod_sign_i               | input    | 1                    | 乘积符号位                                 |
| in_shift_amt_i               | input    | EXPWIDTH + 1         | 数据信号，移位量                           |
| in_exp_shifted_i             | input    | EXPWIDTH + 1         | 数据信号，移位后指数                       |
| 特殊情况与异常输入           |          |                      |                                            |
| in_special_case_valid_i      | input    | 1                    | 特殊路径有效标志                           |
| in_special_case_nan_i        | input    | 1                    | NaN 结果标志                               |
| in_special_case_inf_i        | input    | 1                    | 无穷标志                                   |
| in_special_case_inv_i        | input    | 1                    | 无效操作标志                               |
| in_special_case_haszero_i    | input    | 1                    | 存在零操作数标志                           |
| in_early_overflow_i          | input    | 1                    | 早期溢出标志                               |
| in_may_be_subnormal_i        | input    | 1                    | 可能为非规格化数                           |
| 只读控制信号                 |          |                      |                                            |
| in_rm_i                      | input    | 3                    | 舍入模式（IEEE 754）                       |
| 输出数据信号                 |          |                      |                                            |
| result_o                     | output   | EXPWIDTH + PRECISION | 数据信号，最终规格化结果                   |
| fflags_o                     | output   | 5                    | 异常标志（NV, OF, UF, DZ, NX）             |
| 输出到加法树的数据与标志信号 |          |                      |                                            |
| to_fadd_fp_prod_sign_o       | output   | 1                    | 输出给加法器的符号位                       |
| to_fadd_fp_prod_exp_o        | output   | EXPWIDTH             | 数据信号，输出给加法器的指数               |
| to_fadd_fp_prod_sig_o        | output   | 2 * PRECISION - 1    | 数据信号，输出给加法器的尾数（怎么扩位？） |
| to_fadd_is_nan_o             | output   | 1                    | 输出 NaN 标志                              |
| to_fadd_is_inf_o             | output   | 1                    | 输出无穷标志                               |
| to_fadd_is_inv_o             | output   | 1                    | 输出无效操作标志                           |
| to_fadd_overflow_o           | output   | 1                    | 输出溢出标志                               |

## 6.5 fadd_s1模块

- 当作为加法树时，EXPWIDTH=5,     PRECISION=8, OUTPC=4
- 当作为累加器时，EXPWIDTH=8,     PRECISION=28, OUTPC=14

| **引脚名称**                              | **方向** | **位宽**             | **说明**                            |
| ----------------------------------------- | -------- | -------------------- | ----------------------------------- |
| 输入数据信号                              |          |                      |                                     |
| a_i                                       | input    | EXPWIDTH + PRECISION | 操作数a                             |
| b_i                                       | input    | EXPWIDTH + PRECISION | 操作数b                             |
| 用于fma指令的标志信号，根据需要选择性保留 |          |                      |                                     |
| b_inter_valid_i                           | input    | 1                    | 中间结果是否有效                    |
| b_inter_flags_is_nan_i                    | input    | 1                    | 中间结果为 NaN 标志                 |
| b_inter_flags_is_inf_i                    | input    | 1                    | 中间结果为无穷标志                  |
| b_inter_flags_is_inv_i                    | input    | 1                    | 无效操作标志                        |
| b_inter_flags_overflow_i                  | input    | 1                    | 溢出标志                            |
| 舍入模式控制信号                          |          |                      |                                     |
| rm_i                                      | input    | 3                    | 舍入模式（IEEE 754）                |
| out_rm_o                                  | output   | 3                    | 传递的舍入模式                      |
| 输出给下一级的数据信号                    |          |                      |                                     |
| out_far_sign_o                            | output   | 1                    | far path 符号位                     |
| out_far_exp_o                             | output   | EXPWIDTH             | far path 指数                       |
| out_far_sig_o                             | output   | OUTPC + 3            | far path 尾数（带保护位）           |
| out_near_sign_o                           | output   | 1                    | near path 符号位                    |
| out_near_exp_o                            | output   | EXPWIDTH             | near path 指数                      |
| out_near_sig_o                            | output   | OUTPC + 3            | near path 尾数（带保护位）          |
| 特殊情况与异常输出                        |          |                      |                                     |
| out_special_case_nan_o                    | output   | 1                    | 结果为 NaN 标志                     |
| out_special_case_inf_sign_o               | output   | 1                    | 无穷结果的符号位                    |
| out_small_add_o                           | output   | 1                    | 小加法标志（表示非规格数加法）      |
| out_far_mul_of_o                          | output   | 1                    | far path 乘法溢出标志               |
| out_near_sig_is_zero_o                    | output   | 1                    | near path 尾数为零标志              |
| 路径选择信号                              |          |                      |                                     |
| out_sel_far_path_o                        | output   | 1                    | 路径选择信号（1 表示选择 far path） |

## 6.6 fadd_s2模块

- 当作为加法树时，EXPWIDTH=5,     PRECISION=4
- 当作为累加器时，EXPWIDTH=8,     PRECISION=14

该模块的功能为：

1.  对第一阶段结果进行舍入处理
   \2. 处理溢出情况
   \3. 生成最终规格化结果
   \4. 输出异常标志

| **引脚名称**            | **方向** | **位宽**             | **说明**                           |
| ----------------------- | -------- | -------------------- | ---------------------------------- |
| 输入数据信号            |          |                      |                                    |
| in_far_sign_i           | input    | 1                    | far path 符号位                    |
| in_far_exp_i            | input    | EXPWIDTH             | far path 指数                      |
| in_far_sig_i            | input    | PRECISION + 3        | far path 尾数（含保护位）          |
| in_near_sign_i          | input    | 1                    | near path 符号位                   |
| in_near_exp_i           | input    | EXPWIDTH             | near path 指数                     |
| in_near_sig_i           | input    | PRECISION + 3        | near path 尾数                     |
| 路径选择信号            |          |                      |                                    |
| in_sel_far_path_i       | input    | 1                    | 路径选择信号（1 表示 far path）    |
| 特殊情况与异常输入      |          |                      |                                    |
| rm_i                    | input    | 3                    | 舍入模式（IEEE 754）               |
| in_far_mul_of_i         | input    | 1                    | far path 乘法溢出标志              |
| in_near_sig_is_zero_i   | input    | 1                    | near path 尾数为零                 |
| in_special_case_valid_i | input    | 1                    | 特殊路径有效标志                   |
| in_special_case_iv_i    | input    | 1                    | 无效操作标志（Invalid）            |
| in_special_case_nan_i   | input    | 1                    | NaN 结果标志                       |
| 输出数据和标志信号      |          |                      |                                    |
| out_result_o            | output   | EXPWIDTH + PRECISION | 最终计算结果                       |
| out_fflags_o            | output   | 5                    | 浮点异常标志（NV, OF, UF, DZ, NX） |
| out_far_uf_o            | output   | 1                    | far path 下溢标志                  |
| out_near_of_o           | output   | 1                    | near path 溢出标志                 |

## 6.7far_path模块

- 当作为加法树时，EXPWIDTH=5,     PRECISION=8, OUTPC=4
- 当作为累加器时，EXPWIDTH=8,     PRECISION=28, OUTPC=14

该模块实现的是浮点加法/减法中的 Far Path 加法路径，即当两个浮点数指数差 expdiff_i 大于等于 2 时使用的路径。

其主要功能包括：

1. 对阶并对尾数对齐：
   使用 shift_right_jam 模块将较小数尾数右移（带粘性位 sticky）对齐。

2. 尾数相加/相减：
   根据 effsub_i 控制是否为减法（即两数符号不同）。采用补码加法方式处理带符号的相加减。

3. 规格化：

4. - 根据加法结果 addr_result 的高位判断是否需要规格化：是否有进位（cout）、是否可以保持原指数（keep）、是否需要减小指数（cancellation）。
   - 最终尾数取出 OUTPC 位有效位并添加保护位、粘性位等。

5. 输出最终结果：

6. - 指数调整后输出为 result_exp_o；
   - 尾数截取为 result_sig_o；
   - 符号位保持为较大数（此处为 A）的符号。



| **名称**      | **方向** | **位宽**        | **说明**                         |
| ------------- | -------- | --------------- | -------------------------------- |
| a_sign_i      | input    | 1               | A 操作数的符号位                 |
| a_exp_i       | input    | [EXPWIDTH-1:0]  | A 操作数的指数位                 |
| a_sig_i       | input    | [PRECISION-1:0] | A 操作数的尾数                   |
| b_sig_i       | input    | [PRECISION-1:0] | B 操作数的尾数（已确认指数更小） |
| expdiff_i     | input    | [EXPWIDTH-1:0]  | 指数差 a_exp_i - b_exp_i         |
| effsub_i      | input    | 1               | 有效减法控制位（不同符号）       |
| small_add_i   | input    | 1               | 特殊小加法标志（用于规约控制）   |
| result_sign_o | output   | 1               | 结果符号位（继承自 A）           |
| result_exp_o  | output   | [EXPWIDTH-1:0]  | 结果指数                         |
| result_sig_o  | output   | [OUTPC+2:0]     | 规格化尾数（包含保留位和粘性位） |

## 6.8 near_path模块

- 当作为加法树时，EXPWIDTH=5,     PRECISION=8, OUTPC=4
- 当作为累加器时，EXPWIDTH=8,     PRECISION=28, OUTPC=14

near_path 模块实现了浮点减法中指数差值较小（通常 <2） 的近路径计算逻辑

| **名称**       | **方向** | **位宽**        | **说明**                            |
| -------------- | -------- | --------------- | ----------------------------------- |
| a_sign_i       | input    | 1               | A 操作数的符号位                    |
| a_exp_i        | input    | [EXPWIDTH-1:0]  | A 操作数的指数                      |
| a_sig_i        | input    | [PRECISION-1:0] | A 操作数的尾数                      |
| b_sign_i       | input    | 1               | B 操作数的符号位                    |
| b_sig_i        | input    | [PRECISION-1:0] | B 操作数的尾数                      |
| need_shift_b_i | input    | 1               | 是否需要对 B 的尾数进行右移         |
| result_sign_o  | output   | 1               | 近路径计算结果的符号位              |
| result_exp_o   | output   | [EXPWIDTH-1:0]  | 近路径计算结果的指数                |
| result_sig_o   | output   | [OUTPC+2:0]     | 近路径计算结果的规格化尾数          |
| sig_is_zero_o  | output   | 1               | 尾数是否为全 0                      |
| a_lt_b_o       | output   | 1               | 是否 A 的尾数小于 B（用于符号判断） |

# 7 舍入模块

该模块用于根据舍入模式处理输入浮点数，WIDTH为父模块PRECISION-1

| **引脚名称** | **方向** | **位宽** | **说明**                          |
| ------------ | -------- | -------- | --------------------------------- |
| in           | input    | WIDTH    | 需舍入的原始尾数                  |
| sign         | input    | 1        | 数值符号位，用于判断向上/向下舍入 |
| roundin      | input    | 1        | 舍入位（Round bit）               |
| stickyin     | input    | 1        | 黏着位（Sticky bit）              |
| rm           | input    | 3        | 舍入模式（RNE/RTZ/RUP/RDN/RMM）   |
| out          | output   | WIDTH    | 尾数舍入结果                      |
| inexact      | output   | 1        | 是否存在精度丢失（即需舍入）      |
| cout         | output   | 1        | 进位标志（舍入产生进位）          |
| r_up         | output   | 1        | 是否发生了向上舍入                |

# 8 辅助计算模块

由于fp8的尾数较少，因此可能会有比ventus长尾数操作逻辑门更少、延迟更小的算法，具有一定的优化空间

## 8.1shift_right_jam右移模块

1. 实现右移并检测黏着位是否为1

| **引脚名称** | **方向** | **位宽**  | **含义**                    |
| ------------ | -------- | --------- | --------------------------- |
| in           | input    | [LEN-1:0] | 原始输入数据                |
| shamt        | input    | [EXP-1:0] | 右移的位数（移位量）        |
| out          | output   | [LEN-1:0] | 右移后的输出结果            |
| sticky       | output   | 1         | 粘滞位，表示移出位是否包含1 |

## 8.2 lza前导0预测模块

该模块用于提前预测运算结果中前导零的位置，加快归一化过程。

| **引脚名称** | **方向** | **位宽**  | **含义**                       |
| ------------ | -------- | --------- | ------------------------------ |
| a            | input    | [LEN-1:0] | 输入操作数 A                   |
| b            | input    | [LEN-1:0] | 输入操作数 B                   |
| c            | output   | [LEN-1:0] | LZA 输出结果，供前导零检测使用 |

## 8.3 lzc前导0计数模块

1. 位于文件ventus-gpgpu-verilog/src/gpgpu_top/sm/pipeline/sfu_v2/float_div_mvp/[lzc.sv](http://lzc.sv/)中，这里用到了cf_math_pkg包中的idx_width函数，详见[cf_math_pkg::idx_width函数](http://tensorth.dscloud.me:28090/pages/viewpage.action?pageId=90013913)

归约二叉树实现的尾零计数器 / 首零计数器模块：

1. 当 MODE 设为 0 时，模块作为尾零计数器使用，cnt_o 表示从最低有效位（LSB）开始的连续 0 的数量；
   当 MODE 设为 1 时，模块作为首零计数器使用，cnt_o 表示从最高有效位（MSB）开始的连续 0 的数量。
2. 如果输入中全为 0（不包含任何 1），则 empty_o 置为 1，同时cnt_o 的值为最大可能的零数量减 1。
   例如（MODE = 0）：

- 输入 in_i = 000_0000，则 empty_o = 1，cnt_o = 6
- 输入 in_i = 000_0001，则 empty_o = 0，cnt_o = 0
- 输入 in_i = 000_1000，则 empty_o = 0，cnt_o = 3



| **参数名** | **类型**     | **默认值** | **说明**                              |
| ---------- | ------------ | ---------- | ------------------------------------- |
| WIDTH      | int unsigned | 2          | 输入向量的位宽                        |
| MODE       | bit          | 1'b0       | 模式选择：0 → 尾零计数，1 → 首零计数  |
| CNT_WIDTH  | int unsigned | 自动计算   | 输出计数结果的位宽（根据 WIDTH 推导） |

 

| **信号名** | **方向** | **位宽**  | **说明**                                 |
| ---------- | -------- | --------- | ---------------------------------------- |
| in_i       | input    | WIDTH     | 输入向量                                 |
| cnt_o      | output   | CNT_WIDTH | 首/尾部 0 的数量                         |
| empty_o    | output   | 1         | 若输入全为 0，该信号为 1，表示计数器“空” |

# 9 存储模块SRAM

![img](https://tensor03.cn6.quickconnect.cn/direct/oo/file/1030_KH2F6SQNFD3A78OS4LJRUBF8LO.doc/B40RVNKPT151TETVFJK8MAIL80/img?tid="tto6RZUAk5iD12wYIH4XHLY1mYCynzQ5J7aQtghIrYy1wIoEM6EdEd8mwtwPf6MqEU_9h8dZIbGmGu23"&linkId="15XlfrIkANyLM8uwRMhOMQmbWbIl3wAP")

## 9.1 amem模块

该模块用于存储经过to_fp9_con转换后的计算结果（fp9格式的A数据）

- **功能**：

- - 按照规定的格式缓存从输入矩阵数据到定制存储器
  - 为后续的 MAC 计算阵列提供低延迟的数据访问。

- **设计细节**：

- - 每个寄存器Tile分32个子块，每个字块占4个bit单元（4byte）共16byte。（待确认大小）
  - bit单元为最小存储单元必须存放连续的数据块。

- - 加载数据平均分配到32个子快中，如有剩余空间，通过texture拷贝补齐以达到均衡负载的目的。补齐运算通过输入地址映射模块完成，不会产生额外的开销。

- 支持多路并行访问，以满足 MAC 计算阵列的高吞吐需求。

 

## 9.2 bmem模块

该模块用于存储经过to_fp9_con转换后的计算结果（fp9格式的B数据）

- **功能**：

- - 按照规定的格式缓存从输入矩阵数据到定制存储器
  - 为后续的 MAC 计算阵列提供低延迟的数据访问。

- **设计细节**：

- - 每个寄存器Tile分32个子块，每个字块占4个bit单元（4byte）共16byte。（待确认大小）
  - bit单元为最小存储单元必须存放连续的数据块。

- - 加载数据平均分配到32个子块中，如有剩余空间，通过texture拷贝补齐以达到均衡负载的目的。补齐运算通过输入地址映射模块完成，不会产生额外的开销。

- 支持多路并行访问，以满足 MAC 计算阵列的高吞吐需求。

 

## 9.3 cmem模块

该模块用于存储经过to_next_con转换后的计算结果（fp4-fp22格式的C数据）

- **功能**：

- - 按照规定的格式缓存从输入矩阵数据到定制存储器
  - 为后续的 MAC 计算阵列提供低延迟的数据访问。

- **设计细节**：

- - 每个寄存器Tile分32个子块，每个字块占4个bit单元（4byte）共16byte。（待确认大小）
  - bit单元为最小存储单元必须存放连续的数据块。

- - 加载数据平均分配到32个子快中，如有剩余空间，通过texture拷贝补齐以达到均衡负载的目的。补齐运算通过输入地址映射模块完成，不会产生额外的开销。

- 支持多路并行访问，以满足 MAC 计算阵列的高吞吐需求。

 

## 9.4 md_data模块

该模块用于存储经过mm_mul_add模块计算后的计算结果，输入为中间矩阵和C矩阵元素累加后的结果，存储数据精度为fp4-fp22。

- **功能**：

- - 按照规定的格式缓存Mac计算整列输出的累加结果
  - 支持多轮累加操作，以实现更高精度的输出。

- **设计细节**：

- - 输出寄存器Tile分32个子块，每个字块占8个寄存器单元（4byte）共32byte。
  - 寄存器单元为最小存储单元必须存放连续的数据块。

- - 将输出数据平均分配到32个子快中。

- 支持多路并行访问，以满足 MAC 计算阵列的高吞吐需求。

 

 