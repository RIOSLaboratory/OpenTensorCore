`timescale 1ns/1ns
`include "define.v"

module tb_tensor_core;
    // 时钟与复位配置
    parameter CLK_PERIOD = 10;  // 100MHz时钟
    reg clk;
    reg rst_n;

    // 配置参数（与DUT匹配）
    parameter VL        = `NUM_THREAD;
    parameter SHAPE_M   = 8;
    parameter SHAPE_K   = 8;
    parameter SHAPE_N   = 8;
    parameter EXPWIDTH  = 5;
    parameter PRECISION = 4;

    // 数据总线宽度计算
    localparam XLEN_FP8   = 8;
    localparam XLEN_FP9   = 9;
    localparam MATRIX_BUS = `MATRIX_BUS_WIDTH;

    // 输入信号
    reg [MATRIX_BUS*XLEN_FP9/XLEN_FP8-1:0] a_i;
    reg [MATRIX_BUS*XLEN_FP9/XLEN_FP8-1:0] b_i;
    reg [MATRIX_BUS-1:0]                   c_i;
    reg [VL*3-1:0]                         rm_i;
    reg                                    in_valid_i;
    reg                                    out_ready_i;
    reg [4:0]                              type_ab;
    reg [2:0]                              type_ab_sub;
    reg [4:0]                              type_cd;

    // 输出信号
    wire                                   in_ready_o;
    wire                                   out_valid_o;
    wire [MATRIX_BUS+VL+1+8+`DEPTH_WARP-1:0] result_o;
    wire [VL*5-1:0]                        fflags_o;
    wire [7:0]                             ctrl_reg_idxw_o;
    wire [`DEPTH_WARP-1:0]                 ctrl_warpid_o;

    // 测试用例存储
    reg [XLEN_FP9-1:0] a_matrix [0:SHAPE_M-1][0:SHAPE_K-1];
    reg [XLEN_FP9-1:0] b_matrix [0:SHAPE_K-1][0:SHAPE_N-1];
    reg [XLEN_FP8-1:0] c_matrix [0:SHAPE_M-1][0:SHAPE_N-1];
    reg [XLEN_FP8-1:0] expected_result [0:SHAPE_M-1][0:SHAPE_N-1];
    integer test_pass;  // 测试通过计数器

    // DUT例化
    tensor_core #(
        .VL        (VL),
        .SHAPE_M   (SHAPE_M),
        .SHAPE_K   (SHAPE_K),
        .SHAPE_N   (SHAPE_N),
        .EXPWIDTH  (EXPWIDTH),
        .PRECISION (PRECISION)
    ) U_tensor_core (
        .clk            (clk),
        .rst_n          (rst_n),
        .a_i            (a_i),
        .b_i            (b_i),
        .c_i            (c_i),
        .rm_i           (rm_i),
        .in_valid_i     (in_valid_i),
        .out_ready_i    (out_ready_i),
        .in_ready_o     (in_ready_o),
        .out_valid_o    (out_valid_o),
        .type_ab        (type_ab),
        .type_ab_sub    (type_ab_sub),
        .type_cd        (type_cd),
        .result_o       (result_o),
        .fflags_o       (fflags_o),
        .ctrl_reg_idxw_o(ctrl_reg_idxw_o),
        .ctrl_warpid_o  (ctrl_warpid_o)
    );

    // 时钟生成
    initial begin
        clk = 1'b0;
        forever #(CLK_PERIOD/2) clk = ~clk;
    end

    // 复位流程
    initial begin
        rst_n = 1'b0;
        #(CLK_PERIOD*5);
        rst_n = 1'b1;
    end

    // 测试主流程
    initial begin
        // 初始化输入
        a_i         = '0;
        b_i         = '0;
        c_i         = '0;
        rm_i        = '0;
        in_valid_i  = 1'b0;
        out_ready_i = 1'b1;
        type_ab     = 5'd0;
        type_ab_sub = 3'd0;
        type_cd     = 5'd0;
        test_pass   = 0;

        // 等待复位释放
        @(posedge rst_n);
        #(CLK_PERIOD*2);

        // 执行各类验证场景
        $display("=== Starting Basic Function Test ===");
        basic_function_test();
        
        $display("\n=== Starting Data Precision Test ===");
        data_precision_test();
        
        $display("\n=== Starting Handshake Test ===");
        handshake_signal_test();
        
        $display("\n=== Starting Operation Mode Test ===");
        operation_mode_test();
        
        $display("\n=== Starting Exception Test ===");
        exception_scenario_test();

        // 完成测试
        $display("\n=== All %0d tests passed ===", test_pass);
        $finish;
    end

    // 1. 基础功能验证：验证基本矩阵运算正确性
    task basic_function_test;
        integer i, j, k;
        reg [XLEN_FP8*2-1:0] temp;  // 临时变量防止溢出
        begin
            // 加载已知测试数据（非随机）
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (k = 0; k < SHAPE_K; k = k + 1) begin
                    a_matrix[i][k] = {1'b0, i + k};  // 简单值便于计算
                end
            end

            for (k = 0; k < SHAPE_K; k = k + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    b_matrix[k][j] = {1'b0, k + j};
                end
            end

            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    c_matrix[i][j] = i + j;
                    // 计算期望结果 (A*B + C)
                    expected_result[i][j] = 0;
                    for (k = 0; k < SHAPE_K; k = k + 1) begin
                        temp = (a_matrix[i][k][XLEN_FP8-1:0] * b_matrix[k][j][XLEN_FP8-1:0]);
                        expected_result[i][j] = expected_result[i][j] + temp[XLEN_FP8-1:0] + c_matrix[i][j];
                    end
                end
            end

            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();
            test_pass = test_pass + 1;
        end
    endtask

    // 2. 数据精度验证：验证不同精度模式下的运算结果
    task data_precision_test;
        integer i, j, k;
        begin
            // 测试零值输入
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (k = 0; k < SHAPE_K; k = k + 1) begin
                    a_matrix[i][k] = 0;  // 零值输入
                    b_matrix[k][i] = {1'b0, 8'hFF};  // 最大值
                end
                c_matrix[i][i] = 0;
                expected_result[i][i] = 0;  // 零乘以任何数为零
            end

            // 切换精度模式
            type_ab = 5'd1;  // 假设1表示高精度模式
            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();

            // 测试极小值
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (k = 0; k < SHAPE_K; k = k + 1) begin
                    a_matrix[i][k] = {1'b0, 8'h01};  // 最小值
                    b_matrix[k][i] = {1'b0, 8'h01};
                end
                c_matrix[i][i] = 0;
                expected_result[i][i] = SHAPE_K * 1 * 1;  // K个1*1相加
            end

            type_ab = 5'd2;  // 低精度模式
            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();
            test_pass = test_pass + 1;
        end
    endtask

    // 3. 握手信号验证：验证valid-ready机制
    task handshake_signal_test;
        begin
            // 测试1: 当in_valid有效但in_ready无效时
            load_test_data(1);  // 加载简单测试集
            @(posedge clk);
            in_valid_i = 1'b1;
            out_ready_i = 1'b0;  // 先不准备接收输出
            repeat(3) @(posedge clk);  // 保持3个周期
            if (in_ready_o) begin
                $error("Handshake test failed: in_ready should be low when out_ready is low");
                $finish;
            end

            // 测试2: 释放out_ready
            @(posedge clk);
            out_ready_i = 1'b1;
            while (!in_ready_o) @(posedge clk);
            in_valid_i = 1'b0;
            wait_result_and_verify();

            // 测试3: 背压测试（连续输入）
            load_test_data(2);
            @(posedge clk);
            in_valid_i = 1'b1;
            @(posedge clk);
            in_valid_i = 1'b1;  // 连续输入
            while (!in_ready_o) @(posedge clk);
            in_valid_i = 1'b0;
            wait_result_and_verify();
            test_pass = test_pass + 1;
        end
    endtask

    // 4. 运算模式验证：验证不同运算类型
    task operation_mode_test;
        integer i, j, k;
        begin
            // 模式1: 普通乘加
            type_ab = 5'd0;
            type_cd = 5'd0;
            load_test_data(3);
            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();

            // 模式2: 饱和运算
            type_ab = 5'd3;  // 假设3表示饱和模式
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    expected_result[i][j] = {XLEN_FP8{1'b1}};  // 饱和到最大值
                end
            end
            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();

            // 模式3: 不同舍入模式
            rm_i = {VL{3'b001}};  // 向上舍入
            load_test_data(4);
            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();
            test_pass = test_pass + 1;
        end
    endtask

    // 5. 异常场景验证：验证边界情况
    task exception_scenario_test;
        begin
            // 测试1: 复位期间输入
            rst_n = 1'b0;
            in_valid_i = 1'b1;
            a_i = {MATRIX_BUS*XLEN_FP9/XLEN_FP8{1'b1}};  // 全1输入
            @(posedge clk);
            if (in_ready_o) begin
                $error("Exception test failed: in_ready should be low during reset");
                $finish;
            end
            rst_n = 1'b1;
            #(CLK_PERIOD*2);

            // 测试2: 无效数据格式
            type_ab = 5'd31;  // 超出范围的类型
            load_test_data(5);
            pack_data_to_bus();
            start_computation();
            @(posedge out_valid_o);
            if (fflags_o == 0) begin
                $error("Exception test failed: fflags should indicate error for invalid type");
                $finish;
            end

            // 测试3: 零矩阵运算
            load_zero_matrix();
            pack_data_to_bus();
            start_computation();
            wait_result_and_verify();
            test_pass = test_pass + 1;
        end
    endtask

    // 辅助任务：加载测试数据
    task load_test_data(input integer seed);
        integer i, j, k;
        begin
            $srandom(seed);  // 固定种子确保可重复
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (k = 0; k < SHAPE_K; k = k + 1) begin
                    a_matrix[i][k] = $urandom_range(0, 2**XLEN_FP9 - 1);
                end
            end

            for (k = 0; k < SHAPE_K; k = k + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    b_matrix[k][j] = $urandom_range(0, 2**XLEN_FP9 - 1);
                end
            end

            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    c_matrix[i][j] = $urandom_range(0, 2**XLEN_FP8 - 1);
                    // 计算期望结果
                    expected_result[i][j] = 0;
                    for (k = 0; k < SHAPE_K; k = k + 1) begin
                        expected_result[i][j] = expected_result[i][j] + 
                            a_matrix[i][k][XLEN_FP8-1:0] * b_matrix[k][j][XLEN_FP8-1:0] + c_matrix[i][j];
                    end
                end
            end
        end
    endtask

    // 辅助任务：加载零矩阵
    task load_zero_matrix;
        integer i, j, k;
        begin
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (k = 0; k < SHAPE_K; k = k + 1) begin
                    a_matrix[i][k] = 0;
                end
            end

            for (k = 0; k < SHAPE_K; k = k + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    b_matrix[k][j] = 0;
                end
            end

            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    c_matrix[i][j] = 0;
                    expected_result[i][j] = 0;
                end
            end
        end
    endtask

    // 数据打包到输入总线
    task pack_data_to_bus;
        integer i, j, k;
        begin
            // 打包a_i (M*K矩阵)
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (k = 0; k < SHAPE_K; k = k + 1) begin
                    a_i[(i*SHAPE_K + k + 1)*XLEN_FP9 - 1 : (i*SHAPE_K + k)*XLEN_FP9] = a_matrix[i][k];
                end
            end

            // 打包b_i (K*N矩阵)
            for (k = 0; k < SHAPE_K; k = k + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    b_i[(k*SHAPE_N + j + 1)*XLEN_FP9 - 1 : (k*SHAPE_N + j)*XLEN_FP9] = b_matrix[k][j];
                end
            end

            // 打包c_i (M*N矩阵)
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    c_i[(i*SHAPE_N + j + 1)*XLEN_FP8 - 1 : (i*SHAPE_N + j)*XLEN_FP8] = c_matrix[i][j];
                end
            end
        end
    endtask

    // 启动运算
    task start_computation;
        begin
            @(posedge clk);
            in_valid_i = 1'b1;
            @(posedge clk);
            while (!in_ready_o) @(posedge clk);
            in_valid_i = 1'b0;
        end
    endtask

    // 等待结果并验证
    task wait_result_and_verify;
        integer i, j;
        reg [XLEN_FP8-1:0] result_matrix [0:SHAPE_M-1][0:SHAPE_N-1];
        begin
            wait(out_valid_o);

            // 解包结果
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    result_matrix[i][j] = result_o[(i*SHAPE_N + j + 1)*XLEN_FP8 - 1 : (i*SHAPE_N + j)*XLEN_FP8];
                end
            end

            // 验证结果
            for (i = 0; i < SHAPE_M; i = i + 1) begin
                for (j = 0; j < SHAPE_N; j = j + 1) begin
                    if (result_matrix[i][j] !== expected_result[i][j]) begin
                        $error("Result mismatch at (%0d,%0d): expected 0x%x, got 0x%x",
                            i, j, expected_result[i][j], result_matrix[i][j]);
                        $finish;
                    end
                end
            end
            $display("Verification passed!");
        end
    endtask

    // 波形记录
    initial begin
        $dumpfile("tensor_core_wave.vcd");
        $dumpvars(0, tb_tensor_core);
    end

endmodule