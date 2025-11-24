`timescale 1ns/1ps
module tensor_core(
       input clk,
       input rst_n,
	// AXI input(a,b)
    input [511:0] s_axis_tdata_a_b,
	input        s_axis_tvalid_a_b,
	output       s_axis_tready_a_b,
	//AXI input(c)
	input [511:0] s_axis_tdata_c,
	input         s_axis_tvalid_c,
	output        s_axis_tready_c,
	//AXI output(d)
	output [511:0] m_axis_tdata_d,
	output        m_axis_tvalid_d,
	input         m_axis_tready_d,
	//control and interrupt
	input		en_i,
	output reg  busy_o,
	output reg  irq_o,
	//configuration interface
	input [3:0]	cfg_layout,
	input		cfg_transpose_en,
	input [4:0]	cfg_type_ab,
	input [2:0]	cfg_type_ab_sub,
	input [4:0]	cfg_type_cd,
	input [3:0] cfg_shape_m,
	input [3:0]	cfg_shape_n,
	input [3:0]	cfg_shape_k
);
	wire [3:0]	reg_layout;
	wire		reg_transpose_en;
	wire [4:0]	reg_type_ab;
	wire [2:0]	reg_type_ab_sub;
	wire [4:0]	reg_type_cd;
	wire [3:0]	reg_shape_m;
	wire [3:0]	reg_shape_n;
	wire [3:0]	reg_shape_k;
	wire		config_valid;
	wire		config_error;
config_registers u_cfg_regs(
	.clock		(clk),
	.reset_n	(rst_n),
	.busy		(busy_o),
	.cfg_latout			(cfg_latout),
	.cfg_transpose_en	(cfg_transpose_en),
	.cfg_type_ab_sub	(cfg_type_ab_sub),
	.cfg_type_ab		(cfg_type_ab),
	.cfg_type_cd		(cfg_type_cd),
	.cfg_shape_m		(cfg_shape_m),
	.cfg_shape_n		(cfg_shape_n),
	.cfg_shape_k		(cfg_shape_k),
	//output
	.layout_reg			(layout_reg),
	.transpose_en_reg	(transpose_en),
	.type_ab_sub_reg	(type_ab_sub),
	.type_cd_reg		(type_cd),
	.shape_m_reg		(reg_shape_m),
	.shape_n_reg		(reg_shape_n),
	.shape_k_reg		(reg_shape_k),
	.config_valid		(config_valid),
	.config_error		(config_error)
);
localparam IDLE = 2'b00;
localparam RUN = 2'b01;
reg [1:0]state;

always@(posedge clk or negedge rst_n)begin
	if(!rst_n)begin
	state <=IDLE;
	busy_o<=1'b0;
	irq_o<=1'b0;
	end else begin
	 case (state)
	 IDLE:begin
	 if(en_i && s_axis_tvalid_a_b && s_axis_tvalid_c)begin
	 state <= RUN;
	 busy_o <= 1'b1;
	 end
end
	RUN:begin
 		busy_o<=1'b1;
end
		default:state <= IDLE;
	endcase
	end
end

//simple handshake reponse
assign s_axis_tready_a_b =1'b1;
assign s_axis_tready_c =1'b1;

assign m_axis_tvalid_d = 1'b0;
assign m_axis_tdata_d =512'b0;



endmodule
