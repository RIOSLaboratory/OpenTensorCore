`timescale 1ns/1ps
module tb_tensor_core;
	//1.define signal
	reg clk;
	reg rst_n;
	reg en_i;
	reg [511:0]	s_axis_tdata_a_b;
	reg		s_axis_tvalid_a_b;
	reg [511:0]	s_axis_tdata_c;
	reg		s_axis_tvalid_c;
	reg		m_axis_tready_d;
	//configure_register
	reg [3:0] 	cfg_layout;
	reg 		cfg_transpose_en;
	reg [4:0]	cfg_type_ab;
	reg [2:0]	cfg_type_ab_sub;
	reg [4:0]	cfg_type_cd;
	reg [3:0] cfg_shape_m;
	reg [3:0]	cfg_shape_n;
	reg [3:0]	cfg_shape_k;
	//use wire for output signal from DUT
	wire		s_axis_tready_a_b;
	wire		s_axis_tready_c;
	wire [511:0]	m_axis_tdata_d;
	wire		m_axis_tvalid_d;
	wire		busy_o;
	wire		irq_o;
	//generate clock(100MHz)
	initial clk = 0;
	always #5 clk = ~clk;
	//instantiate DUT
tensor_core u_dot(
	.clk	(clk),
	.rst_n	(rst_n),
	//A,B
	.s_axis_tdata_a_b	(s_axis_tdata_a_b),
	.s_axis_tvalid_a_b	(s_axis_tvalid_a_b),
	.s_axis_tready_a_b	(s_axis_tready_a_b),
	//C
	.s_axis_tdata_c		(s_axis_tdata_c),	
	.s_axis_tvalid_c	(s_axis_tvalid_c),
	.s_axis_tready_c	(s_axis_tready_c),
	//D
	.m_axis_tdata_d		(m_axis_tdata_d),
	.m_axis_tvalid_d	(m_axis_tvalid_d),
	.m_axis_tready_d	(m_axis_tready_d),
	//control
	.en_i			(en_i),
	.busy_o			(busy_o),	
	.irq_o			(irq_o),		
	//configure port
	.cfg_layout			(cfg_layout),
	.cfg_transpose_en	(cfg_transpose_en),
	.cfg_type_ab_sub	(cfg_type_ab_sub),
	.cfg_type_ab		(cfg_type_ab),
	.cfg_type_cd		(cfg_type_cd),
	.cfg_shape_m		(cfg_shape_m),
	.cfg_shape_n		(cfg_shape_n),
	.cfg_shape_k		(cfg_shape_k)
	);
	//core logic
	initial begin
		//(A)initial all signal
		rst_n = 0;
		en_i = 0;
		s_axis_tdata_a_b = 0;
		s_axis_tvalid_a_b = 0;
		s_axis_tdata_c = 0;
		s_axis_tvalid_c = 0;
		m_axis_tready_d = 1;
		//initial configuration
		cfg_layout=0;cfg_transpose_en=0;
		cfg_type_ab=0;cfg_type_ab_sub=0;cfg_type_cd=0;
		cfg_shape_m=0;cfg_shape_n=0;cfg_shape_k=2;
		//(B)reset 
		#100;
		//
		$display("/n---[Item 1]Reset Released");
		if(busy_o===0 && en_i ===0) 
			$display("PASS:Reset succeddful(busy_o is 0).");
		else 
		 $display("FAIL:Reset failed!busy_o=%b,en_i=%b",busy_o,en_i);
		rst_n = 1;
		#20;
		//item:2
		$display("\n---[Item 2]Testing Configuration---");
		if(busy_o===0)begin
			cfg_shape_m=4'd4;
			$display("Pass:Configuration sequence executed(Simulated).");
		end else begin
			$display("FAIL:Can not configure!DEvice is busy.");
		end
		#20;
		//Item 3
		$display("\n---[Item 3]Testing Startup Sequence---");
		en_i=1;
		#10
		if(busy_o===0)begin
		  $display("PASS:Waiting for data");
		  end else begin
		  $display("FALL:busy_o went High too early!");
		  end
		//send first data packet
		s_axis_tvalid_a_b=1;
		s_axis_tdata_a_b={512{1'b1}};
		s_axis_tvalid_c=1;
		#10
		//3 check busy
		if(busy_o===1)begin
		  $display("PASS:Data arrived,System Started.");
		  end else begin
		  $display("FAIL:Data arrived but busy_o is still low");
		end
		#100;
		$display("\n---All Tests Finished---");
		$finish;
		//(c)specific test item
		


		#500;
		$display("Simulation FInished");
		$finish;
	end
	// generate waveforn
	initial begin
	$fsdbDumpfile("../../wav/top/top.fsdb");
	$fsdbDumpvars(0,tb_tensor_core);
	end
endmodule		










	



















