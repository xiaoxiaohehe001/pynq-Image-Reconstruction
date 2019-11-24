#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

typedef float Dtype_f;
typedef float Dtype_w;
typedef float Dtype_mul;
typedef float Dtype_acc;
using namespace std;

void convtrans_pad(
		ap_uint<16> CHin, ap_uint<16> CHout,
		ap_uint<16> Hin,  ap_uint<16> Win,
		Dtype_f feature_in[], Dtype_f feature_out[]
)
{
	//规定形参类型
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_in offset=slave
	#pragma HLS INTERFACE s_axilite port=CHout
	#pragma HLS INTERFACE s_axilite port=Hin
	#pragma HLS INTERFACE s_axilite port=CHin
	#pragma HLS INTERFACE s_axilite port=Win
	#pragma HLS INTERFACE s_axilite port=return

	ap_uint<16> Hout,Wout;

	Hout=Hin*2+1;
	Wout=Win*2+1;

	for(int cout=0;cout<CHout;cout++)
		for(int i=0;i<Hout;i++)
			for(int j=0;j<Wout;j++)
			{
				feature_out[i*CHout*Wout+j*CHout+cout]=1;
			}

	for(int cout=0;cout<CHout;cout++)
		for(int i=0;i<Hout;i=i+2)
			for(int j=0;j<Wout;j++)
			{
				feature_out[i*CHout*Wout+j*CHout+cout]=0;
			}

	for(int cout=0;cout<CHout;cout++)
		for(int j=0;j<Wout;j=j+2)
			for(int i=0;i<Hout;i++)
			{
				feature_out[j*CHout*Hout+i*CHout+cout]=0;
			}

	for(int cout=0;cout<CHout;cout++)
		for(int i=1;i<Hout;i++)
			for(int j=1;j<Wout;j++)
			{
				if(feature_out[i*CHout*Wout+j*CHout+cout]==1)
				{
					feature_out[i*CHout*Wout+j*CHout+cout]=feature_in[((i-1)/2)*CHout*Win+((j-1)/2)*CHin+cout];
				}
			}
}
