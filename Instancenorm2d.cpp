#include <ap_int.h>
#include <ap_fixed.h>
#include <math.h>

typedef float Dtype_f;
typedef float Dtype_w;
typedef float Dtype_mul;
typedef float Dtype_acc;
using namespace std;

void Instancenorm2d(
		ap_uint<16> CHin, ap_uint<16> CHout,
		ap_uint<16> Hin,  ap_uint<16> Win,
        ap_uint<1> relu_en, float gamma, float beta,
		Dtype_f feature_in[], Dtype_f feature_out[]
)   //relu_en为1，执行InstanceNorm之后再执行relu_en
{
	//规定形参类型
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_out offset=slave
	#pragma HLS INTERFACE m_axi depth=4294967295 port=feature_in offset=slave
	#pragma HLS INTERFACE s_axilite port=relu_en
	#pragma HLS INTERFACE s_axilite port=CHout
	#pragma HLS INTERFACE s_axilite port=Hin
	#pragma HLS INTERFACE s_axilite port=CHin
	#pragma HLS INTERFACE s_axilite port=Win
	#pragma HLS INTERFACE s_axilite port=gamma
	#pragma HLS INTERFACE s_axilite port=beta
	#pragma HLS INTERFACE s_axilite port=return

	ap_uint<16> Hout,Wout;
    float mean=0;
    float var=0;
    float eps=1e-5;
    Dtype_f feature_out_wait;
    Dtype_mul  result=0;
    Dtype_acc sum=0;
    Dtype_f sum_fangcha=0;
	Wout=Win;
	Hout=Hin;

	for(int cout=0;cout<CHout;cout++)
		for(int i=0;i<Hout;i++)
			for(int j=0;j<Wout;j++)
			{
				sum+=feature_in[i*CHout*Win+j*CHout+cout];
			}
    mean=sum/(Hout*Wout*CHout);

	for(int cout=0;cout<CHout;cout++)
		for(int i=0;i<Hout;i++)
			for(int j=0;j<Wout;j++)
			{
				sum_fangcha+=(feature_in[i*CHout*Win+j*CHout+cout]-mean)*(feature_in[i*CHout*Win+j*CHout+cout]-mean);
			}
    var=sum_fangcha/(Hout*Wout*CHout);
	for(int cout=0;cout<CHout;cout++)
		for(int i=0;i<Hout;i++)
			for(int j=0;j<Wout;j++)
			{
				feature_out_wait=(feature_in[i*CHout*Win+j*CHout+cout]-mean)/pow(var+eps,0.5);
				result=feature_out_wait*gamma+beta;
				if(relu_en && (result<0))
				{
					feature_out[i*Wout*CHout+j*CHout+cout]=0;
				}
				else
				{
					feature_out[i*Wout*CHout+j*CHout+cout]=result;
				}
			}
}
