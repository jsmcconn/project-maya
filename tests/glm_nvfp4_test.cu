// Original NVFP4 blocks + distinct ModelOpt tensor scales against ggml's independent
// dequantizer and double-precision dots. Exercise CPU, GPU decode and FP16 prefill.
#include "strata/kernels/glm_fast.hpp"
#include "strata/kernels/glm_batch.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "ggml.h"
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
namespace gf = strata::kernels::glmf;
namespace gb = strata::kernels::glmb;
namespace kc = strata::kernels::cpu;
namespace mq = strata::prefill::mmq;
static void ck(cudaError_t e) { if(e != cudaSuccess) { std::fprintf(stderr,"CUDA: %s\n",cudaGetErrorString(e)); std::exit(1); } }
static void req(bool b, const char* s) { if(!b) { std::fprintf(stderr,"FAIL: %s\n",s); std::exit(1); } }
template<class T> static T* dev(const T* p, size_t n) { T* q; ck(cudaMalloc(&q,n*sizeof(T))); if(p) ck(cudaMemcpy(q,p,n*sizeof(T),cudaMemcpyHostToDevice)); return q; }
template<class T> static std::vector<T> read(const T* p, size_t n) { std::vector<T> h(n); ck(cudaMemcpy(h.data(),p,n*sizeof(T),cudaMemcpyDeviceToHost)); return h; }
static std::vector<float> q8read(const uint8_t* p,int n) {
    auto b=read(p,(size_t)n/32*36); std::vector<float> f(n);
    for(int i=0;i<n;++i) { uint16_t h; std::memcpy(&h,b.data()+(i/32)*36,2); f[i]=ggml_fp16_to_fp32(h)*(int8_t)b[(i/32)*36+4+i%32]; } return f;
}
static float silu(float g,float u) { g=std::min(g,.7f); u=std::clamp(u,-.7f,.7f); return g/(1.f+std::exp(-g))*u; }
static std::vector<float> product(const float* w,const float* x,int rows,int cols,float scale) {
    std::vector<float> y(rows); for(int r=0;r<rows;++r) { double v=0; for(int c=0;c<cols;++c) v+=(double)w[r*cols+c]*x[c]; y[r]=(float)v*scale; } return y;
}
static void near(const std::vector<float>& a,const std::vector<float>& b,float tol,const char* name) {
    double worst=0,ref=0; req(a.size()==b.size(),"shape");
    for(size_t i=0;i<a.size();++i) { req(std::isfinite(a[i]),name); worst=std::max(worst,(double)std::abs(a[i]-b[i])); ref=std::max(ref,(double)std::abs(b[i])); }
    std::printf("%s max_abs %.9g ref %.9g\n",name,worst,ref); req(worst<=tol*(1+ref),name);
}
static void run(int gpu) {
    ck(cudaSetDevice(gpu)); constexpr int E=256,F=128; std::string err; kc::NativeFmt nf;
    req(kc::native_fmt(40,40,E,F,nf,err),err.c_str());
    const size_t raw=nf.bytes-12; std::vector<uint8_t> b(nf.bytes); uint32_t seed=17;
    for(size_t i=0;i<raw;i+=36) {
        const uint8_t scales[]={0,1,7,8,24,40,56,63};
        for(int j=0;j<4;++j) b[i+j]=scales[(i/36+j)%8];
        for(int j=4;j<36;++j) { seed=1664525*seed+1013904223; b[i+j]=(uint8_t)(seed>>24); }
    }
    std::vector<float> w(raw/36*64); ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float(b.data(),w.data(),w.size());
    std::vector<float> x(E); for(int i=0;i<E;++i) x[i]=std::sin(i*.31f)*.8f;
    float* dx=dev(x.data(),x.size()); uint8_t* xq=dev<uint8_t>(nullptr,E/32*36);
    gf::quantize_q8_1(dx,xq,E,nullptr); auto xgpu=q8read(xq,E);
    for(const auto& scales: {std::vector<float>{.25f,1.5f,.0625f},std::vector<float>{1,1,1},std::vector<float>{0,1,0}}) {
        std::memcpy(b.data()+raw,scales.data(),12); const auto sf=nf.scaled(b.data());
        std::vector<uint8_t> aq(nf.act_bytes),hq(nf.h_bytes); std::vector<float> xc(E),hc(F),yc(E);
        kc::native_quant_act(nf,x.data(),aq.data()); ggml_get_type_traits((ggml_type)nf.gu_act)->to_float(aq.data(),xc.data(),E);
        const void* ap=aq.data(); float* hp=hc.data(); kc::native_gu_rows_split(sf,b.data(),b.data()+nf.up_off,&ap,1,&hp,0,F,.7f);
        auto g=product(w.data(),xc.data(),F,E,scales[0]),u=product(w.data()+F*E,xc.data(),F,E,scales[1]);
        std::vector<float> hr(F); for(int i=0;i<F;++i) hr[i]=silu(g[i],u[i]); near(hc,hr,2e-5f,"CPU GU scale before clamp");
        kc::native_quant_h(nf,hc.data(),hq.data()); std::vector<float> hdec(F); ggml_get_type_traits((ggml_type)nf.d_act)->to_float(hq.data(),hdec.data(),F);
        const void* hqp=hq.data(); float* yp=yc.data(); kc::native_down_rows_split(sf,b.data()+nf.down_off,&hqp,1,&yp,0,E);
        near(yc,product(w.data()+2*F*E,hdec.data(),E,F,scales[2]),2e-5f,"CPU down scale");
        uint8_t* db=dev(b.data(),b.size()); unsigned long long ptrs[8]={(unsigned long long)db}; float weights[8]={.75f}; int zero=0;
        gf::MoeDev m; m.plan_ptr=dev(ptrs,8); m.plan_w=dev(weights,8); m.cpu_flag=dev(&zero,1);
        auto dh=dev<uint8_t>(nullptr,F/32*36*8); auto dy=dev<float>(nullptr,E);
        gf::moe_gate_up(40,m,1,E,F,.7f,xq,dh,nullptr,0,nullptr,0,nullptr,nullptr);
        auto hg=q8read(dh,F); g=product(w.data(),xgpu.data(),F,E,scales[0]); u=product(w.data()+F*E,xgpu.data(),F,E,scales[1]);
        for(int i=0;i<F;++i) hr[i]=silu(g[i],u[i]); near(hg,hr,.008f,"GPU GU including Q8 rounding");
        gf::moe_down(40,m,1,E,F,nf.down_off,dh,nullptr,dy,nullptr);
        near(read(dy,E),product(w.data()+2*F*E,hg.data(),E,F,scales[2]*.75f),2e-5f,"GPU NVFP4 down dot");
        // Production prefill operations: exact block -> FP16, FP32 accumulating GEMM,
        // tensor scales before SwiGLU, FP16 H, down GEMM then down tensor scale.
        req(mq::to_f16_supported(40),"NVFP4 FP16 converter");
        auto dw=dev<uint16_t>(nullptr,w.size()); mq::to_f16(40,db,dw,w.size(),nullptr);
        std::vector<uint16_t> x16(E); for(int i=0;i<E;++i) x16[i]=ggml_fp32_to_fp16(x[i]);
        auto d16=dev(x16.data(),x16.size()); auto gu=dev<float>(nullptr,2*F); auto hh=dev<float>(nullptr,F);
        cublasHandle_t handle; req(cublasCreate(&handle)==CUBLAS_STATUS_SUCCESS,"cublas create"); float one=1,zz=0;
        req(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,2*F,1,E,&one,dw,CUDA_R_16F,E,d16,CUDA_R_16F,E,&zz,gu,CUDA_R_32F,2*F,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)==CUBLAS_STATUS_SUCCESS,"GU gemm");
        auto ds=(const float*)(db+raw); gb::nvfp4_scale_product(gu,2*F,2*F,ds,true,nullptr); gb::swiglu_rows(gu,hh,1,F,.7f,nullptr);
        auto h=read(hh,F); std::vector<float> xr(E); for(int i=0;i<E;++i) xr[i]=ggml_fp16_to_fp32(x16[i]);
        g=product(w.data(),xr.data(),F,E,scales[0]); u=product(w.data()+F*E,xr.data(),F,E,scales[1]); for(int i=0;i<F;++i) hr[i]=silu(g[i],u[i]); near(h,hr,2e-5f,"prefill GU scale before clamp");
        std::vector<uint16_t> h16(F); for(int i=0;i<F;++i) { h16[i]=ggml_fp32_to_fp16(h[i]); h[i]=ggml_fp16_to_fp32(h16[i]); } auto hd=dev(h16.data(),h16.size());
        req(cublasGemmEx(handle,CUBLAS_OP_T,CUBLAS_OP_N,E,1,F,&one,dw+2*F*E,CUDA_R_16F,F,hd,CUDA_R_16F,F,&zz,dy,CUDA_R_32F,E,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP)==CUBLAS_STATUS_SUCCESS,"down gemm");
        gb::nvfp4_scale_product(dy,E,E,ds,false,nullptr); near(read(dy,E),product(w.data()+2*F*E,h.data(),E,F,scales[2]),2e-5f,"prefill down scale");
        cublasDestroy(handle); for(void* p:{(void*)db,(void*)m.plan_ptr,(void*)m.plan_w,(void*)m.cpu_flag,(void*)dh,(void*)dy,(void*)dw,(void*)d16,(void*)gu,(void*)hh,(void*)hd}) ck(cudaFree(p));
    }
    // Float weight fallback must consume the existing Q8_1 activations without
    // dereferencing a missing F32 input; cover both batching entry points.
    for(int type:{gf::kTypeF32,gf::kTypeBF16}) for(int cols:{1,3,16}) {
        constexpr int R=19; std::vector<float> wf(E*R); for(int i=0;i<E*R;++i) wf[i]=std::cos(i*.77f)*.03f;
        std::vector<uint16_t> bf(E*R); for(int i=0;i<E*R;++i) { bf[i]=ggml_fp32_to_bf16(wf[i]).bits; if(type==gf::kTypeBF16) { uint32_t bits=(uint32_t)bf[i]<<16; std::memcpy(&wf[i],&bits,4); } }
        void* ww=type==gf::kTypeF32?(void*)dev(wf.data(),wf.size()):(void*)dev(bf.data(),bf.size());
        std::vector<uint8_t> acts((size_t)cols*E/32*36); auto single=read(xq,E/32*36); for(int t=0;t<cols;++t) std::memcpy(acts.data()+t*single.size(),single.data(),single.size());
        auto da=dev(acts.data(),acts.size()); auto out=dev<float>(nullptr,R*cols); gf::MvJob j; j.w=ww;j.xq=da;j.type=type;j.n_in=E;j.n_out=R;j.y=out;j.ncol=cols;j.xq_cstride=single.size();j.y_cstride=R;
        auto ref=product(wf.data(),xgpu.data(),R,E,1); std::vector<float> refs; for(int t=0;t<cols;++t) refs.insert(refs.end(),ref.begin(),ref.end());
        req(gf::mv(&j,1,nullptr),"float Q8 mv"); near(read(out,R*cols),refs,2e-5f,"float Q8 columns"); j.ncol=1;
        req(gf::mv_nt(&j,1,cols,nullptr),"float Q8 mv_nt"); near(read(out,R*cols),refs,2e-5f,"float Q8 mv_nt");
        ck(cudaFree(ww));ck(cudaFree(da));ck(cudaFree(out));
    }
    ck(cudaFree(dx));ck(cudaFree(xq)); ck(cudaDeviceSynchronize()); req(gf::launch_errors()==0,"launch errors"); std::printf("GPU %d NVFP4 PASS\n",gpu);
}

static void bf16_expert(int gpu) {
    ck(cudaSetDevice(gpu)); constexpr int E=256,F=128; kc::NativeFmt nf; std::string err;
    req(kc::native_fmt(30,30,E,F,nf,err),err.c_str());
    std::vector<uint16_t> bits(3*F*E);std::vector<float> w(bits.size());
    for(size_t i=0;i<w.size();++i) { bits[i]=ggml_fp32_to_bf16(std::cos(i*.17f)*.05f).bits; uint32_t b=(uint32_t)bits[i]<<16;std::memcpy(&w[i],&b,4); }
    auto db=dev(bits.data(),bits.size());std::vector<float>x(E);for(int i=0;i<E;++i)x[i]=std::sin(i*.21f)*.6f;
    auto dx=dev(x.data(),x.size());auto dq=dev<uint8_t>(nullptr,E/32*36);gf::quantize_q8_1(dx,dq,E,nullptr);auto xd=q8read(dq,E);
    unsigned long long pointers[8]={(unsigned long long)db};float weights[8]={.75f};int zero=0;
    gf::MoeDev m;m.plan_ptr=dev(pointers,8);m.plan_w=dev(weights,8);m.cpu_flag=dev(&zero,1);
    auto hq=dev<uint8_t>(nullptr,F/32*36*8);auto out=dev<float>(nullptr,E);
    gf::moe_gate_up(30,m,1,E,F,.7f,dq,hq,nullptr,0,nullptr,0,nullptr,nullptr);auto h=q8read(hq,F);
    auto g=product(w.data(),xd.data(),F,E,1),u=product(w.data()+F*E,xd.data(),F,E,1);std::vector<float> ref(F);
    for(int i=0;i<F;++i)ref[i]=silu(g[i],u[i]);near(h,ref,.008f,"BF16 MTP GPU GU Q8 rounding");
    gf::moe_down(30,m,1,E,F,nf.down_off,hq,nullptr,out,nullptr);near(read(out,E),product(w.data()+2*F*E,h.data(),E,F,.75f),2e-5f,"BF16 MTP GPU down");
    std::vector<uint8_t> aq(nf.act_bytes),cq(nf.h_bytes);std::vector<float> xc(E),hc(F),yc(E),hd(F);
    kc::native_quant_act(nf,x.data(),aq.data());ggml_get_type_traits((ggml_type)nf.gu_act)->to_float(aq.data(),xc.data(),E);
    const void* ap=aq.data();float* hp=hc.data();auto blob=(const uint8_t*)bits.data();kc::native_gu_rows_split(nf,blob,blob+nf.up_off,&ap,1,&hp,0,F,.7f);
    g=product(w.data(),xc.data(),F,E,1);u=product(w.data()+F*E,xc.data(),F,E,1);for(int i=0;i<F;++i)ref[i]=silu(g[i],u[i]);near(hc,ref,2e-5f,"BF16 MTP CPU GU");
    kc::native_quant_h(nf,hc.data(),cq.data());ggml_get_type_traits((ggml_type)nf.d_act)->to_float(cq.data(),hd.data(),F);
    const void* cp=cq.data();float* yp=yc.data();kc::native_down_rows_split(nf,blob+nf.down_off,&cp,1,&yp,0,E);near(yc,product(w.data()+2*F*E,hd.data(),E,F,1),2e-5f,"BF16 MTP CPU down");
    req(mq::to_f16_supported(30),"BF16 MTP prefill converter");auto d16=dev<uint16_t>(nullptr,bits.size());mq::to_f16(30,db,d16,bits.size(),nullptr);auto actual=read(d16,bits.size());for(size_t i=0;i<actual.size();++i)req(actual[i]==ggml_fp32_to_fp16(w[i]),"BF16 MTP prefill exact conversion");
    for(void* p:{(void*)db,(void*)dx,(void*)dq,(void*)m.plan_ptr,(void*)m.plan_w,(void*)m.cpu_flag,(void*)hq,(void*)out,(void*)d16})ck(cudaFree(p));
    ck(cudaDeviceSynchronize());req(gf::launch_errors()==0,"BF16 MTP launches");std::printf("GPU %d BF16 MTP expert PASS\n",gpu);
}

static void window_case(int gpu) {
    ck(cudaSetDevice(gpu));constexpr int E=256,F=128;kc::NativeFmt nf;std::string err;req(kc::native_fmt(40,40,E,F,nf,err),err.c_str());
    std::vector<uint8_t> b(nf.bytes);uint32_t seed=29;uint8_t* experts[2];
    for(int e=0;e<2;++e) {
        for(size_t i=0;i<nf.bytes-12;i+=36) { for(int j=0;j<4;++j)b[i+j]=24+(i/36+j)%8;for(int j=4;j<36;++j){seed=1664525*seed+1013904223;b[i+j]=seed>>24;} }
        float scales[3]={.25f*(e+1),.75f*(e+1),.0625f*(e+1)};std::memcpy(b.data()+nf.bytes-12,scales,12);experts[e]=dev(b.data(),b.size());
    }
    for(int T:{2,5,16}) {
        std::vector<float> x(T*E),weights(T*8),shared(T*E);std::vector<unsigned long long> pointers(T*8);std::vector<int> flags(T,0);
        for(int t=0;t<T;++t) { for(int i=0;i<E;++i){x[t*E+i]=std::sin(t*.37f+i*.21f)*.8f;shared[t*E+i]=std::cos(t*.31f+i*.19f)*.001f;}pointers[t*8]=(unsigned long long)experts[t%2];pointers[t*8+1]=t%3==2?0:(unsigned long long)experts[1-t%2];weights[t*8]=.65f;weights[t*8+1]=.35f; }
        auto dx=dev(x.data(),x.size());auto xq=dev<uint8_t>(nullptr,T*E/32*36);for(int t=0;t<T;++t)gf::quantize_q8_1(dx+t*E,xq+t*E/32*36,E,nullptr);
        auto pp=dev(pointers.data(),pointers.size());auto pw=dev(weights.data(),weights.size());auto cf=dev(flags.data(),flags.size());auto sh=dev(shared.data(),shared.size());auto hq=dev<uint8_t>(nullptr,T*8*F/32*36);auto ref=dev<float>(nullptr,T*E);auto out=dev<float>(nullptr,T*E);
        for(int t=0;t<T;++t) {gf::MoeDev md;md.plan_ptr=pp+t*8;md.plan_w=pw+t*8;md.cpu_flag=cf+t;gf::moe_gate_up(40,md,2,E,F,.7f,xq+t*E/32*36,hq+t*8*F/32*36,nullptr,0,nullptr,0,nullptr,nullptr);gf::moe_down(40,md,2,E,F,nf.down_off,hq+t*8*F/32*36,sh+t*E,ref+t*E,nullptr);}
        const auto reference=read(ref,T*E);
        gf::MoeWin mw;mw.plan_ptr=pp;mw.plan_w=pw;mw.cpu_flag=cf;mw.xq=xq;mw.xq_stride=E/32*36;mw.hq=hq;mw.part=dev<float>(nullptr,T*8*E);mw.sh_out=sh;mw.out=out;mw.out_stride=E;mw.grp_ptr=dev<unsigned long long>(nullptr,T*8);mw.grp_start=dev<int>(nullptr,T*8+1);mw.grp_ent=dev<int>(nullptr,T*8);mw.n_groups=dev<int>(nullptr,1);mw.T=T;mw.k=2;mw.n_embd=E;mw.n_ff=F;mw.limit=.7f;mw.down_off=nf.down_off;
        for(bool par:{false,true}) { req(gf::moe_win_experts(40,40,mw,nullptr,par),"NVFP4 batched expert dispatch");const auto actual=read(out,T*E);req(std::memcmp(actual.data(),reference.data(),actual.size()*4)==0,"NVFP4 grouped experts must match token kernels bit for bit"); }
        for(void* p:{(void*)dx,(void*)xq,(void*)pp,(void*)pw,(void*)cf,(void*)sh,(void*)hq,(void*)ref,(void*)out,(void*)mw.part,(void*)mw.grp_ptr,(void*)mw.grp_start,(void*)mw.grp_ent,(void*)mw.n_groups})ck(cudaFree(p));
    }
    for(auto p:experts)ck(cudaFree(p));ck(cudaDeviceSynchronize());req(gf::launch_errors()==0,"window launch errors");std::printf("GPU %d NVFP4 grouped window bit-exact PASS\n",gpu);
}
int main(){
    if(kc::cpu_avx2_ok() && !std::getenv("STRATA_NO_IQ256") &&
       (!std::getenv("STRATA_GLM_CPU_MT") || std::atoi(std::getenv("STRATA_GLM_CPU_MT")) != 0))
        req(kc::native_multi_rows_exact(40),"NVFP4 multi-token rows must match ggml exactly");
    int n=0; ck(cudaGetDeviceCount(&n));req(n>0,"GPU required");for(int i=0;i<n;++i){run(i);bf16_expert(i);window_case(i);}
}
