# Julia wrapper for header: nvToolsExt.h
# Automatically generated using Clang.jl


function nvtxInitialize(reserved)
    ccall((:nvtxInitialize, libnvtx), Cvoid,
          (Ptr{Cvoid},),
          reserved)
end

function nvtxDomainMarkEx(domain, eventAttrib)
    ccall((:nvtxDomainMarkEx, libnvtx), Cvoid,
          (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
          domain, eventAttrib)
end

function nvtxMarkEx(eventAttrib)
    ccall((:nvtxMarkEx, libnvtx), Cvoid,
          (Ptr{nvtxEventAttributes_t},),
          eventAttrib)
end

function nvtxMarkA(message)
    ccall((:nvtxMarkA, libnvtx), Cvoid,
          (Cstring,),
          message)
end

function nvtxMarkW(message)
    ccall((:nvtxMarkW, libnvtx), Cvoid,
          (Ptr{Cwchar_t},),
          message)
end

function nvtxDomainRangeStartEx(domain, eventAttrib)
    ccall((:nvtxDomainRangeStartEx, libnvtx), nvtxRangeId_t,
          (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
          domain, eventAttrib)
end

function nvtxRangeStartEx(eventAttrib)
    ccall((:nvtxRangeStartEx, libnvtx), nvtxRangeId_t,
          (Ptr{nvtxEventAttributes_t},),
          eventAttrib)
end

function nvtxRangeStartA(message)
    ccall((:nvtxRangeStartA, libnvtx), nvtxRangeId_t,
          (Cstring,),
          message)
end

function nvtxRangeStartW(message)
    ccall((:nvtxRangeStartW, libnvtx), nvtxRangeId_t,
          (Ptr{Cwchar_t},),
          message)
end

function nvtxDomainRangeEnd(domain, id)
    ccall((:nvtxDomainRangeEnd, libnvtx), Cvoid,
          (nvtxDomainHandle_t, nvtxRangeId_t),
          domain, id)
end

function nvtxRangeEnd(id)
    ccall((:nvtxRangeEnd, libnvtx), Cvoid,
          (nvtxRangeId_t,),
          id)
end

function nvtxDomainRangePushEx(domain, eventAttrib)
    ccall((:nvtxDomainRangePushEx, libnvtx), Cint,
          (nvtxDomainHandle_t, Ptr{nvtxEventAttributes_t}),
          domain, eventAttrib)
end

function nvtxRangePushEx(eventAttrib)
    ccall((:nvtxRangePushEx, libnvtx), Cint,
          (Ptr{nvtxEventAttributes_t},),
          eventAttrib)
end

function nvtxRangePushA(message)
    ccall((:nvtxRangePushA, libnvtx), Cint,
          (Cstring,),
          message)
end

function nvtxRangePushW(message)
    ccall((:nvtxRangePushW, libnvtx), Cint,
          (Ptr{Cwchar_t},),
          message)
end

function nvtxDomainRangePop(domain)
    ccall((:nvtxDomainRangePop, libnvtx), Cint,
          (nvtxDomainHandle_t,),
          domain)
end

function nvtxRangePop()
    ccall((:nvtxRangePop, libnvtx), Cint, ())
end

function nvtxDomainResourceCreate(domain, attribs)
    ccall((:nvtxDomainResourceCreate, libnvtx), nvtxResourceHandle_t,
          (nvtxDomainHandle_t, Ptr{nvtxResourceAttributes_t}),
          domain, attribs)
end

function nvtxDomainResourceDestroy(resource)
    ccall((:nvtxDomainResourceDestroy, libnvtx), Cvoid,
          (nvtxResourceHandle_t,),
          resource)
end

function nvtxDomainNameCategoryA(domain, category, name)
    ccall((:nvtxDomainNameCategoryA, libnvtx), Cvoid,
          (nvtxDomainHandle_t, UInt32, Cstring),
          domain, category, name)
end

function nvtxDomainNameCategoryW(domain, category, name)
    ccall((:nvtxDomainNameCategoryW, libnvtx), Cvoid,
          (nvtxDomainHandle_t, UInt32, Ptr{Cwchar_t}),
          domain, category, name)
end

function nvtxNameCategoryA(category, name)
    ccall((:nvtxNameCategoryA, libnvtx), Cvoid,
          (UInt32, Cstring),
          category, name)
end

function nvtxNameCategoryW(category, name)
    ccall((:nvtxNameCategoryW, libnvtx), Cvoid,
          (UInt32, Ptr{Cwchar_t}),
          category, name)
end

function nvtxNameOsThreadA(threadId, name)
    ccall((:nvtxNameOsThreadA, libnvtx), Cvoid,
          (UInt32, Cstring),
          threadId, name)
end

function nvtxNameOsThreadW(threadId, name)
    ccall((:nvtxNameOsThreadW, libnvtx), Cvoid,
          (UInt32, Ptr{Cwchar_t}),
          threadId, name)
end

function nvtxDomainRegisterStringA(domain, string)
    ccall((:nvtxDomainRegisterStringA, libnvtx), nvtxStringHandle_t,
          (nvtxDomainHandle_t, Cstring),
          domain, string)
end

function nvtxDomainRegisterStringW(domain, string)
    ccall((:nvtxDomainRegisterStringW, libnvtx), nvtxStringHandle_t,
          (nvtxDomainHandle_t, Ptr{Cwchar_t}),
          domain, string)
end

function nvtxDomainCreateA(name)
    ccall((:nvtxDomainCreateA, libnvtx), nvtxDomainHandle_t,
          (Cstring,),
          name)
end

function nvtxDomainCreateW(name)
    ccall((:nvtxDomainCreateW, libnvtx), nvtxDomainHandle_t,
          (Ptr{Cwchar_t},),
          name)
end

function nvtxDomainDestroy(domain)
    ccall((:nvtxDomainDestroy, libnvtx), Cvoid,
          (nvtxDomainHandle_t,),
          domain)
end
# Julia wrapper for header: nvToolsExtCuda.h
# Automatically generated using Clang.jl


function nvtxNameCuDeviceA(device, name)
    ccall((:nvtxNameCuDeviceA, libnvtx), Cvoid,
          (CUdevice, Cstring),
          device, name)
end

function nvtxNameCuDeviceW(device, name)
    ccall((:nvtxNameCuDeviceW, libnvtx), Cvoid,
          (CUdevice, Ptr{Cwchar_t}),
          device, name)
end

function nvtxNameCuContextA(context, name)
    ccall((:nvtxNameCuContextA, libnvtx), Cvoid,
          (CUcontext, Cstring),
          context, name)
end

function nvtxNameCuContextW(context, name)
    ccall((:nvtxNameCuContextW, libnvtx), Cvoid,
          (CUcontext, Ptr{Cwchar_t}),
          context, name)
end

function nvtxNameCuStreamA(stream, name)
    ccall((:nvtxNameCuStreamA, libnvtx), Cvoid,
          (CUstream, Cstring),
          stream, name)
end

function nvtxNameCuStreamW(stream, name)
    ccall((:nvtxNameCuStreamW, libnvtx), Cvoid,
          (CUstream, Ptr{Cwchar_t}),
          stream, name)
end

function nvtxNameCuEventA(event, name)
    ccall((:nvtxNameCuEventA, libnvtx), Cvoid,
          (CUevent, Cstring),
          event, name)
end

function nvtxNameCuEventW(event, name)
    ccall((:nvtxNameCuEventW, libnvtx), Cvoid,
          (CUevent, Ptr{Cwchar_t}),
          event, name)
end
