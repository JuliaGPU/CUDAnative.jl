# Julia wrapper for header: cupti_result.h
# Automatically generated using Clang.jl


function cuptiGetResultString(result, str)
    @check ccall((:cuptiGetResultString, libcupti), CUptiResult,
                 (CUptiResult, Ptr{Cstring}),
                 result, str)
end
# Julia wrapper for header: cupti_version.h
# Automatically generated using Clang.jl


function cuptiGetVersion(version)
    @check ccall((:cuptiGetVersion, libcupti), CUptiResult,
                 (Ptr{UInt32},),
                 version)
end
# Julia wrapper for header: cupti_activity.h
# Automatically generated using Clang.jl


function cuptiGetTimestamp(timestamp)
    @check ccall((:cuptiGetTimestamp, libcupti), CUptiResult,
                 (Ptr{UInt64},),
                 timestamp)
end

function cuptiGetContextId(context, contextId)
    @check ccall((:cuptiGetContextId, libcupti), CUptiResult,
                 (CUcontext, Ptr{UInt32}),
                 context, contextId)
end

function cuptiGetStreamId(context, stream, streamId)
    @check ccall((:cuptiGetStreamId, libcupti), CUptiResult,
                 (CUcontext, CUstream, Ptr{UInt32}),
                 context, stream, streamId)
end

function cuptiGetStreamIdEx(context, stream, perThreadStream, streamId)
    @check ccall((:cuptiGetStreamIdEx, libcupti), CUptiResult,
                 (CUcontext, CUstream, UInt8, Ptr{UInt32}),
                 context, stream, perThreadStream, streamId)
end

function cuptiGetDeviceId(context, deviceId)
    @check ccall((:cuptiGetDeviceId, libcupti), CUptiResult,
                 (CUcontext, Ptr{UInt32}),
                 context, deviceId)
end

function cuptiActivityEnable(kind)
    @check ccall((:cuptiActivityEnable, libcupti), CUptiResult,
                 (CUpti_ActivityKind,),
                 kind)
end

function cuptiActivityDisable(kind)
    @check ccall((:cuptiActivityDisable, libcupti), CUptiResult,
                 (CUpti_ActivityKind,),
                 kind)
end

function cuptiActivityEnableContext(context, kind)
    @check ccall((:cuptiActivityEnableContext, libcupti), CUptiResult,
                 (CUcontext, CUpti_ActivityKind),
                 context, kind)
end

function cuptiActivityDisableContext(context, kind)
    @check ccall((:cuptiActivityDisableContext, libcupti), CUptiResult,
                 (CUcontext, CUpti_ActivityKind),
                 context, kind)
end

function cuptiActivityGetNumDroppedRecords(context, streamId, dropped)
    @check ccall((:cuptiActivityGetNumDroppedRecords, libcupti), CUptiResult,
                 (CUcontext, UInt32, Ptr{Csize_t}),
                 context, streamId, dropped)
end

function cuptiActivityGetNextRecord(buffer, validBufferSizeBytes, record)
    @check ccall((:cuptiActivityGetNextRecord, libcupti), CUptiResult,
                 (Ptr{UInt8}, Csize_t, Ptr{Ptr{CUpti_Activity}}),
                 buffer, validBufferSizeBytes, record)
end

function cuptiActivityRegisterCallbacks(funcBufferRequested, funcBufferCompleted)
    @check ccall((:cuptiActivityRegisterCallbacks, libcupti), CUptiResult,
                 (CUpti_BuffersCallbackRequestFunc, CUpti_BuffersCallbackCompleteFunc),
                 funcBufferRequested, funcBufferCompleted)
end

function cuptiActivityFlush(context, streamId, flag)
    @check ccall((:cuptiActivityFlush, libcupti), CUptiResult,
                 (CUcontext, UInt32, UInt32),
                 context, streamId, flag)
end

function cuptiActivityFlushAll(flag)
    @check ccall((:cuptiActivityFlushAll, libcupti), CUptiResult,
                 (UInt32,),
                 flag)
end

function cuptiActivityGetAttribute(attr, valueSize, value)
    @check ccall((:cuptiActivityGetAttribute, libcupti), CUptiResult,
                 (CUpti_ActivityAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                 attr, valueSize, value)
end

function cuptiActivitySetAttribute(attr, valueSize, value)
    @check ccall((:cuptiActivitySetAttribute, libcupti), CUptiResult,
                 (CUpti_ActivityAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                 attr, valueSize, value)
end

function cuptiActivityConfigureUnifiedMemoryCounter(config, count)
    @check ccall((:cuptiActivityConfigureUnifiedMemoryCounter, libcupti), CUptiResult,
                 (Ptr{CUpti_ActivityUnifiedMemoryCounterConfig}, UInt32),
                 config, count)
end

function cuptiGetAutoBoostState(context, state)
    @check ccall((:cuptiGetAutoBoostState, libcupti), CUptiResult,
                 (CUcontext, Ptr{CUpti_ActivityAutoBoostState}),
                 context, state)
end

function cuptiActivityConfigurePCSampling(ctx, config)
    @check ccall((:cuptiActivityConfigurePCSampling, libcupti), CUptiResult,
                 (CUcontext, Ptr{CUpti_ActivityPCSamplingConfig}),
                 ctx, config)
end

function cuptiGetLastError()
    @check ccall((:cuptiGetLastError, libcupti), CUptiResult, ())
end

function cuptiSetThreadIdType(type)
    @check ccall((:cuptiSetThreadIdType, libcupti), CUptiResult,
                 (CUpti_ActivityThreadIdType,),
                 type)
end

function cuptiGetThreadIdType(type)
    @check ccall((:cuptiGetThreadIdType, libcupti), CUptiResult,
                 (Ptr{CUpti_ActivityThreadIdType},),
                 type)
end

function cuptiComputeCapabilitySupported(major, minor, support)
    @check ccall((:cuptiComputeCapabilitySupported, libcupti), CUptiResult,
                 (Cint, Cint, Ptr{Cint}),
                 major, minor, support)
end

function cuptiDeviceSupported(dev, support)
    @check ccall((:cuptiDeviceSupported, libcupti), CUptiResult,
                 (CUdevice, Ptr{Cint}),
                 dev, support)
end

function cuptiFinalize()
    @check ccall((:cuptiFinalize, libcupti), CUptiResult, ())
end

function cuptiActivityPushExternalCorrelationId(kind, id)
    @check ccall((:cuptiActivityPushExternalCorrelationId, libcupti), CUptiResult,
                 (CUpti_ExternalCorrelationKind, UInt64),
                 kind, id)
end

function cuptiActivityPopExternalCorrelationId(kind, lastId)
    @check ccall((:cuptiActivityPopExternalCorrelationId, libcupti), CUptiResult,
                 (CUpti_ExternalCorrelationKind, Ptr{UInt64}),
                 kind, lastId)
end

function cuptiActivityEnableLatencyTimestamps(enable)
    @check ccall((:cuptiActivityEnableLatencyTimestamps, libcupti), CUptiResult,
                 (UInt8,),
                 enable)
end
# Julia wrapper for header: cupti_callbacks.h
# Automatically generated using Clang.jl


function cuptiSupportedDomains(domainCount, domainTable)
    @check ccall((:cuptiSupportedDomains, libcupti), CUptiResult,
                 (Ptr{Csize_t}, Ptr{CUpti_DomainTable}),
                 domainCount, domainTable)
end

function cuptiSubscribe(subscriber, callback, userdata)
    @check ccall((:cuptiSubscribe, libcupti), CUptiResult,
                 (Ptr{CUpti_SubscriberHandle}, CUpti_CallbackFunc, Ptr{Cvoid}),
                 subscriber, callback, userdata)
end

function cuptiUnsubscribe(subscriber)
    @check ccall((:cuptiUnsubscribe, libcupti), CUptiResult,
                 (CUpti_SubscriberHandle,),
                 subscriber)
end

function cuptiGetCallbackState(enable, subscriber, domain, cbid)
    @check ccall((:cuptiGetCallbackState, libcupti), CUptiResult,
                 (Ptr{UInt32}, CUpti_SubscriberHandle, CUpti_CallbackDomain,
                  CUpti_CallbackId),
                 enable, subscriber, domain, cbid)
end

function cuptiEnableCallback(enable, subscriber, domain, cbid)
    @check ccall((:cuptiEnableCallback, libcupti), CUptiResult,
                 (UInt32, CUpti_SubscriberHandle, CUpti_CallbackDomain, CUpti_CallbackId),
                 enable, subscriber, domain, cbid)
end

function cuptiEnableDomain(enable, subscriber, domain)
    @check ccall((:cuptiEnableDomain, libcupti), CUptiResult,
                 (UInt32, CUpti_SubscriberHandle, CUpti_CallbackDomain),
                 enable, subscriber, domain)
end

function cuptiEnableAllDomains(enable, subscriber)
    @check ccall((:cuptiEnableAllDomains, libcupti), CUptiResult,
                 (UInt32, CUpti_SubscriberHandle),
                 enable, subscriber)
end

function cuptiGetCallbackName(domain, cbid, name)
    @check ccall((:cuptiGetCallbackName, libcupti), CUptiResult,
                 (CUpti_CallbackDomain, UInt32, Ptr{Cstring}),
                 domain, cbid, name)
end
# Julia wrapper for header: cupti_events.h
# Automatically generated using Clang.jl


function cuptiSetEventCollectionMode(context, mode)
    @check ccall((:cuptiSetEventCollectionMode, libcupti), CUptiResult,
                 (CUcontext, CUpti_EventCollectionMode),
                 context, mode)
end

function cuptiDeviceGetAttribute(device, attrib, valueSize, value)
    @check ccall((:cuptiDeviceGetAttribute, libcupti), CUptiResult,
                 (CUdevice, CUpti_DeviceAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                 device, attrib, valueSize, value)
end

function cuptiDeviceGetTimestamp(context, timestamp)
    @check ccall((:cuptiDeviceGetTimestamp, libcupti), CUptiResult,
                 (CUcontext, Ptr{UInt64}),
                 context, timestamp)
end

function cuptiDeviceGetNumEventDomains(device, numDomains)
    @check ccall((:cuptiDeviceGetNumEventDomains, libcupti), CUptiResult,
                 (CUdevice, Ptr{UInt32}),
                 device, numDomains)
end

function cuptiDeviceEnumEventDomains(device, arraySizeBytes, domainArray)
    @check ccall((:cuptiDeviceEnumEventDomains, libcupti), CUptiResult,
                 (CUdevice, Ptr{Csize_t}, Ptr{CUpti_EventDomainID}),
                 device, arraySizeBytes, domainArray)
end

function cuptiDeviceGetEventDomainAttribute(device, eventDomain, attrib, valueSize, value)
    @check ccall((:cuptiDeviceGetEventDomainAttribute, libcupti), CUptiResult,
                 (CUdevice, CUpti_EventDomainID, CUpti_EventDomainAttribute, Ptr{Csize_t},
                  Ptr{Cvoid}),
                 device, eventDomain, attrib, valueSize, value)
end

function cuptiGetNumEventDomains(numDomains)
    @check ccall((:cuptiGetNumEventDomains, libcupti), CUptiResult,
                 (Ptr{UInt32},),
                 numDomains)
end

function cuptiEnumEventDomains(arraySizeBytes, domainArray)
    @check ccall((:cuptiEnumEventDomains, libcupti), CUptiResult,
                 (Ptr{Csize_t}, Ptr{CUpti_EventDomainID}),
                 arraySizeBytes, domainArray)
end

function cuptiEventDomainGetAttribute(eventDomain, attrib, valueSize, value)
    @check ccall((:cuptiEventDomainGetAttribute, libcupti), CUptiResult,
                 (CUpti_EventDomainID, CUpti_EventDomainAttribute, Ptr{Csize_t},
                  Ptr{Cvoid}),
                 eventDomain, attrib, valueSize, value)
end

function cuptiEventDomainGetNumEvents(eventDomain, numEvents)
    @check ccall((:cuptiEventDomainGetNumEvents, libcupti), CUptiResult,
                 (CUpti_EventDomainID, Ptr{UInt32}),
                 eventDomain, numEvents)
end

function cuptiEventDomainEnumEvents(eventDomain, arraySizeBytes, eventArray)
    @check ccall((:cuptiEventDomainEnumEvents, libcupti), CUptiResult,
                 (CUpti_EventDomainID, Ptr{Csize_t}, Ptr{CUpti_EventID}),
                 eventDomain, arraySizeBytes, eventArray)
end

function cuptiEventGetAttribute(event, attrib, valueSize, value)
    @check ccall((:cuptiEventGetAttribute, libcupti), CUptiResult,
                 (CUpti_EventID, CUpti_EventAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                 event, attrib, valueSize, value)
end

function cuptiEventGetIdFromName(device, eventName, event)
    @check ccall((:cuptiEventGetIdFromName, libcupti), CUptiResult,
                 (CUdevice, Cstring, Ptr{CUpti_EventID}),
                 device, eventName, event)
end

function cuptiEventGroupCreate(context, eventGroup, flags)
    @check ccall((:cuptiEventGroupCreate, libcupti), CUptiResult,
                 (CUcontext, Ptr{CUpti_EventGroup}, UInt32),
                 context, eventGroup, flags)
end

function cuptiEventGroupDestroy(eventGroup)
    @check ccall((:cuptiEventGroupDestroy, libcupti), CUptiResult,
                 (CUpti_EventGroup,),
                 eventGroup)
end

function cuptiEventGroupGetAttribute(eventGroup, attrib, valueSize, value)
    @check ccall((:cuptiEventGroupGetAttribute, libcupti), CUptiResult,
                 (CUpti_EventGroup, CUpti_EventGroupAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                 eventGroup, attrib, valueSize, value)
end

function cuptiEventGroupSetAttribute(eventGroup, attrib, valueSize, value)
    @check ccall((:cuptiEventGroupSetAttribute, libcupti), CUptiResult,
                 (CUpti_EventGroup, CUpti_EventGroupAttribute, Csize_t, Ptr{Cvoid}),
                 eventGroup, attrib, valueSize, value)
end

function cuptiEventGroupAddEvent(eventGroup, event)
    @check ccall((:cuptiEventGroupAddEvent, libcupti), CUptiResult,
                 (CUpti_EventGroup, CUpti_EventID),
                 eventGroup, event)
end

function cuptiEventGroupRemoveEvent(eventGroup, event)
    @check ccall((:cuptiEventGroupRemoveEvent, libcupti), CUptiResult,
                 (CUpti_EventGroup, CUpti_EventID),
                 eventGroup, event)
end

function cuptiEventGroupRemoveAllEvents(eventGroup)
    @check ccall((:cuptiEventGroupRemoveAllEvents, libcupti), CUptiResult,
                 (CUpti_EventGroup,),
                 eventGroup)
end

function cuptiEventGroupResetAllEvents(eventGroup)
    @check ccall((:cuptiEventGroupResetAllEvents, libcupti), CUptiResult,
                 (CUpti_EventGroup,),
                 eventGroup)
end

function cuptiEventGroupEnable(eventGroup)
    @check ccall((:cuptiEventGroupEnable, libcupti), CUptiResult,
                 (CUpti_EventGroup,),
                 eventGroup)
end

function cuptiEventGroupDisable(eventGroup)
    @check ccall((:cuptiEventGroupDisable, libcupti), CUptiResult,
                 (CUpti_EventGroup,),
                 eventGroup)
end

function cuptiEventGroupReadEvent(eventGroup, flags, event, eventValueBufferSizeBytes,
                                  eventValueBuffer)
    @check ccall((:cuptiEventGroupReadEvent, libcupti), CUptiResult,
                 (CUpti_EventGroup, CUpti_ReadEventFlags, CUpti_EventID, Ptr{Csize_t},
                  Ptr{UInt64}),
                 eventGroup, flags, event, eventValueBufferSizeBytes, eventValueBuffer)
end

function cuptiEventGroupReadAllEvents(eventGroup, flags, eventValueBufferSizeBytes,
                                      eventValueBuffer, eventIdArraySizeBytes,
                                      eventIdArray, numEventIdsRead)
    @check ccall((:cuptiEventGroupReadAllEvents, libcupti), CUptiResult,
                 (CUpti_EventGroup, CUpti_ReadEventFlags, Ptr{Csize_t}, Ptr{UInt64},
                  Ptr{Csize_t}, Ptr{CUpti_EventID}, Ptr{Csize_t}),
                 eventGroup, flags, eventValueBufferSizeBytes, eventValueBuffer,
                 eventIdArraySizeBytes, eventIdArray, numEventIdsRead)
end

function cuptiEventGroupSetsCreate(context, eventIdArraySizeBytes, eventIdArray,
                                   eventGroupPasses)
    @check ccall((:cuptiEventGroupSetsCreate, libcupti), CUptiResult,
                 (CUcontext, Csize_t, Ptr{CUpti_EventID}, Ptr{Ptr{CUpti_EventGroupSets}}),
                 context, eventIdArraySizeBytes, eventIdArray, eventGroupPasses)
end

function cuptiEventGroupSetsDestroy(eventGroupSets)
    @check ccall((:cuptiEventGroupSetsDestroy, libcupti), CUptiResult,
                 (Ptr{CUpti_EventGroupSets},),
                 eventGroupSets)
end

function cuptiEventGroupSetEnable(eventGroupSet)
    @check ccall((:cuptiEventGroupSetEnable, libcupti), CUptiResult,
                 (Ptr{CUpti_EventGroupSet},),
                 eventGroupSet)
end

function cuptiEventGroupSetDisable(eventGroupSet)
    @check ccall((:cuptiEventGroupSetDisable, libcupti), CUptiResult,
                 (Ptr{CUpti_EventGroupSet},),
                 eventGroupSet)
end

function cuptiEnableKernelReplayMode(context)
    @check ccall((:cuptiEnableKernelReplayMode, libcupti), CUptiResult,
                 (CUcontext,),
                 context)
end

function cuptiDisableKernelReplayMode(context)
    @check ccall((:cuptiDisableKernelReplayMode, libcupti), CUptiResult,
                 (CUcontext,),
                 context)
end

function cuptiKernelReplaySubscribeUpdate(updateFunc, customData)
    @check ccall((:cuptiKernelReplaySubscribeUpdate, libcupti), CUptiResult,
                 (CUpti_KernelReplayUpdateFunc, Ptr{Cvoid}),
                 updateFunc, customData)
end
# Julia wrapper for header: cupti_metrics.h
# Automatically generated using Clang.jl


function cuptiGetNumMetrics(numMetrics)
    @check ccall((:cuptiGetNumMetrics, libcupti), CUptiResult,
                 (Ptr{UInt32},),
                 numMetrics)
end

function cuptiEnumMetrics(arraySizeBytes, metricArray)
    @check ccall((:cuptiEnumMetrics, libcupti), CUptiResult,
                 (Ptr{Csize_t}, Ptr{CUpti_MetricID}),
                 arraySizeBytes, metricArray)
end

function cuptiDeviceGetNumMetrics(device, numMetrics)
    @check ccall((:cuptiDeviceGetNumMetrics, libcupti), CUptiResult,
                 (CUdevice, Ptr{UInt32}),
                 device, numMetrics)
end

function cuptiDeviceEnumMetrics(device, arraySizeBytes, metricArray)
    @check ccall((:cuptiDeviceEnumMetrics, libcupti), CUptiResult,
                 (CUdevice, Ptr{Csize_t}, Ptr{CUpti_MetricID}),
                 device, arraySizeBytes, metricArray)
end

function cuptiMetricGetAttribute(metric, attrib, valueSize, value)
    @check ccall((:cuptiMetricGetAttribute, libcupti), CUptiResult,
                 (CUpti_MetricID, CUpti_MetricAttribute, Ptr{Csize_t}, Ptr{Cvoid}),
                 metric, attrib, valueSize, value)
end

function cuptiMetricGetIdFromName(device, metricName, metric)
    @check ccall((:cuptiMetricGetIdFromName, libcupti), CUptiResult,
                 (CUdevice, Cstring, Ptr{CUpti_MetricID}),
                 device, metricName, metric)
end

function cuptiMetricGetNumEvents(metric, numEvents)
    @check ccall((:cuptiMetricGetNumEvents, libcupti), CUptiResult,
                 (CUpti_MetricID, Ptr{UInt32}),
                 metric, numEvents)
end

function cuptiMetricEnumEvents(metric, eventIdArraySizeBytes, eventIdArray)
    @check ccall((:cuptiMetricEnumEvents, libcupti), CUptiResult,
                 (CUpti_MetricID, Ptr{Csize_t}, Ptr{Cint}),
                 metric, eventIdArraySizeBytes, eventIdArray)
end

function cuptiMetricGetNumProperties(metric, numProp)
    @check ccall((:cuptiMetricGetNumProperties, libcupti), CUptiResult,
                 (CUpti_MetricID, Ptr{UInt32}),
                 metric, numProp)
end

function cuptiMetricEnumProperties(metric, propIdArraySizeBytes, propIdArray)
    @check ccall((:cuptiMetricEnumProperties, libcupti), CUptiResult,
                 (CUpti_MetricID, Ptr{Csize_t}, Ptr{CUpti_MetricPropertyID}),
                 metric, propIdArraySizeBytes, propIdArray)
end

function cuptiMetricGetRequiredEventGroupSets(context, metric, eventGroupSets)
    @check ccall((:cuptiMetricGetRequiredEventGroupSets, libcupti), CUptiResult,
                 (CUcontext, CUpti_MetricID, Ptr{Ptr{Cint}}),
                 context, metric, eventGroupSets)
end

function cuptiMetricCreateEventGroupSets(context, metricIdArraySizeBytes, metricIdArray,
                                         eventGroupPasses)
    @check ccall((:cuptiMetricCreateEventGroupSets, libcupti), CUptiResult,
                 (CUcontext, Csize_t, Ptr{CUpti_MetricID}, Ptr{Ptr{Cint}}),
                 context, metricIdArraySizeBytes, metricIdArray, eventGroupPasses)
end

function cuptiMetricGetValue(device, metric, eventIdArraySizeBytes, eventIdArray,
                             eventValueArraySizeBytes, eventValueArray, timeDuration,
                             metricValue)
    @check ccall((:cuptiMetricGetValue, libcupti), CUptiResult,
                 (CUdevice, CUpti_MetricID, Csize_t, Ptr{Cint}, Csize_t, Ptr{UInt64},
                  UInt64, Ptr{CUpti_MetricValue}),
                 device, metric, eventIdArraySizeBytes, eventIdArray,
                 eventValueArraySizeBytes, eventValueArray, timeDuration, metricValue)
end

function cuptiMetricGetValue2(metric, eventIdArraySizeBytes, eventIdArray,
                              eventValueArraySizeBytes, eventValueArray,
                              propIdArraySizeBytes, propIdArray, propValueArraySizeBytes,
                              propValueArray, metricValue)
    @check ccall((:cuptiMetricGetValue2, libcupti), CUptiResult,
                 (CUpti_MetricID, Csize_t, Ptr{Cint}, Csize_t, Ptr{UInt64}, Csize_t,
                  Ptr{CUpti_MetricPropertyID}, Csize_t, Ptr{UInt64}, Ptr{CUpti_MetricValue}),
                 metric, eventIdArraySizeBytes, eventIdArray, eventValueArraySizeBytes,
                 eventValueArray, propIdArraySizeBytes, propIdArray,
                 propValueArraySizeBytes, propValueArray, metricValue)
end
# Julia wrapper for header: cupti_profiler_target.h
# Automatically generated using Clang.jl


function cuptiProfilerInitialize(pParams)
    @check ccall((:cuptiProfilerInitialize, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_Initialize_Params},),
                 pParams)
end

function cuptiProfilerDeInitialize(pParams)
    @check ccall((:cuptiProfilerDeInitialize, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_DeInitialize_Params},),
                 pParams)
end

function cuptiProfilerCounterDataImageCalculateSize(pParams)
    @check ccall((:cuptiProfilerCounterDataImageCalculateSize, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_CounterDataImage_CalculateSize_Params},),
                 pParams)
end

function cuptiProfilerCounterDataImageInitialize(pParams)
    @check ccall((:cuptiProfilerCounterDataImageInitialize, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_CounterDataImage_Initialize_Params},),
                 pParams)
end

function cuptiProfilerCounterDataImageCalculateScratchBufferSize(pParams)
    @check ccall((:cuptiProfilerCounterDataImageCalculateScratchBufferSize, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_CounterDataImage_CalculateScratchBufferSize_Params},),
                 pParams)
end

function cuptiProfilerCounterDataImageInitializeScratchBuffer(pParams)
    @check ccall((:cuptiProfilerCounterDataImageInitializeScratchBuffer, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_CounterDataImage_InitializeScratchBuffer_Params},),
                 pParams)
end

function cuptiProfilerBeginSession(pParams)
    @check ccall((:cuptiProfilerBeginSession, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_BeginSession_Params},),
                 pParams)
end

function cuptiProfilerEndSession(pParams)
    @check ccall((:cuptiProfilerEndSession, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_EndSession_Params},),
                 pParams)
end

function cuptiProfilerSetConfig(pParams)
    @check ccall((:cuptiProfilerSetConfig, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_SetConfig_Params},),
                 pParams)
end

function cuptiProfilerUnsetConfig(pParams)
    @check ccall((:cuptiProfilerUnsetConfig, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_UnsetConfig_Params},),
                 pParams)
end

function cuptiProfilerBeginPass(pParams)
    @check ccall((:cuptiProfilerBeginPass, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_BeginPass_Params},),
                 pParams)
end

function cuptiProfilerEndPass(pParams)
    @check ccall((:cuptiProfilerEndPass, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_EndPass_Params},),
                 pParams)
end

function cuptiProfilerEnableProfiling(pParams)
    @check ccall((:cuptiProfilerEnableProfiling, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_EnableProfiling_Params},),
                 pParams)
end

function cuptiProfilerDisableProfiling(pParams)
    @check ccall((:cuptiProfilerDisableProfiling, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_DisableProfiling_Params},),
                 pParams)
end

function cuptiProfilerIsPassCollected(pParams)
    @check ccall((:cuptiProfilerIsPassCollected, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_IsPassCollected_Params},),
                 pParams)
end

function cuptiProfilerFlushCounterData(pParams)
    @check ccall((:cuptiProfilerFlushCounterData, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_FlushCounterData_Params},),
                 pParams)
end

function cuptiProfilerPushRange(pParams)
    @check ccall((:cuptiProfilerPushRange, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_PushRange_Params},),
                 pParams)
end

function cuptiProfilerPopRange(pParams)
    @check ccall((:cuptiProfilerPopRange, libcupti), CUptiResult,
                 (Ptr{CUpti_Profiler_PopRange_Params},),
                 pParams)
end
