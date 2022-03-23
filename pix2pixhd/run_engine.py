import os
import sys
from random import randint
import numpy as np
import tensorrt

try:
    from PIL import Image
    import pycuda.driver as cuda
    import pycuda.gpuarray as gpuarray
    import pycuda.autoinit
    import argparse
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({})
Please make sure you have pycuda and the example dependencies installed.
https://wiki.tiker.net/PyCuda/Installation/Linux
pip(3) install tensorrt[examples]
""".format(err))
    exit(1)

try:
    import tensorrt as trt
    from tensorrt.parsers import caffeparser
    from tensorrt.parsers import onnxparser    
except ImportError as err:
    sys.stderr.write("""ERROR: failed to import module ({})
Please make sure you have the TensorRT Library installed
and accessible in your LD_LIBRARY_PATH
""".format(err))
    exit(1)


G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

class Profiler(trt.infer.Profiler):
    """
    Example Implimentation of a Profiler
    Is identical to the Profiler class in trt.infer so it is possible
    to just use that instead of implementing this if further
    functionality is not needed
    """
    def __init__(self, timing_iter):
        trt.infer.Profiler.__init__(self)
        self.timing_iterations = timing_iter
        self.profile = []

    def report_layer_time(self, layerName, ms):
        record = next((r for r in self.profile if r[0] == layerName), (None, None))
        if record == (None, None):
            self.profile.append((layerName, ms))
        else:
            self.profile[self.profile.index(record)] = (record[0], record[1] + ms)

    def print_layer_times(self):
        totalTime = 0
        for i in range(len(self.profile)):
            print("{:40.40} {:4.3f}ms".format(self.profile[i][0], self.profile[i][1] / self.timing_iterations))
            totalTime += self.profile[i][1]
        print("Time over all layers: {:4.2f} ms per iteration".format(totalTime / self.timing_iterations))


def get_input_output_names(trt_engine):
    nbindings = trt_engine.get_nb_bindings();
    maps = []

    for b in range(0, nbindings):
        dims = trt_engine.get_binding_dimensions(b).to_DimsCHW()
        name = trt_engine.get_binding_name(b)
        type = trt_engine.get_binding_data_type(b)
        
        if (trt_engine.binding_is_input(b)):
            maps.append(name)
            print("Found input: ", name)
        else:
            maps.append(name)
            print("Found output: ", name)

        print("shape=" + str(dims.C()) + " , " + str(dims.H()) + " , " + str(dims.W()))
        print("dtype=" + str(type))
    return maps

def create_memory(engine, name,  buf, mem, batchsize, inp, inp_idx):
    binding_idx = engine.get_binding_index(name)
    if binding_idx == -1:
        raise AttributeError("Not a valid binding")
    print("Binding: name={}, bindingIndex={}".format(name, str(binding_idx)))
    dims = engine.get_binding_dimensions(binding_idx).to_DimsCHW()
    eltCount = dims.C() * dims.H() * dims.W() * batchsize

    if engine.binding_is_input(binding_idx):
        h_mem = inp[inp_idx]
        inp_idx = inp_idx + 1
    else:
        h_mem = np.random.uniform(0.0, 255.0, eltCount).astype(np.dtype('f4'))

    d_mem = cuda.mem_alloc(eltCount * 4)
    cuda.memcpy_htod(d_mem, h_mem)
    buf.insert(binding_idx, int(d_mem))
    mem.append(d_mem)
    return inp_idx


#Run inference on device
def time_inference(engine, batch_size, inp):
    bindings = []
    mem = []
    inp_idx = 0
    for io in get_input_output_names(engine):
        inp_idx = create_memory(engine, io,  bindings, mem,
                                batch_size, inp, inp_idx)

    context = engine.create_execution_context()
    g_prof = Profiler(500)
    context.set_profiler(g_prof)
    for i in range(iter):
        context.execute(batch_size, bindings)
    g_prof.print_layer_times()
    
    context.destroy() 
    return


def convert_to_datatype(v):
    if v==8:
        return trt.infer.DataType.INT8
    elif v==16:
        return trt.infer.DataType.HALF
    elif v==32:
        return trt.infer.DataType.FLOAT
    else:
        print("ERROR: Invalid model data type bit depth: " + str(v))
        return trt.infer.DataType.INT8

def run_trt_engine(engine_file, bs, it):
    engine = trt.utils.load_engine(G_LOGGER, engine_file)
    time_inference(engine, bs, it)

def run_onnx(onnx_file, data_type, bs, inp):
    # Create onnx_config
    apex = onnxparser.create_onnxconfig()
    apex.set_model_file_name(onnx_file)
    apex.set_model_dtype(convert_to_datatype(data_type))

     # create parser
    trt_parser = onnxparser.create_onnxparser(apex)
    assert(trt_parser)
    data_type = apex.get_model_dtype()
    onnx_filename = apex.get_model_file_name()
    trt_parser.parse(onnx_filename, data_type)
    trt_parser.report_parsing_info()
    trt_parser.convert_to_trtnetwork()
    trt_network = trt_parser.get_trtnetwork()
    assert(trt_network)

    # create infer builder
    trt_builder = trt.infer.create_infer_builder(G_LOGGER)
    trt_builder.set_max_batch_size(max_batch_size)
    trt_builder.set_max_workspace_size(max_workspace_size)
    
    if (apex.get_model_dtype() == trt.infer.DataType_kHALF):
        print("-------------------  Running FP16 -----------------------------")
        trt_builder.set_half2_mode(True)
    elif (apex.get_model_dtype() == trt.infer.DataType_kINT8): 
        print("-------------------  Running INT8 -----------------------------")
        trt_builder.set_int8_mode(True)
    else:
        print("-------------------  Running FP32 -----------------------------")
        
    print("----- Builder is Done -----")
    print("----- Creating Engine -----")
    trt_engine = trt_builder.build_cuda_engine(trt_network)
    print("----- Engine is built -----")
    time_inference(engine, bs, inp)
