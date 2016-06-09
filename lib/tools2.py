import time

class Timer(object):
    def __init__(self):
        
        self.start_time = None
        self.time1 = None
        self.time2 = None
        self.time3 = None
        self.train = []
        self.comm = []
        
    def start(self):
        self.start_time=time.time()
        
    def end(self, mode):
        
        duration = time.time()-self.start_time
        
        if mode=='train':
            self.train.append(duration)
        elif mode=='comm':
            self.comm.append(duration)
        else:
            raise NotImplementedError
            
    def result(self):
        
        train = sum(self.train)
        comm = sum(self.comm)
        
        return (train, comm)
    
    def reset(self):
        
        self.train[:] = []
        self.comm[:] = []
        
        
def queue_reduce(value, config):
    
    'reduce value to rank0'
    
    reduced = None
    rank = config['rank']
    
    if rank == 1:
        config['queue_gpu_1to0'].put(value)
        
    if rank == 0:
        
        value_1 = config['queue_gpu_1to0'].get()
        
        sum_0 = value_1 + value
        
    if rank == 3:
        
        config['queue_gpu_3to2'].put(value)
        
    if rank == 2:
        
        value_3 = config['queue_gpu_3to2'].get()
        
        sum_2 = value_3 + value
        
    sync('reduce_step1', config)
    
    if rank==2:
        
        config['queue_gpu_2to0'].put(sum_2)
        
    if rank==0:
        
        sum_2 = config['queue_gpu_2to0'].get()
        
        reduced = sum_0 + sum_2
    
    
    return reduced
    
def sync(message, config):
    
    '''
    4GPU synchronzing based on queue 
    
    '''
    
    rank = config['rank']
    
    if rank == 0:
        config['queue_gpu_0to1'].put(message)
        config['queue_gpu_0to2'].put(message)
        assert config['queue_gpu_1to0'].get() == message
        assert config['queue_gpu_2to0'].get() == message
    elif rank == 1:
        config['queue_gpu_1to0'].put(message)
        assert config['queue_gpu_0to1'].get() == message
    elif rank == 2:
        config['queue_gpu_2to3'].put(message)
        config['queue_gpu_2to0'].put(message)
        assert config['queue_gpu_3to2'].get() == message
        assert config['queue_gpu_0to2'].get() == message
    elif rank == 3:
        config['queue_gpu_3to2'].put(message)
        assert config['queue_gpu_2to3'].get() == message
        
        
def prepare_copper(total_params, theano, drv):

    param_ga_list = []
    param_other_list = []
    param_ga_other_list = []
    h_list = []
    shape_list = []
    dtype_list = []

    for param in total_params:

        param_other = theano.shared(param.get_value(),borrow=False)

        param_other_list.append(param_other)
    
        param_ga_other = \
            theano.misc.pycuda_utils.to_gpuarray(
                param_other.container.value)

        param_ga_other_list.append(param_ga_other)
    
        param_ga = \
            theano.misc.pycuda_utils.to_gpuarray(param.container.value)
        param_ga_list.append(param_ga)
    
        h_list.append(drv.mem_get_ipc_handle(param_ga.ptr))
        shape_list.append(param_ga.shape)
        dtype_list.append(param_ga.dtype)

    vecadd_fun = theano.function([], \
                        updates=[(param, param + param_other) \
                        for param, param_other in zip(total_params, param_other_list)])
    division_factor = 1.0 / ranksize   
    average_fun = theano.function([], \
                        updates=[(param, param * division_factor) \
                        for param in total_params])
                    
    copy_fun = theano.function([], \
                        updates=[(param, param_other) \
                        for param, param_other in zip(total_params, param_other_list)])
                        
    return (
            param_ga_other_list,
            h_list,
            shape_list,
            dtype_list,
            vecadd_fun,
            average_fun,
            copy_fun
            )
            
def prepare_1to3(total_params,theano, drv):
    
    param_ga_list = []
    param_other1_list = []
    param_other2_list = []
    param_other3_list = []    
    param_other_lists = []

    param_ga_other1_list = []
    param_ga_other2_list = []
    param_ga_other3_list = []    
    param_ga_other_lists = []

    h_list = []
    shape_list = []
    dtype_list = []

    for param in total_params:

        param_other1 = theano.shared(param.get_value())
        param_other2 = theano.shared(param.get_value())
        param_other3 = theano.shared(param.get_value()) 
        param_other1_list.append(param_other1)
        param_other2_list.append(param_other2)
        param_other3_list.append(param_other3)
    
        param_ga_other1 = theano.misc.pycuda_utils.to_gpuarray(param_other1.container.value)
        param_ga_other2 = theano.misc.pycuda_utils.to_gpuarray(param_other2.container.value)
        param_ga_other3 = theano.misc.pycuda_utils.to_gpuarray(param_other3.container.value)        
        param_ga_other1_list.append(param_ga_other1)
        param_ga_other2_list.append(param_ga_other2)
        param_ga_other3_list.append(param_ga_other3)                        

        average_fun = theano.function([], updates=[(param,
                                          (param + param_other1 + param_other2 + param_other3) / 4.)])
    
        param_ga = theano.misc.pycuda_utils.to_gpuarray(param.container.value)
        param_ga_list.append(param_ga)
        shape_list.append(param_ga.shape)
        dtype_list.append(param_ga.dtype)
    
        h = drv.mem_get_ipc_handle(param_ga.ptr)
        h_list.append(h)

    param_ga_other_lists.append(param_ga_other1_list) 
    param_ga_other_lists.append(param_ga_other2_list)
    param_ga_other_lists.append(param_ga_other3_list)
    
    average_fun_test = theano.function([], \
                        updates=[(param, (param + param_other1 + param_other2 + param_other3) / 4.) \
                        for param, param_other1, param_other2, param_other3 in \
                        zip(total_params, param_other1_list, param_other2_list, param_other3_list)])
                        
    return ( 
            param_ga_other_lists,
            h_list,
            shape_list,
            dtype_list,
            average_fun_test
            )
                     
def prepare_both(total_params,theano, drv):
    
    # copper
    param_ga_list = []
    param_other_list = []
    param_ga_other_list = []
    h_list = []
    shape_list = []
    dtype_list = []

    for param in total_params:

        param_other = theano.shared(param.get_value(),borrow=False)

        param_other_list.append(param_other)

        param_ga_other = \
            theano.misc.pycuda_utils.to_gpuarray(
                param_other.container.value)

        param_ga_other_list.append(param_ga_other)

        param_ga = \
            theano.misc.pycuda_utils.to_gpuarray(param.container.value)
        param_ga_list.append(param_ga)

        h_list.append(drv.mem_get_ipc_handle(param_ga.ptr))
        shape_list.append(param_ga.shape)
        dtype_list.append(param_ga.dtype)

    vecadd_fun = theano.function([], \
                        updates=[(param, param + param_other) \
                        for param, param_other in zip(total_params, param_other_list)])
    division_factor = 1.0 / ranksize   
    division_fun = theano.function([], \
                        updates=[(param, param * division_factor) \
                        for param in total_params])
                
    copy_fun = theano.function([], \
                        updates=[(param, param_other) \
                        for param, param_other in zip(total_params, param_other_list)])
                    
    # 1to3     
    param_test_list = []               
    param_other1_list = []
    param_other2_list = []
    param_other3_list = []

    param_ga_other1_list = []
    param_ga_other2_list = []
    param_ga_other3_list = []    
    param_ga_other_lists = []

    for param in total_params:
    
        param_test = theano.shared(param.get_value())
        param_test_list.append(param_test)

        param_other1 = theano.shared(param.get_value())
        param_other2 = theano.shared(param.get_value())
        param_other3 = theano.shared(param.get_value()) 
        param_other1_list.append(param_other1)
        param_other2_list.append(param_other2)
        param_other3_list.append(param_other3)

        param_ga_other1 = theano.misc.pycuda_utils.to_gpuarray(param_other1.container.value)
        param_ga_other2 = theano.misc.pycuda_utils.to_gpuarray(param_other2.container.value)
        param_ga_other3 = theano.misc.pycuda_utils.to_gpuarray(param_other3.container.value)        
        param_ga_other1_list.append(param_ga_other1)
        param_ga_other2_list.append(param_ga_other2)
        param_ga_other3_list.append(param_ga_other3)
    
    param_ga_other_lists.append(param_ga_other1_list) 
    param_ga_other_lists.append(param_ga_other2_list)
    param_ga_other_lists.append(param_ga_other3_list)

    average_fun_test = theano.function([], \
                        #updates=[(param_test, (param + param_other1 + param_other2 + param_other3) / 4.) \
                        updates=[(param_test, (param + param_other1 + param_other2 + param_other3)/4.) \
                        for param, param_test, param_other1, param_other2, param_other3 in \
                        zip(total_params, param_test_list, param_other1_list, param_other2_list, param_other3_list)])
                        
    return (param_ga_other_list,
            param_ga_other_lists,
            h_list,
            shape_list,
            dtype_list,
            average_fun_test,
            vecadd_fun,
            average_fun,
            copy_fun
            )