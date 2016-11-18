import numpy as np
import copy 

#Coroutine of a Ringbuffer

class Flow :
    def __init__(self,d,e,l):
        self.data = d
        self.energy = e
        self.label = l

def coroutine(func):
    """
    Decorator that allows to forget about the first call of a coroutine .next()
    method or .send(None)
    This call is done inside the decorator
    :param func: the coroutine to decorate
    """
    def start(*args,**kwargs):
        cr = func(*args,**kwargs)
        next(cr)
        return cr
    return start

#Test if the source equals None  
def defineOutput(source, beginning, end):
    output = None
    if(source == None):
        output = None
    else : 
        output = copy.deepcopy(source[beginning : end])
    return output


@coroutine
def ring_buffer(next, window, covering):
    """
    Ring buffer inside a coroutine that allows to bufferize received data
    Hand send it to next method when window size is reached. A covering size
    can be set to include this amount of the previous data with the next send.
    :param next: next coroutine to send data
    :param window: data size to send
    :param covering: data size sent with the next window
    """
    try:
        buffer_data = [None]*(window*10)
        buffer_energy = [None]*(window*10)
        buffer_label = [None]*(window*10)
        buf = Flow(None, None, None)
        write_index = 0
        read_index = 0
        data_size = 0 
        offset = window - covering
        while True :
            input = yield
            #print (len(input.energy))
            if input.data is None and input.energy is None and input.label is None :
                continue

            if (input.data != None):
                n = len(input.data)
            elif (input.energy != None):
                n = len(input.energy)
            elif (input.label != None):
                n = len(input.label)

            # add new data to buffer
            for j in range (0, n):

                if(input.energy == None):
                    buffer_energy = None
                else : 
                    buffer_energy[write_index] = input.energy[j]
                    m = len(buffer_energy)
   
                if(input.data == None):
                    buffer_data = None
                else : 
                    buffer_data[write_index] = input.data[j]
                    m = len(buffer_data)

                if(input.label == None):
                    buffer_label = None
                else : 
                    buffer_label[write_index] = input.label[j]
                    m = len(buffer_label)

                #update write_index
                write_index = (write_index + 1 ) % m
                # data size between indexes
                data_size =  write_index - read_index if read_index < write_index else m- read_index + write_index
                #test if a data window can be sent
                if data_size > window:
                    # send a window (testing the case we must concatenate the beginning and end of the buffer)
                    if (read_index < (read_index + window)%m):  
                        buf.data = defineOutput(input.data, read_index, read_index + window)
                        buf.energy = defineOutput(input.energy, read_index, read_index + window)
                        buf.label = defineOutput(input.label, read_index, read_index + window)                 
                        next.send(buf)

                    else:  
                        buf.data = defineOutput(input.data, read_index, m) + defineOutput(input.data, 0, window - m+ read_index)  
                        buf.energy = defineOutput(input.energy, read_index, m) + defineOutput(input.energy, 0, window - m+ read_index)  
                        buf.label = defineOutput(input.label, read_index, m) + defineOutput(input.label, 0, window - m+ read_index)
                        next.send(buf)

                    read_index = (read_index + offset) % m  

    except GeneratorExit:
        next.close()
