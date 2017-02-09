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
    if(source is None):
        output = None
    else :
        output = copy.deepcopy(source[beginning : end])
    #return np.array(output)
    return output


@coroutine
def ring_buffer(next, window, covering, sample_size=1):
    """
    Ring buffer inside a coroutine that allows to bufferize received data
    Hand send it to next method when window size is reached. A covering size
    can be set to include this amount of the previous data with the next send.
    :param next: next coroutine to send data
    :param window: data size to send
    :param covering: data size sent with the next window
    """
    try:
        bufsize = 10*window
        #buffer_data = [None]*bufsize
        buffer_data = np.empty((bufsize, sample_size), dtype='|O')
        #buffer_energy = [None]*bufsize
        buffer_energy = np.empty((bufsize), dtype='|O')
        #buffer_label = [None]*bufsize
        buffer_label = np.empty((bufsize), dtype='|O')
        buf = Flow(None, None, None)
        write_index = 0
        read_index = 0
        data_size = 0
        offset = window - covering
        while True :
            input = yield
            if input.data is None and input.energy is None and input.label is None :
                continue

            if (input.data is not None):
                n = input.data.shape[-1]
            elif (input.energy is not None):
                n = input.energy.shape[-1]
            elif (input.label is not None):
                n = input.label.shape[-1]

            # add new data to buffer
            for j in range (0, n):
                if(input.energy is None):
                    buffer_energy = None
                else :
                    buffer_energy[write_index] = input.energy[j]

                if(input.data is None):
                    buffer_data = None
                else :
                    buffer_data[write_index] = input.data[j]

                if(input.label is None):
                    buffer_label = None
                else :
                    buffer_label[write_index] = input.label[j]

                #update write_index
                write_index = (write_index + 1 ) % bufsize
                # data size between indexes
                data_size =  write_index - read_index if read_index < write_index else bufsize- read_index + write_index
                #test if a data window can be sent
                if data_size >= window:
                    # send a window (testing the case we must concatenate the beginning and end of the buffer)
                    if (read_index < (read_index + window-1)%bufsize):
                        buf.data = defineOutput(buffer_data, read_index, read_index + window)
                        buf.energy = defineOutput(buffer_energy, read_index, read_index + window)
                        buf.label = defineOutput(buffer_label, read_index, read_index + window)
                        next.send(buf)
                    else:
                        buf.data = np.vstack((defineOutput(buffer_data, read_index, bufsize),
                                              defineOutput(buffer_data, 0, window - bufsize + read_index)))
                        buf.energy = np.hstack((defineOutput(buffer_energy, read_index, bufsize),
                                                defineOutput(buffer_energy, 0, window - bufsize + read_index)))
                        buf.label = np.hstack((defineOutput(buffer_label, read_index, bufsize),
                                               defineOutput(buffer_label, 0, window - bufsize + read_index)))
                        next.send(buf)
                    read_index = (read_index + offset) % bufsize

    except GeneratorExit:
        next.close()
