#Coroutine of a Ringbuffer

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

@coroutine
def return_result(return_func):
    """
    Call the given function with what is received from previous coroutine as
    parameter
    :param return_func: return function
    """
    try:
        while True :
            input = yield
            return_result(input)
    except GeneratorExit:
        next.close()

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
        buffer = [None]*(window*10)
        write_index = 0
        read_index = 0
        data_size = 0
        offset = window - covering
        while True :
            input = yield
            print (len(input))
            if input is None:
                continue
            # add new data to buffer
            for j in range (0, len(input)):
                buffer[write_index] = input[j]
                #update write_index
                write_index = (write_index + 1 ) % len(buffer)
                # data size between indexes
                data_size =  write_index - read_index if read_index < write_index else len(buffer) - read_index + write_index
                #test if a data window can be sent
                if data_size > window:
                    # send a window (testing the case we must concatenate the beginning and end of the buffer)
                    if (read_index < (read_index + window)%len(buffer)):
                        next.send(buffer[read_index : read_index + window])
                    else:
                        next.send(buffer[read_index : len(buffer)] + buffer[0 : (window - len(buffer) + read_index)])
                    read_index = (read_index + offset) % len(buffer)

    except GeneratorExit:
        next.close()
