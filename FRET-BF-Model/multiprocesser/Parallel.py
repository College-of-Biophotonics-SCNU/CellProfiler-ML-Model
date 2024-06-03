import multiprocessing


class Parallel:
    def __init__(self, numProcs=None):
        cpus = multiprocessing.cpu_count()
        if numProcs is None or numProcs > cpus or numProcs < 1:
            numProcs = cpus
        self.pool = multiprocessing.Pool(numProcs)

    def compute(self, operation, data):
        return self.pool.apply(operation, data)

    def close(self):
        self.pool.close()
        self.pool.join()

