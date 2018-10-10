from nets import smallnet
from multiprocessing.context import Queue, Pool, Process, current_process

def worker(hyperparams, results):
    smallnet.run(hyperparams)
    t = tasks.get()
    result = t * 2
    results.put([current_process().name, t, result])

def main():
    n = 100
    myTasks = Queue()
    myResults = Queue()
    Workers = [Process(target=worker, args=(myTasks, myResults)) for i in range(n)]
    for each in Workers:
        each.start()

    for each in range(n):
        myTasks.put(each)

    while n:
        result = myResults.get()
        print("Res: %s" % result)
        n -= 1


if __name__ == '__main__':
    main()