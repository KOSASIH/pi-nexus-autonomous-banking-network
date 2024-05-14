import logging
import sys

def distributed_auditor(rank, size):
"""Create a distributed auditor that logs messages locally and sends them to the centralized auditor."""
    logger = logging.getLogger('distributed_auditor')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(rank)02d] %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info(f"Distributed auditor started on rank {rank}/{size}")

    while True:
        message = input("Enter a message: ")
        logger.info(f"Logging message: {message}")
        MPI.COMM_WORLD.send({'rank': rank, 'message': message}, dest=0)

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    distributed_auditor(rank, size)
