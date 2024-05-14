import logging
import sys

def centralized_auditor(rank, size):
    """Create a centralized auditor that logs messages from all ranks."""
    logger = logging.getLogger('centralized_auditor')
    logger.setLevel(logging.DEBUG)
    formatter = logging.Formatter('%(asctime)s [%(rank)02d] %(message)s')
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    logger.info(f"Centralized auditor started on rank {rank}/{size}")

    while True:
        message = MPI.COMM_WORLD.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG)
        logger.info(f"Received message from rank {message['rank']}: {message['message']}")

if __name__ == "__main__":
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    centralized_auditor(rank, size)
