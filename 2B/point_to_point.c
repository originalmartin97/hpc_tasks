/*  =================================================================================================================
 * |                                                                                                                 |
 * |      The code demonstrates a basic point-to-point communication using MPI (Message Passing Interface)          |
 * |                                                                                                                 |
 * |-----------------------------------------------------------------------------------------------------------------|
 * |                                          [[ In Summary ]]                                                       |
 * |-----------------------------------------------------------------------------------------------------------------|
 * |This code demonstrates a simple 'ping-pong' communication pattern where two processes exchange a single          |
 * |character.                                                                                                       |
 * |It showcases the basic elements of MPI programming, including initialization, communication, finalization.       |
 * |-----------------------------------------------------------------------------------------------------------------|
 * |                                      [[ Usage and Execution ]]                                                  |
 * |-----------------------------------------------------------------------------------------------------------------|
 * | ~~ uses at least two processes (works with more but won't really do anything in this state)                     |
 * | Steps to use----------------------------------------------------------------------------------------------------|
 * | ~~ compile with mpicc                                                                                           |
 * | ~~ run with 'srun' command                                                                                      |
 * |    ~~ use flag '-n' for specifying number of processes                                                          |
 * | ~~ If ran correctly the program will give outpot for the two processes and beforehand states the given number   |
 * |    of processes for the communication size.                                                                     |
 * | ~~ DONE                                                                                                         |
 *  ================================================================================================================= */


#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>

int main( int argc, char *argv[] )
{
        int number_of_processes,
            rank, // process ID; unique identifier to distinguish different processes running in parallel
            source, // specifies from which process the message will be received
            destination, // specifies to which process the message will be sent
            status_code, // return code, stores the value of MPI function calls
            number_of_elements=1,
            message_tag=1;
        char inbox, outbox='X';
        MPI_Status Stat;

        MPI_Init( &argc, &argv ); // Initializes the MPI environment
        MPI_Comm_size( MPI_COMM_WORLD, &number_of_processes ); // gets the total number of processes
        //              (communicator/''chanel''/group of processes, integer pointer size)

        printf("Total number of processes: %d\n", number_of_processes);
        MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // assigns a unique rank to each process within communicator

        if ( rank == 0 ){
                destination = 1;
                source = 1;
                destination = 1;

                // MPI_Send(
                //      void *buffer - pointer to data being sent
                //      int counter - maximum size of the data (being sent)
                //      MPI_Datatype datatype - data type (e.g. MPI_CHAR, MPI_INT)
                //      int destinaton - rank of the destination process
                //      int tag - message tag for identification
                //      MPI_Comm communicator - MPI communicator (e.g. MPI_COMM_WORLD) [what group / ''chanel'' / communicator]
                // )

                printf("I am rank %d and I am sending %c to rank of %d\n", rank, outbox, destination);
                status_code = MPI_Send(&outbox, number_of_elements, MPI_CHAR, destination, message_tag, MPI_COMM_WORLD );

                // MPI_Recv(
                // void *buffer - pointer to the data being received
                // int counter - maximum number of elements
                // same
                // same
                // same
                // MPI_Status *status - structure that holds message details
                // )
                status_code = MPI_Recv(&inbox, number_of_elements, MPI_CHAR, source, message_tag, MPI_COMM_WORLD, &Stat );
                printf("I am rank %d and I have received %c from %d\n", rank, inbox, source);

        }

        else if ( rank == 1 ){
                destination = 0;
                source = 0;


                status_code = MPI_Recv( &inbox, number_of_elements, MPI_CHAR, source, message_tag, MPI_COMM_WORLD, &Stat );
                printf("I am rank %d and I have received %c from %d\n", rank, inbox, source);

                printf("I am rank %d and I am sending %c to rank of %d\n", rank, outbox, destination);
                status_code = MPI_Send( &outbox, number_of_elements, MPI_CHAR, destination, message_tag, MPI_COMM_WORLD );
        }

        MPI_Finalize(); // shuts down the MPI environmnet and cleans up resources;
        return 0;
}
