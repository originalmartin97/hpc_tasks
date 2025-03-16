/*  =================================================================================================================
 * |                                                                                                                 |
 * |      The code demonstrates a basic point-to-point communication using MPI (Message Passing Interface)           |
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
        printf("\n");

        int number_of_processes,
            rank, // process ID; unique identifier to distinguish different processes running in parallel
            source, // specifies from which process the message will be received
            destination, // specifies to which process the message will be sent
            status_code, // return code, stores the value of MPI function calls
            number_of_elements=1,
            message_tag=1;
        char inbox, outbox='X';

        MPI_Request send_request, receive_request;
        MPI_Status send_status, receive_status;

        MPI_Init( &argc, &argv ); // Initializes the MPI environment
        MPI_Comm_size( MPI_COMM_WORLD, &number_of_processes ); // gets the total number of processes
        MPI_Comm_rank( MPI_COMM_WORLD, &rank ); // assigns a unique rank to each process within communicator

        {
                destination = 1 - rank;
                source = 1 - rank;

                printf("Process of rank %d:\n", rank);
                status_code = MPI_Isend(&outbox, number_of_elements, MPI_CHAR, destination, message_tag, MPI_COMM_WORLD, &send_request );
                printf("\t sending message '%c' to process of rank %d ...\n", outbox, destination);
                MPI_Wait( &send_request, &send_status );
                printf("\t DONE.\n\n");

                printf("Process of rank %d:\n", rank);
                status_code = MPI_Irecv(&inbox, number_of_elements, MPI_CHAR, source, message_tag,MPI_COMM_WORLD, &receive_request );
                printf("\t Receiving message...\n");
                MPI_Wait( &receive_request, &receive_status );
                printf("\t Received message '%c' from process of rank %d.\n", inbox, source);
                printf("\t DONE.\n\n");
        }

        MPI_Finalize(); // shuts down the MPI environmnet and cleans up resources;
        return 0;
}
