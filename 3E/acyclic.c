#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

#define MAX_DEPENDENCIES 10
#define MAX_PROCESSES 100

typedef struct
{
	int id;
	int dependency_count;
	int dependent_count;
	int dependents[MAX_PROCESSES];
} Workflow;

Workflow workflow[MAX_PROCESSES];
int process_count;

int read_workflow(const char *filename)
{
	FILE *file = fopen(filename, "r");
	if (!file)
	{
		printf("Bad filename\n");
		return -1;
	}

	char line[25];
	while (fgets(line, sizeof(line), file))
	{ // fgets reads a line from the file parameters are the buffer, the size of the buffer and the file
		if (line[0] == '\n')
			continue;

		int from, to;
		sscanf(line, "%d --> %d", &from, &to); // sscanf reads formatted input from a string parameters are the string, the format and the variables to store the values

		if (process_count <= from)
			process_count = from + 1;
		if (process_count <= to)
			process_count = to + 1; // Track the highest process ID

		workflow[from].id = from;                                         // Track the process IDs
		workflow[to].id = to;                                             // Track the process IDs
		workflow[from].dependents[workflow[from].dependent_count++] = to; // Track dependents
		workflow[to].dependency_count++;                                  // Track dependencies
	}

	fclose(file);
	return 0;
}

void broadcast_workflow(int rank)
{
	MPI_Bcast(&process_count, 1, MPI_INT, 0, MPI_COMM_WORLD);                           // Broadcast the number of processes
	MPI_Bcast(workflow, sizeof(Workflow) * MAX_PROCESSES, MPI_BYTE, 0, MPI_COMM_WORLD); // Broadcast the workflow data
}

void execute_workflow(int rank)
{
	int dependency_count = workflow[rank].dependency_count;
	int received_count = 0;
	int token[MAX_PROCESSES] = {0};
	int token_count = 0;
	int signal = 0;
	MPI_Request requests[3]; // Three irecvs: signal, token, token_count
	MPI_Status status;

	// Initialize non-blocking receives
	MPI_Irecv(&signal, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &requests[0]);
	MPI_Irecv(token, MAX_PROCESSES, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requests[1]);
	MPI_Irecv(&token_count, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &requests[2]);

	// Wait for all dependencies to send their signals
	while (received_count < dependency_count)
	{
		int index;
		MPI_Waitany(3, requests, &index, &status);// Waits for any one of the three non-blocking requests to complete.
							  // `index` will store which request finished (0, 1, or 2).
							  // `status` contains details about the completed request (e.g., source, tag).

		if (index == 0) // Signal received
		{
			received_count++;
			printf("PROCESS %d: Received signal from PROCESS %d\n", rank, status.MPI_SOURCE);
			// Repost Irecv for future signals
			MPI_Irecv(&signal, 1, MPI_INT, MPI_ANY_SOURCE, 2, MPI_COMM_WORLD, &requests[0]);
		}
		else if (index == 1) // Token received
		{
			printf("PROCESS %d: Received token from PROCESS %d: ", rank, status.MPI_SOURCE);
			for (int i = 0; i < token_count; i++)
				printf("%d ", token[i]);
			printf("\n");

			// Check for cyclic dependencies
			printf("PROCESS %d: Checking for cyclic dependencies...\n", rank);
			for (int i = 0; i < token_count; i++)
			{
				if (token[i] == rank)
				{ // Cycle detected
					printf("🚨 Cyclic dependency detected! PROCESS %d found itself in the path: ", rank);
					for (int j = 0; j < token_count; j++)
						printf("%d ", token[j]);
					printf("\n");
					MPI_Abort(MPI_COMM_WORLD, 1);
					return;
				}
			}

			// Repost Irecv for future tokens
			MPI_Irecv(token, MAX_PROCESSES, MPI_INT, MPI_ANY_SOURCE, 0, MPI_COMM_WORLD, &requests[1]);
		}
		else if (index == 2) // Token count received
		{
			// Repost Irecv for future token counts
			MPI_Irecv(&token_count, 1, MPI_INT, MPI_ANY_SOURCE, 1, MPI_COMM_WORLD, &requests[2]);
		}

		// Append own rank to the token
		token[token_count] = rank;
		token_count++;

		// Forward the token to all dependent processes only if dependencies are not yet met
		if (received_count < dependency_count)
		{
			for (int i = 0; i < workflow[rank].dependent_count; i++)
			{
				int dependent_rank = workflow[rank].dependents[i];
				printf("PROCESS %d: Forwarding token to PROCESS %d: ", rank, dependent_rank);
				for (int j = 0; j < token_count; j++)
					printf("%d ", token[j]);
				printf("\n");
				MPI_Send(token, token_count, MPI_INT, dependent_rank, 0, MPI_COMM_WORLD);
				MPI_Send(&token_count, 1, MPI_INT, dependent_rank, 1, MPI_COMM_WORLD);
			}
		}
	}

	// All dependencies met, execute work
	printf("PROCESS %d: All dependencies met, executing work\n", rank);

	// Notify dependent processes that work is done
	for (int i = 0; i < workflow[rank].dependent_count; i++)
	{
		int dependent_rank = workflow[rank].dependents[i];
		printf("PROCESS %d: Notifying PROCESS %d that work is done\n", rank, dependent_rank);
		MPI_Send(&rank, 1, MPI_INT, dependent_rank, 2, MPI_COMM_WORLD); // Send a signal to dependent processes
	}

	MPI_Barrier(MPI_COMM_WORLD); // Synchronize all processes before finishing
}

/**
 * Cleans up the workflow data.
 */
void cleanup_workflow()
{
	// No dynamic memory to free in this simplified version
	process_count = 0;
}

int main(int argc, char **argv)
{

	MPI_Init(&argc, &argv);

	int rank, size;
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);
	MPI_Comm_size(MPI_COMM_WORLD, &size);

	if (rank == 0)
	{
		if (read_workflow("workflow.txt") != 0)
		{
			MPI_Finalize();
			return -1;
		}

		if (process_count > size)
		{
			printf("ERROR: The workflow requires %d processes, but only %d are available.\n", process_count, size);
			MPI_Abort(MPI_COMM_WORLD, 1);
			return -1;
		}
	}

	broadcast_workflow(rank);

	// Ensure all processes have the correct workflow data
	MPI_Barrier(MPI_COMM_WORLD);

	// Execute the workflow
	execute_workflow(rank);

	cleanup_workflow();

	MPI_Finalize();
	return 0;
}
