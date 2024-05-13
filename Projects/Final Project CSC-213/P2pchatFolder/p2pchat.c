#include <pthread.h>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#include "socket.h"
#include "ui.h"
#include "message.h"

// Keep the username in a global so we can access it from the callback
const char *username;
int *sockets;
int size = 10;
int clients = 0;

void *server_listen(void *args);
void *server_init(void *args);

/**
  Loop through all sockets in global variable and send the message to all of them.
  If a message was recieved from another peer, use skip_fd to avoid sending it back
  to the same peer.

  @param  message: the message to send
  @param  skip_fd: a socket to skip
*/
void send_message_to_peers(char *message, int skip_fd)
{
  for (int i = 0; i < clients; i++)
  {
    if (sockets[i] == skip_fd)
      continue;
    // Ignore error messages from sending a message because
    // bad peers are kicked from the global array when a recieve_message fails
    send_message(sockets[i], message);
  }
}

/**
  Add a new socket file descriptor to the global varibable and spawn a new thread
  to listen for new messages

  @param  socket_fd: the new peer connection
*/
void accept_new_peer(int socket_fd)
{
  sockets[clients] = socket_fd;
  clients++;
  pthread_t client_thread;
  int ret = pthread_create(&client_thread, NULL, server_listen, &(sockets[clients - 1]));
  if (ret != 0)
  {
    perror("Could not create server to listen for new connections\n");
    exit(ret);
  }
}

/**
 * @brief Remove the socket_fd from the global array
 *
 * @param socket_fd the socket to remove
 */
void kick_peer(int socket_fd)
{
  for (int i = 0; i < clients; i++)
  {
    if (sockets[i] == socket_fd)
    {
      sockets[i] = sockets[clients - 1];
      clients--;
      break;
    }
  }
}

// This function is run whenever the user hits enter after typing a message
void input_callback(const char *message)
{
  if (strcmp(message, ":quit") == 0 || strcmp(message, ":q") == 0)
  {
    for (int i = 0; i < clients; i++)
    {
      close(sockets[i]);
    }
    free(sockets);
    ui_exit();
  }
  else
  {
    // length +2 to account for the ":" and the null terminator
    int length = strlen(message) + strlen(username) + 2;
    // Reconstruct message,
    char *new_buffer = malloc((length) * sizeof(char));
    snprintf(new_buffer, length, "%s:%s", username, message);
    // Use -1 for skip_fd to avoid skipping any peers
    send_message_to_peers(new_buffer, -1);
    ui_display(username, message);
    free(new_buffer);
  }
}

int main(int argc, char **argv)
{
  // Make sure the arguments include a username
  if (argc != 2 && argc != 4)
  {
    fprintf(stderr, "Usage: %s <username> [<peer> <port number>]\n", argv[0]);
    exit(1);
  }
  // Make space for the global sockets array
  sockets = malloc(sizeof(int) * size);

  // Open a server socket
  unsigned short port = 0;
  int server_socket_fd = server_socket_open(&port);
  if (server_socket_fd == -1)
  {
    perror("Server socket was not opened");
    exit(EXIT_FAILURE);
  }

  // Save the username in a global
  username = argv[1];

  // Did the user specify a peer we should connect to?
  if (argc == 4)
  {
    // Unpack arguments
    char *peer_hostname = argv[2];
    unsigned short peer_port = atoi(argv[3]);
    // Connect to the peer determined by the command line arguments
    int socket_fd = socket_connect(peer_hostname, peer_port);
    if (socket_fd == -1)
    {
      perror("Failed to connect");
      exit(EXIT_FAILURE);
    }
    accept_new_peer(socket_fd);
  }

  // Set up a server socket to accept incoming connections
  pthread_t server_thread;
  int ret = pthread_create(&server_thread, NULL, server_init, (void *)&server_socket_fd);
  if (ret != 0)
  {
    perror("Could not create server to listen for new connections\n");
    exit(ret);
  }

  // Set up the user interface. The input_callback function will be called
  // each time the user hits enter to send a message.
  ui_init(input_callback);
  char *display_message = malloc(sizeof(char) * 40);
  snprintf(display_message, 40, "Server is listening on port %d", port);
  // Once the UI is running, you can use it to display log messages
  ui_display("INFO", display_message);

  // Run the UI loop. This function only returns once we call ui_stop() somewhere in the program.
  ui_run();

  return 0;
}

/*
  Listen for new messages from a peer in a loop. When a message is received,
  send it to all other peers. On failure to read a message, kill the thread
  and modify the global array to remove the socket

  /param args: a pointer to the socket file descriptor to listen to
*/
void *server_listen(void *args)
{
  int client_socket_fd = *((int *)args);
  while (1)
  {
    // Read a message from the client
    char *message = receive_message(client_socket_fd);
    if (message == NULL)
    {
      kick_peer(client_socket_fd);
      return NULL;
    }
    // Send to all peers except the one we recieved the message from
    char *buffer = message;
    send_message_to_peers(message, client_socket_fd);
    char *username = strsep(&buffer, ":");
    ui_display(username, buffer);
    free(message);
  }
}

/*
  Listens for new incoming connects on the given socket. When a new connection is
  received, store the socket fd of the new peer and spawn a thread to listen for
  messages

  /param args: a pointer to a socket fd to listen for new connections
*/
void *server_init(void *args)
{
  int server_socket_fd = *((int *)args);
  // Start listening for connections, with a maximum of one queued connection
  if (listen(server_socket_fd, 1))
  {
    perror("listen failed");
    exit(EXIT_FAILURE);
  }
  // Listen for new clients in a loop
  while (1)
  {
    // Wait for a client to connect
    int client_socket_fd = server_socket_accept(server_socket_fd);
    if (client_socket_fd == -1)
    {
      perror("accept failed");
      exit(EXIT_FAILURE);
    }
    if (clients >= size)
    {
      size *= 2;
      sockets = realloc(socket, sizeof(int) * size);
    }
    accept_new_peer(client_socket_fd);
  }
  return NULL;
}