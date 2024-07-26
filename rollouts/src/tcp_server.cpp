#include "tcp_server.h"


TCPServer::TCPServer(const uint port)
{
    // Creating socket file descriptor
    if ((_server_fd = socket(AF_INET, SOCK_STREAM, 0)) == 0) 
    {
        std::cerr << "Creating socket failed.\n";
        exit(EXIT_FAILURE);
    }

    // Setting up the address structure
    _address.sin_family = AF_INET;
    _address.sin_addr.s_addr = INADDR_ANY;
    _address.sin_port = htons(port);

    // Binding the socket to the port
    if (bind(_server_fd, (struct sockaddr*)&_address, sizeof(_address)) < 0) 
    {
        std::cerr << "Binding socket failed.\n";
        perror("binding");
        close(_server_fd);
        exit(EXIT_FAILURE);
    }

    std::cout << "Waiting for client...\n";
    // Listening for connections
    if (listen(_server_fd, 3) < 0) 
    {
        std::cerr << "Waiting for client failed.\n";
        perror("listening");
        close(_server_fd);
        exit(EXIT_FAILURE);
    }

    // Accepting a connection
    if ((_client_socket = accept(_server_fd, (struct sockaddr*)&_address, (socklen_t*)&_addrlen)) < 0) 
    {
        std::cerr << "Connection to client cannot be established.\n";
        perror("connecting");
        close(_server_fd);
        exit(EXIT_FAILURE);
    }
    std::cout << "Client connected.\n";
}


std::vector<double>
TCPServer::WaitingForRequest()
{
    memset(&_buffer, 0, sizeof(_buffer));
    _raw_request_data.clear();
    int bytes_read = read(_client_socket, _buffer, _buffer_size);
    if (bytes_read>0)
    {
            std::cout << "new data\n";

        std::stringstream ss(_buffer);
        std::string word;
        while (ss >> word)
        {
            _raw_request_data.push_back(std::stod(word));
            std::cout << word << " ";
        }
        std::cout << std::endl;
    }
    return _raw_request_data;
}

int
TCPServer::SendingResponse(const std::string &msg)
{
    memset(&_buffer, 0, sizeof(_buffer));
    std::strcpy(_buffer, msg.c_str()); 
    int result_flag = send(_client_socket, _buffer, strlen(_buffer), 0);
    return result_flag;
}