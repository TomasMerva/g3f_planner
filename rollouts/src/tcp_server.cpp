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
        TCPServer::CloseServer();
    }

    if (TCPServer::WaitForClient() != TCPClientStatus::CONNECTED)
    {
        TCPServer::CloseServer();
    }
    
}


TCPClientStatus 
TCPServer::ReadRequest(std::vector<double>& tcp_recv_data)
{
    memset(&_buffer, 0, sizeof(_buffer));
    tcp_recv_data.clear();
    int bytes_read = read(_client_socket, _buffer, _buffer_size);
    if (_buffer[0] == 'x')
    {
        std::cout << "Client disconnected.\n";
        close(_client_socket);
        return TCPClientStatus::DISCONNECTED;
    } 
    if (bytes_read>0)
    {
        std::cout << bytes_read << "\n";
        std::stringstream ss(_buffer);
        std::string word;
        while (ss >> word)
        {
            tcp_recv_data.push_back(std::stod(word));
        }
        return TCPClientStatus::NEW_MESSAGE;
    }
    return TCPClientStatus::NO_MESSAGE;
}

TCPClientStatus
TCPServer::SendingResponse(const std::string &msg)
{
    memset(&_buffer, 0, sizeof(_buffer));
    std::strcpy(_buffer, msg.c_str()); 
    if (send(_client_socket, _buffer, strlen(_buffer), 0) <= 0)
    {
        return TCPClientStatus::DISCONNECTED;
    }
    return TCPClientStatus::CONNECTED;
}

bool
TCPServer::_ListeningForConnections()
{
    std::cout << "Waiting for client...\n";
    if (listen(_server_fd, 1) < 0) 
    {
        std::cerr << "Waiting for client failed.\n";
        perror("listening");
        return false;
    }
    return true;
}

bool 
TCPServer::_AcceptingClientConnection()
{
    if ((_client_socket = accept(_server_fd, (struct sockaddr*)&_address, (socklen_t*)&_addrlen)) < 0) 
    {
        std::cerr << "Connection to client cannot be established.\n";
        perror("connecting");
        return false;
    }
    std::cout << "Client connected.\n";
    return true;
}


TCPClientStatus 
TCPServer::WaitForClient()
{
    if(! TCPServer::_ListeningForConnections())
    {
        return TCPClientStatus::DISCONNECTED;
    }

    if(! TCPServer::_AcceptingClientConnection())
    {
        return TCPClientStatus::DISCONNECTED;
    }
    return TCPClientStatus::CONNECTED;
}

void
TCPServer::CloseServer()
{
    close(_server_fd);
    exit(EXIT_FAILURE);
}