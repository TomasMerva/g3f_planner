#pragma once

#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdlib.h>     /* strtod */
#include <string>
#include <vector>
#include <sstream> 
#include <iostream>

enum class TCPClientStatus
{
    CONNECTED = 0,
    DISCONNECTED = -1,
    NEW_MESSAGE = 1,
    NO_MESSAGE = 2,
};


class TCPServer
{
    public:
        TCPServer(const uint port);
        TCPClientStatus WaitForClient();
        TCPClientStatus ReadRequest(std::vector<double>& tcp_recv_data);
        TCPClientStatus SendingResponse(const std::string &msg);
        void CloseServer();

    private:
        int _server_fd{0};
        int _client_socket{0};

        struct sockaddr_in _address;
        int _addrlen{sizeof(_address)};
        static const uint _buffer_size{1024};
        char _buffer[_buffer_size];
        

        bool _ListeningForConnections();
        bool _AcceptingClientConnection();



};



