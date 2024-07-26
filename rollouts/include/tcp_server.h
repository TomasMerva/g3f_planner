#pragma once

#include <cstring>
#include <unistd.h>
#include <arpa/inet.h>
#include <stdlib.h>     /* strtod */
#include <string>
#include <vector>
#include <sstream> 
#include <iostream>

class TCPServer
{
    public:
        TCPServer(const uint port);
        std::vector<double> WaitingForRequest();
        int SendingResponse(const std::string &msg);

    private:
        int _server_fd{0};
        int _client_socket{0};

        struct sockaddr_in _address;
        int _addrlen{sizeof(_address)};
        static const uint _buffer_size{1024};
        char _buffer[_buffer_size];
        
        std::vector<double> _raw_request_data;


};



