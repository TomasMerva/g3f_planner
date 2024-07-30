
#include "fabrics_server.h"



int main(int argc, char* argv[])
{
    const std::string CONFIG_FILE_PATH = (argc > 1) ? argv[1] : "../config/dingo_kinova_config.yaml";
    const uint PORT{8080};
    FabricsServer fabrics_server(PORT, CONFIG_FILE_PATH);



    while(1)
    {
        fabrics_server.Update();
    }
    
    // TCPServer server(PORT);

    // std::vector<double> tcp_recv_data;
    // while(1)
    // {
    //     std::ostringstream oss;

    //     TCPClientStatus TCP_status = server.ReadRequest(tcp_recv_data);
    //     if (TCP_status ==  TCPClientStatus::NEW_MESSAGE)
    //     {
    //         std::cout << tcp_recv_data.size() << "\n";
    //         for (auto d: tcp_recv_data)
    //         {
    //             // std::cout << d << " ";
    //             oss << std::to_string(d);
    //         }
    //         std::cout << std::endl;
    //     }
    //     else if (TCP_status == TCPClientStatus::DISCONNECTED)
    //     {
    //         if (server.WaitForClient() != TCPClientStatus::CONNECTED)
    //         {
    //             server.CloseServer();
    //         }
    //     }
        
    //     //std::cout << oss.str() << "\n";
    //     server.SendingResponse(oss.str());
    // }
    
    // server.CloseServer();
    return 0;
}