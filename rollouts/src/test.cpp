#include "tcp_server.h"

int main()
{
    const uint PORT{8080};
    TCPServer server(PORT);

    while(1)
    {
        std::vector<double> data = server.WaitingForRequest();

        std::ostringstream oss;
        for (auto d: data)
        {
            std::cout << d << " ";
            oss << std::to_string(d);
        }
        //std::cout << oss.str() << "\n";
        server.SendingResponse(oss.str());
    }

    return 0;
}