#include "fabrics_server.h"



int main(int argc, char* argv[])
{
    const std::string CONFIG_FILE_PATH = (argc > 1) ? argv[1] : "../../config/dingo_kinova_config.yaml";
    const uint PORT{8080};
    FabricsServer fabrics_server(PORT, CONFIG_FILE_PATH);

    while(1)
    {
        fabrics_server.Update();
    }
    
    return 0;
}