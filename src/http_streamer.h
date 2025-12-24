#ifndef HTTP_STREAMER_H
#define HTTP_STREAMER_H

#include <vector>
#include <string>
#include <memory>
#include <atomic>
#include <mutex>
#include <set>
#include "CivetServer.h"

class HttpStreamer : public CivetHandler, public CivetWebSocketHandler {
public:
    HttpStreamer(const std::vector<std::string>& options);
    ~HttpStreamer();

    void start();
    void stop();
    void pushH264Data(const std::vector<uint8_t>& data);

    // CivetHandler interface
    bool handleGet(CivetServer *server, struct mg_connection *conn) override;

    // CivetWebSocketHandler interface
    bool handleConnection(CivetServer *server, const struct mg_connection *conn) override;
    void handleReadyState(CivetServer *server, struct mg_connection *conn) override;
    bool handleData(CivetServer *server, struct mg_connection *conn, int bits, char *data, size_t data_len) override;
    void handleClose(CivetServer *server, const struct mg_connection *conn) override;

private:
    std::unique_ptr<CivetServer> server_;
    std::vector<std::string> options_;
    
    std::mutex connections_mutex_;
    std::set<struct mg_connection*> connections_;
    
    std::atomic<bool> running_;
};

#endif // HTTP_STREAMER_H
